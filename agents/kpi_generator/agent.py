from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any, Literal
from datetime import date
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from agents.kpi_generator.models import Task, Grade, TeamKpi, Employee
import re
from collections import defaultdict


# --- 1. Pydantic 모델 정의 ---

class TaskOutput(BaseModel):
    task_id: int
    start_date: date
    end_date: date
    task_name: str
    task_detail: str
    target_level: str
    weight: int
    team_kpi_id: int
    emp_no: str
    model_config = ConfigDict(from_attributes=True)


class GradeOutput(BaseModel):
    grade_id: int
    grade_s: str
    grade_a: str
    grade_b: str
    grade_c: str
    grade_d: str
    grade_rule: str
    task_id: Optional[int]
    model_config = ConfigDict(from_attributes=True)


class GeneratedKpiResponse(BaseModel):
    tasks: List[TaskOutput]
    grades: List[GradeOutput]


class KpiPlan(BaseModel):
    team_kpi_id: int = Field(..., description="분석 대상 팀 KPI의 ID")
    kpi_type: Literal["quantitative", "qualitative"] = Field(...,
                                                             description="KPI 유형 ('quantitative' 또는 'qualitative')")
    participants: List[str] = Field(..., description="해당 KPI를 담당할 직원(emp_no) 목록")


class KpiAssignmentResult(BaseModel):
    assignments: List[KpiPlan]

class GradeCriteriaLLM(BaseModel):
    grade_s: str = Field(..., description="S등급 목표 및 기준")
    grade_a: str = Field(..., description="A등급 목표 및 기준")
    grade_b: str = Field(..., description="B등급 목표 및 기준")
    grade_c: str = Field(..., description="C등급 목표 및 기준")
    grade_d: str = Field(..., description="D등급 목표 및 기준")
    grade_rule: str = Field(..., description="전체 등급에 대한 평가 규칙 또는 설명")

class FinalIndividualKpi(BaseModel):
    task_name: str = Field(..., description="개인 KPI의 이름")
    task_detail: str = Field(..., description="개인 KPI의 상세 설명")
    target_level: str = Field(..., description="A등급 이상을 받기 위한 핵심 목표 수준")
    weight: int = Field(..., description="이 KPI의 가중치 (한 직원의 모든 KPI 가중치 합은 100)")
    linked_team_kpi_id_original: int = Field(..., description="이 KPI가 파생된 원본 팀 KPI의 ID")
    grade_criteria: GradeCriteriaLLM = Field(..., description="개인화된 5단계 등급 기준")

class FinalKpiResult(BaseModel):
    kpis: List[FinalIndividualKpi]

# --- 2. LangGraph 상태 정의 ---

class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    config: Dict[str, Any]
    team_id: int
    team_emp_nos: List[str]
    employee_profiles: Dict[str, Any]
    team_kpis_raw: List[Dict[str, Any]]
    team_kpi_grade_criteria: Dict[int, Dict[str, Any]]
    past_kpis_by_emp: Dict[str, List[Dict[str, Any]]]
    kpi_assignments: List[KpiPlan]
    kpi_drafts_by_emp: Dict[str, List[Dict[str, Any]]]
    persisted_tasks: List[Dict[str, Any]]
    persisted_grades: List[Dict[str, Any]]
    kpi_validation_status: Optional[str]
    rdb_persistence_status: Optional[str]


# --- 3. LangGraph 노드 정의 ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def extract_num_and_unit(s: str) -> (float, str):
    """문자열에서 숫자와 그 뒤에 오는 단위를 함께 추출합니다."""
    if not isinstance(s, str):
        return 0.0, ""

    s_no_comma = s.replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", s_no_comma)

    if not match:
        return 0.0, ""

    num = float(match.group(1))
    # 숫자 바로 뒤부터의 문자열을 단위로 간주
    unit = s_no_comma[match.end():].lstrip()
    # "달성", "이상" 같은 단어는 제외하고 핵심 단위만 추출 (선택적)
    unit = re.sub(r"(달성|이상|이하|미만|초과).*", "", unit).strip()

    return num, unit


async def retrieve_data_node(state: AgentState) -> AgentState:
    print(f"Executing retrieve_data_node for team {state['team_id']}...")
    db_session: AsyncSession = state["config"]["db_session"]
    team_id = state["team_id"]

    current_year = date.today().year

    emp_res = await db_session.execute(
        select(Employee.emp_no, Employee.salary, Employee.cl, Employee.position)
        .where(Employee.team_id == team_id, Employee.role == "MEMBER")
    )
    rows = emp_res.all()
    team_emp_nos = [r.emp_no for r in rows]
    employee_profiles = {r.emp_no: {"salary": r.salary, "cl": r.cl, "job_role": r.position} for r in rows}

    if not team_emp_nos:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found or has no members.")

    kpi_res = await db_session.execute(
        select(TeamKpi).where(
            TeamKpi.team_id == team_id,
            TeamKpi.year == current_year
        )
    )
    team_kpis_raw = [kpi.__dict__ for kpi in kpi_res.scalars().all()]

    team_kpi_ids = [k['team_kpi_id'] for k in team_kpis_raw]
    grade_res = await db_session.execute(
        select(Grade).where(Grade.task_id.is_(None), Grade.team_kpi_id.in_(team_kpi_ids)))
    team_kpi_grade_criteria = {g.team_kpi_id: g.__dict__ for g in grade_res.scalars().all()}

    past_kpis_by_emp = defaultdict(list)
    if team_emp_nos:
        past_kpi_res = await db_session.execute(
            select(Task.emp_no, Task.task_name, Task.task_detail, Grade.grade_s)
            .join(Grade, Grade.task_id == Task.task_id)
            .where(Task.emp_no.in_(team_emp_nos))
        )
        for row in past_kpi_res.all():
            past_kpis_by_emp[row.emp_no].append({
                "task_name": row.task_name,
                "task_detail": row.task_detail,
                "achieved_target": row.grade_s,
            })

    return {
        **state,
        "team_emp_nos": team_emp_nos,
        "employee_profiles": employee_profiles,
        "team_kpis_raw": team_kpis_raw,
        "team_kpi_grade_criteria": team_kpi_grade_criteria,
        "past_kpis_by_emp": dict(past_kpis_by_emp),
    }


async def classify_and_assign_kpis_node(state: AgentState) -> AgentState:
    print("Executing classify_and_assign_kpis_node...")
    structured_llm = llm.with_structured_output(KpiAssignmentResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 HR 성과관리 전문가입니다. 주어진 팀 KPI 목록과 팀원 정보를 보고, 각 KPI에 대해 다음 작업을 수행하세요.
        1. KPI 목표가 숫자로 측정 가능하면 "quantitative", 과정/품질 중심이면 "qualitative"로 분류하세요.
        2. KPI 내용, 직원 프로필, 그리고 **과거 성과(past_kpis_by_emp)를 종합적으로 고려**하여 가장 적합한 담당자를 2명 이상 지정하세요. 모든 KPI에는 최소 2명의 담당자가 있어야 합니다. 모든 팀원은 최소 2개 이상의 KPI를 할당받아야 합니다. 단, 각 직원의 직무(job_role)와 역량에 가장 부합하는 KPI를 우선적으로 할당해 주세요. 만약 어떤 직원에게 적합한 정량적 KPI가 없다면, 그 직원에게는 팀 공통의 정성적 KPI를 할당하여 모든 직원이 최소 2개의 목표를 갖도록 해주세요."
        3. 결과를 반드시 KpiAssignmentResult 스키마에 맞는 JSON으로만 출력하세요.

        - 팀 KPI 목록: {team_kpis}
        - 팀원 정보: {employee_profiles}
        - 팀원별 과거 성과: {past_kpis_by_emp}
        
        최종 확인: 위에 생성한 KPI 목록을 보고, 할당되지 않은 팀원이 있는지 최종적으로 확인하세요. 만약 누락된 직원이 있다면, 그래도 그들에게 가장 적합한 팀 공통 목표(예: 팀 운영 효율성)를 추가 할당해서 최종 결과를 다시 만들어주세요.
        
        """),
        ("human", "위 내용을 바탕으로 KPI를 분류하고 담당자를 지정해주세요."),
    ])
    response = await (prompt | structured_llm).ainvoke({
        "team_kpis": state["team_kpis_raw"],
        "employee_profiles": state["employee_profiles"],
        "past_kpis_by_emp": state["past_kpis_by_emp"],
    })
    return {**state, "kpi_assignments": response.assignments}


async def generate_personalized_kpis_node(state: AgentState) -> AgentState:
    print("Executing generate_personalized_kpis_node...")
    structured_llm = llm.with_structured_output(FinalKpiResult)

    current_year = date.today().year

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 유능한 HR 리더입니다. 아래 제공된 직원의 정보와 그에게 할당된 팀 목표들을 바탕으로, {year}년 개인 성과 목표(KPI) 세트를 생성해주세요.

        [규칙]
        1. 할당된 각 팀 목표(`assigned_team_kpis`)에 대해 개인화된 KPI를 1개씩 생성합니다.
        2. 생성된 모든 개인 KPI의 가중치(`weight`) 합은 100이 되도록 분배해주세요.
        3. `grade_criteria`는 S,A,B,C,D 각 등급이 명확히 구분되도록 설정해야 합니다.
        4. **정량적 목표**는 팀 목표의 수치를 직원의 역량에 맞게 스케일링하여 할당하고, 등급 기준도 그에 맞게 설정해주세요.
        5. **정성적 목표**는 수치 대신, 과정 중심의 단계별 목표를 등급 기준으로 제시해주세요. (예: S등급 '전략 승인', A등급 '전략 수립 완료', B등급 '초안 80% 완성')
        6. 최종 결과는 반드시 `FinalKpiResult` 스키마에 맞는 JSON 형식으로만 출력해주세요.

        **[입력 정보]**
        - 직원 프로필: {employee_profile}
        - 직원의 과거 성과: {past_kpis}
        - 이 직원에게 할당된 올해의 팀 KPI 목록 (원본 등급 기준 포함): {assigned_team_kpis}
        """),
        ("human", "위 정보를 바탕으로 이 직원의 최종 개인 KPI 세트를 생성해주세요."),
    ])

    drafts_by_emp = {}
    team_kpis_map = {tk["team_kpi_id"]: tk for tk in state["team_kpis_raw"]}

    # 직원별로 할당된 팀 KPI들을 그룹화
    assigned_team_kpis_by_emp = defaultdict(list)
    for plan in state["kpi_assignments"]:
        for emp_no in plan.participants:
            team_kpi_info = team_kpis_map.get(plan.team_kpi_id, {})
            if team_kpi_info:
                team_kpi_info["original_grades"] = state["team_kpi_grade_criteria"].get(plan.team_kpi_id, {})
                assigned_team_kpis_by_emp[emp_no].append(team_kpi_info)

    # 직원별로 LLM 호출하여 개인화된 KPI 생성
    for emp_no, assigned_kpis in assigned_team_kpis_by_emp.items():
        if not assigned_kpis: continue

        response = await (prompt | structured_llm).ainvoke({
            "year": current_year,
            "employee_profile": state["employee_profiles"][emp_no],
            "past_kpis": state["past_kpis_by_emp"].get(emp_no, []),
            "assigned_team_kpis": assigned_kpis,
        })

        drafts_by_emp[emp_no] = [kpi.model_dump() for kpi in response.kpis]

    return {**state, "kpi_drafts_by_emp": drafts_by_emp}


async def validate_kpis_node(state: AgentState) -> AgentState:
    print("Executing advanced validate_kpis_node…")
    drafts_by_emp = state["kpi_drafts_by_emp"]
    all_errors = []

    # --- 1단계: 규칙 기반 검증 ---
    for emp_no, drafts in drafts_by_emp.items():
        if not drafts: continue

        # 가중치 합계 검증
        total_weight = sum(d.get("weight", 0) for d in drafts)
        if total_weight != 100:
            all_errors.append(f"RuleError [{emp_no}]: 가중치 총합이 {total_weight}입니다 (100이 아님).")

        # 개별 KPI 검증
        for i, d in enumerate(drafts):
            if not all(k in d for k in ["task_name", "weight", "grade_criteria"]):
                all_errors.append(f"RuleError [{emp_no}-{i}]: 필수 키가 누락되었습니다.")
            if not (isinstance(d.get("weight"), int) and d.get("weight", 0) > 0):
                all_errors.append(f"RuleError [{emp_no}-{i}]: weight는 양의 정수여야 합니다.")

    # 규칙 기반 에러가 있으면 바로 리비전 요청
    if all_errors:
        print("Validation Errors (Rule-based):", all_errors)
        # 자동 수정 루프를 위해 상태를 'needs_revision'으로 설정 가능
        return {**state, "kpi_validation_status": "needs_revision",
                "messages": state["messages"] + [AIMessage(content="\n".join(all_errors))]}

    # --- 2단계: LLM을 활용한 논리적/정성적 검증 ---
    print("Executing LLM-based validation...")
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 엄격한 HR 시니어 매니저입니다. 한 직원의 연간 KPI 포트폴리오를 검토하고, 아래 기준에 따라 평가해주세요.
        - **현실성**: 직원의 프로필(직무, 연봉)에 비해 목표가 비현실적으로 높거나 낮지 않습니까?
        - **명확성**: 각 KPI의 목표가 모호하지 않고 구체적입니까?
        - **역할 부합도**: KPI들이 직원의 핵심 역할과 잘 연계되어 있습니까?

        검토 후, 치명적인 문제가 있다면 "PROBLEM:"으로 시작하는 한 문장의 피드백을, 문제가 없다면 "OK"를 반환해주세요.

        **[직원 프로필]**
        {employee_profile}

        **[생성된 KPI 포트폴리오]**
        {kpi_drafts}
        """),
        ("human", "위 KPI 포트폴리오를 검토하고 평가해주세요."),
    ])

    validation_chain = validation_prompt | llm

    for emp_no, drafts in drafts_by_emp.items():
        if not drafts: continue

        feedback = await validation_chain.ainvoke({
            "employee_profile": state["employee_profiles"][emp_no],
            "kpi_drafts": drafts
        })

        if "PROBLEM:" in feedback.content:
            all_errors.append(f"LLMValidationError [{emp_no}]: {feedback.content}")

    if all_errors:
        print("Validation Errors (LLM-based):", all_errors)
        return {**state, "kpi_validation_status": "needs_revision",
                "messages": state["messages"] + [AIMessage(content="\n".join(all_errors))]}

    # 모든 검증 통과
    print("All KPIs passed validation.")
    return {**state, "kpi_validation_status": "valid"}

async def persist_to_rdb_node(state: AgentState) -> AgentState:
    print("Executing persist_to_rdb_node...")
    db_session: AsyncSession = state["config"]["db_session"]

    saved_tasks = []
    saved_grades = []

    for emp_no, drafts in state["kpi_drafts_by_emp"].items():
        for d in drafts:
            # 1. Task 생성
            task = Task(
                start_date=date.today().replace(month=1, day=1),
                end_date=date.today().replace(month=12, day=31),
                emp_no=emp_no,
                team_kpi_id=d["linked_team_kpi_id_original"],
                task_name=d["task_name"],
                task_detail=d["task_detail"],
                target_level=d["target_level"],
                weight=d["weight"]
            )
            db_session.add(task)
            await db_session.flush()

            # 2. Grade 생성 (이제 모든 Task에 대해 Grade를 생성)
            criteria = d["grade_criteria"]

            grade = Grade(
                task_id=task.task_id,
                team_kpi_id=None,
                **criteria
            )
            db_session.add(grade)

            await db_session.flush()
            saved_tasks.append(TaskOutput.model_validate(task).model_dump())
            saved_grades.append(GradeOutput.model_validate(grade).model_dump())

    return {**state, "persisted_tasks": saved_tasks, "persisted_grades": saved_grades,
            "rdb_persistence_status": "success"}

# --- 4. LangGraph 워크플로우 정의 ---
def create_kpi_generation_graph():
    workflow = StateGraph(AgentState)

    # 사용할 노드들을 정의
    workflow.add_node("retrieve_data", retrieve_data_node)
    workflow.add_node("classify_and_assign_kpis", classify_and_assign_kpis_node)
    workflow.add_node("generate_personalized_kpis", generate_personalized_kpis_node)  # 신규 노드
    workflow.add_node("validate_kpis", validate_kpis_node)  # 가중치 합계 등 기본 검증은 유지
    workflow.add_node("persist_to_rdb", persist_to_rdb_node)

    # 워크플로우 엣지(흐름)를 새롭게 정의
    workflow.set_entry_point("retrieve_data")
    workflow.add_edge("retrieve_data", "classify_and_assign_kpis")
    workflow.add_edge("classify_and_assign_kpis", "generate_personalized_kpis")  # 핵심 생성 노드로 연결
    workflow.add_edge("generate_personalized_kpis", "validate_kpis")
    workflow.add_edge("validate_kpis", "persist_to_rdb")
    workflow.add_edge("persist_to_rdb", END)

    return workflow.compile()

kpi_graph = create_kpi_generation_graph()

async def run_kpi_generation_for_team(team_id: int, db: AsyncSession) -> dict:
    initial_state = AgentState(
        messages=[],
        team_id=team_id,
        config={"db_session": db}
    )
    final_state = await kpi_graph.ainvoke(initial_state)
    return {
        "tasks": final_state.get("persisted_tasks", []),
        "grades": final_state.get("persisted_grades", []),
    }