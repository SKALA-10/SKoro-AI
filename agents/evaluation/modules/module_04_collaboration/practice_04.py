# %% [markdown]
# # Module 3 - 협업 분석 모듈 구현
# 
# 이 노트북은 AI 성과관리 시스템의 모듈 3 (협업 분석 모듈)을 구현합니다.
# 
# ## 주요 기능
# - Task Summary 기반 협업 관계 분석
# - 팀 협업 네트워크 매트릭스 생성
# - 협업 편중도 신뢰성 있는 판단
# - JSON 형태로 team_evaluations.ai_collaboration_matrix에 저장

# %%
# 필요한 라이브러리 import
from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END
import json
import re
from dotenv import load_dotenv
# 환경 변수 로드
load_dotenv()

import sys
import os
import json
import re
import time
import logging
from typing import Dict, List, Optional, Any, Literal, TypedDict
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime

# 환경 설정 (practice.py와 동일하게)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))  # 4단계로 수정
sys.path.append(project_root)

from config.settings import DatabaseConfig
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row

# 로깅 설정 (필요시)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx의 INFO 레벨 로그 비활성화
logging.getLogger("httpx").setLevel(logging.WARNING)

# DB 설정
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# %%
# Module3AgentState 정의
class Module3AgentState(TypedDict):
    """
    모듈 3 (협업 분석 모듈)의 내부 상태를 정의합니다.
    이 상태는 모듈 3 내의 모든 서브모듈이 공유하고 업데이트합니다.
    """
    messages: Annotated[List[HumanMessage], operator.add]
    report_type: Literal["quarterly", "annual"]
    team_id: int
    period_id: int
    target_task_summary_ids: List[int]
    target_team_kpi_ids: List[int]
    team_evaluation_id: int
    collaboration_relationships: List[Dict]
    individual_collaboration_analysis: Dict[str, Dict]
    team_collaboration_matrix: Dict

# %%
# DB 유틸리티 함수들
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from typing import Optional, List, Dict, Any
import sys
import os

# 설정 파일 경로 (실제 환경에 맞게 수정 필요)
# from config.settings import DatabaseConfig
# db_config = DatabaseConfig()
# DATABASE_URL = db_config.DATABASE_URL

# 임시 DB 설정 (실제 환경에서는 위 코드 사용)


def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환합니다."""
    if row is None:
        return {}
    return row._asdict()

# %%
# 모듈 3 전용 DB 함수들
def fetch_collaboration_tasks_by_kpi(team_kpi_id: int, period_id: int) -> List[Dict]:
    """KPI별 협업 가능한 Task들을 조회합니다."""
    with engine.connect() as connection:
        query = text("""
            SELECT t.task_id, t.task_name, t.emp_no, t.start_date, t.end_date,
                   ts.task_summary, ts.task_summary_Id, 
                   ts.ai_contribution_score, ts.ai_analysis_comment_task,
                   e.emp_name
            FROM tasks t
            JOIN task_summaries ts ON t.task_id = ts.task_id
            JOIN employees e ON t.emp_no = e.emp_no
            WHERE t.team_kpi_id = :team_kpi_id AND ts.period_id = :period_id
            ORDER BY t.start_date
        """)
        results = connection.execute(query, {"team_kpi_id": team_kpi_id, "period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]

def fetch_peer_talk_summary(emp_no: str, period_id: int, report_type: str) -> Optional[str]:
    """개인의 Peer Talk 요약을 조회합니다."""
    with engine.connect() as connection:
        if report_type == "quarterly":
            # feedback_reports에서 조회 (분기별)
            query = text("""
                SELECT fr.ai_peer_talk_summary
                FROM feedback_reports fr
                JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                WHERE fr.emp_no = :emp_no AND te.period_id = :period_id
                LIMIT 1
            """)
        else:  # annual
            # final_evaluation_reports에서 조회 (연말)
            query = text("""
                SELECT fer.ai_peer_talk_summary
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
                LIMIT 1
            """)
        
        result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).scalar()
        return result

def fetch_team_members_with_tasks(team_id: int, period_id: int) -> List[Dict]:
    """팀원들과 그들의 Task 통계를 조회합니다. (팀장 제외)"""
    with engine.connect() as connection:
        query = text("""
            SELECT e.emp_no, e.emp_name, e.role,
                   COUNT(DISTINCT t.task_id) as total_task_count,
                   AVG(ts.ai_contribution_score) as avg_contribution_score
            FROM employees e
            LEFT JOIN tasks t ON e.emp_no = t.emp_no
            LEFT JOIN task_summaries ts ON t.task_id = ts.task_id AND ts.period_id = :period_id
            WHERE e.team_id = :team_id AND e.role != 'MANAGER'
            GROUP BY e.emp_no, e.emp_name, e.role
        """)
        results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]



def save_collaboration_matrix_to_db(team_evaluation_id: int, collaboration_matrix: Dict) -> bool:
    """협업 매트릭스를 team_evaluations 테이블의 ai_collaboration_matrix 컬럼에 저장합니다."""
    
    # Decimal을 float로 변환하는 함수
    def convert_decimal(obj):
        if isinstance(obj, dict):
            return {k: convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimal(v) for v in obj]
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'Decimal':
            return float(obj)
        else:
            return obj
    
    # Decimal 변환 후 JSON으로 직렬화
    converted_matrix = convert_decimal(collaboration_matrix)
    collaboration_matrix_json = json.dumps(converted_matrix, ensure_ascii=False)
    
    with engine.connect() as connection:
        try:
            query = text("""
                UPDATE team_evaluations 
                SET ai_collaboration_matrix = :collaboration_matrix
                WHERE team_evaluation_id = :team_evaluation_id
            """)
            
            connection.execute(query, {
                "collaboration_matrix": collaboration_matrix_json,
                "team_evaluation_id": team_evaluation_id
            })
            connection.commit()
            
            print(f"협업 매트릭스 저장 성공: team_evaluation_id={team_evaluation_id}")
            print(f"저장된 JSON 데이터:")
            print(json.dumps(converted_matrix, ensure_ascii=False, indent=2))
            
            return True
            
        except Exception as e:
            print(f"협업 매트릭스 저장 실패: {e}")
            connection.rollback()
            return False


def fetch_team_kpi_progress(team_kpi_ids: List[int], period_id: int) -> List[Dict]:
    """팀 KPI 진행률 정보를 조회합니다."""
    if not team_kpi_ids:
        return []
    
    kpi_ids_str = ','.join(map(str, team_kpi_ids))
    
    with engine.connect() as connection:
        query = text(f"""
            SELECT tk.team_kpi_id, tk.kpi_name, tk.target_value, tk.current_value,
                   tk.ai_kpi_progress_rate, tk.ai_kpi_analysis_comment
            FROM team_kpis tk
            WHERE tk.team_kpi_id IN ({kpi_ids_str}) AND tk.period_id = :period_id
        """)
        results = connection.execute(query, {"period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]

def fetch_feedback_report_data(emp_no: str, period_id: int) -> Optional[Dict]:
    """개인의 분기별 피드백 리포트 데이터를 조회합니다."""
    with engine.connect() as connection:
        query = text("""
            SELECT fr.contribution_rate, fr.ai_overall_contribution_summary_comment,
                   fr.ai_peer_talk_summary
            FROM feedback_reports fr
            JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
            WHERE fr.emp_no = :emp_no AND te.period_id = :period_id
            LIMIT 1
        """)
        result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
        return row_to_dict(result) if result else None

# %%
# LLM 유틸리티 함수들
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LLM 클라이언트 초기화
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답 텍스트에서 ```json ... ``` 블록만 추출합니다."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def _get_llm_content(response):
    content = response.content
    if isinstance(content, list):
        # 리스트면 문자열만 추출
        content = "\n".join([str(x) for x in content if isinstance(x, str)])
    return content

# %%
def call_llm_for_collaboration_detection(task_summary: str, task_name: str, potential_collaborators: List[str], emp_name: str) -> Dict:
    """Task Summary에서 실제 협업 관계를 감지합니다."""
    
    system_prompt = """
    당신은 SK 조직의 업무 협업 분석 전문가입니다.
    주어진 Task Summary 내용을 분석하여 실제로 다른 동료와 협업했는지 판단하고,
    협업한 경우 구체적으로 누구와 협업했는지 식별해주세요.

    분석 기준:
    - "함께", "협력", "지원", "도움", "협업", "공동" 등의 키워드 존재
    - 다른 사람의 이름이나 역할 언급
    - 회의, 논의, 검토 등 상호작용 활동 언급
    - 단순한 보고나 개별 작업은 협업으로 간주하지 않음

    결과는 JSON 형식으로만 응답해주세요.
    """

    potential_collaborators_str = ", ".join(potential_collaborators) if potential_collaborators else "없음"
    
    human_prompt = f"""
    <분석 대상 Task>
    담당자: {emp_name}
    Task 이름: {task_name}
    Task 요약: {task_summary}
    </분석 대상 Task>

    <잠재적 협업자 목록>
    {potential_collaborators_str}
    </잠재적 협업자 목록>

    JSON 응답:
    {{
        "is_collaboration": [true/false - 실제 협업 여부],
        "collaborators": ["협업자 사번 리스트"],
        "collaboration_description": "[협업 내용 간단 설명]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = _get_llm_content(response)
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        is_collaboration = llm_parsed_data.get("is_collaboration", False)
        collaborators = llm_parsed_data.get("collaborators", [])
        description = llm_parsed_data.get("collaboration_description", "")
        
        return {
            "is_collaboration": is_collaboration,
            "collaborators": collaborators if isinstance(collaborators, list) else [],
            "description": description
        }
        
    except Exception as e:
        print(f"LLM 협업 감지 오류: {e}")
        return {"is_collaboration": False, "collaborators": [], "description": "분석 실패"}

# %%
def call_llm_for_team_role_analysis(task_summaries: List[str], emp_name: str, emp_no: str) -> Dict:
    """개인의 Task Summary들을 종합하여 팀 내 역할을 분석합니다."""
    
    task_summaries_text = "\n".join([f"- {summary}" for summary in task_summaries])
    
    system_prompt = """
    당신은 SK 조직의 역할 분석 전문가입니다.
    개인의 Task Summary들을 분석하여 해당 직원의 주요 업무 내용과 팀 내 역할 유형을 파악해주세요.

    역할 유형 분류:
    - 핵심 개발자: 주요 기술 개발 담당
    - 기획 리더: 프로젝트 기획 및 방향성 설정
    - 조율자: 팀 간 협업 및 일정 관리
    - 품질 관리자: 테스트, 검증, 품질 보증
    - 분석 전문가: 데이터 분석, 리서치
    - 지원 전문가: 기술 지원, 문제 해결
    - 독립형 전문가: 전문 영역 독립 수행

    결과는 JSON 형식으로만 응답해주세요.
    """

    human_prompt = f"""
    <분석 대상>
    이름: {emp_name}
    사번: {emp_no}
    Task Summary 목록:
    {task_summaries_text}
    </분석 대상>

    JSON 응답:
    {{
        "main_work_content": "[주요 업무 내용 (한줄)]",
        "role_type": "[역할 유형]",
        "team_role": "[주요 업무 내용 | 역할 유형]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = _get_llm_content(response)
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        return {
            "main_work_content": llm_parsed_data.get("main_work_content", "업무 내용 분석 실패"),
            "role_type": llm_parsed_data.get("role_type", "분석 실패"),
            "team_role": llm_parsed_data.get("team_role", "분석 실패")
        }
        
    except Exception as e:
        print(f"LLM 역할 분석 오류: {e}")
        return {
            "main_work_content": "분석 실패",
            "role_type": "분석 실패", 
            "team_role": "분석 실패"
        }

# %%
def call_llm_for_collaboration_bias_analysis(collaboration_data: Dict, emp_name: str, emp_no: str) -> Dict:
    """협업 편중도를 신뢰성 있게 분석합니다."""
    
    total_tasks = collaboration_data.get("total_tasks", 0)
    collaboration_tasks = collaboration_data.get("collaboration_tasks", 0)
    collaborator_counts = collaboration_data.get("collaborator_counts", {})
    dependency_metrics = collaboration_data.get("dependency_metrics", {})
    
    system_prompt = """
    당신은 SK 조직의 협업 편중도 분석 전문가입니다.
    제공된 협업 데이터를 기반으로 해당 직원의 협업 편중도를 분석해주세요.

    분석 기준:
    1. 높음(과의존 위험): 
       - 다른 팀원들이 이 사람에게 과도하게 의존
       - 핵심 업무를 혼자 담당하여 병목 위험
       - 협업 비율이 매우 높고 다른 사람들의 업무 진행에 필수적
    
    2. 보통(적절한 분산):
       - 협업과 독립 업무의 균형이 적절
       - 대체 가능한 구조로 운영
       - 다른 팀원들과 고르게 협업
    
    3. 낮음(협업 부족):
       - 협업 참여도가 낮음
       - 다른 팀원들과의 연결점 부족
       - 주변부 역할만 담당

    결과는 JSON 형식으로만 응답해주세요.
    """

    collaboration_info = f"""
    총 Task 수: {total_tasks}
    협업 Task 수: {collaboration_tasks}
    협업률: {(collaboration_tasks/total_tasks*100) if total_tasks > 0 else 0:.1f}%
    협업자별 횟수: {collaborator_counts}
    의존도 지표: {dependency_metrics}
    """

    human_prompt = f"""
    <분석 대상>
    이름: {emp_name}
    사번: {emp_no}
    협업 데이터:
    {collaboration_info}
    </분석 대상>

    JSON 응답:
    {{
        "bias_level": "[높음/보통/낮음]",
        "bias_description": "[편중도 상세 설명]",
        "risk_assessment": "[위험도 평가 및 이유]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = _get_llm_content(response)
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        bias_level = llm_parsed_data.get("bias_level", "보통")
        if bias_level not in ["높음", "보통", "낮음"]:
            bias_level = "보통"
            
        return {
            "bias_level": bias_level,
            "bias_description": llm_parsed_data.get("bias_description", ""),
            "risk_assessment": llm_parsed_data.get("risk_assessment", "")
        }
        
    except Exception as e:
        print(f"LLM 편중도 분석 오류: {e}")
        return {
            "bias_level": "보통",
            "bias_description": "분석 실패",
            "risk_assessment": "분석 실패"
        }

# %%
def call_llm_for_peer_talk_summary(peer_talk_content: str, emp_name: str) -> str:
    """Peer Talk 내용을 한 줄로 요약합니다."""
    
    system_prompt = """
    당신은 SK 조직의 동료평가 요약 전문가입니다.
    주어진 Peer Talk 내용(강점, 우려, 협업 관찰)을 핵심만 추려서 한 줄로 요약해주세요.
    
    요약 형식: "핵심 강점 키워드, 주요 우려사항"
    예시: "리더십 강함, 완벽주의 성향 주의"

    따옴표 없이 내용만 응답해주세요.
    """

    human_prompt = f"""
    <Peer Talk 내용>
    {peer_talk_content}
    </Peer Talk 내용>

    한 줄 요약:
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client


    try:
        response = chain.invoke({})
        summary = _get_llm_content(response).strip()
        
        # 따옴표 제거 처리
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        
        return summary if summary else f"{emp_name} 동료평가 요약 없음"
        
    except Exception as e:
        print(f"LLM Peer Talk 요약 오류: {e}")
        return f"{emp_name} 동료평가 분석 실패"



# %%
def call_llm_for_overall_evaluation(collaboration_analysis: Dict, emp_name: str, emp_no: str) -> str:
    """개인의 종합 협업 평가를 생성합니다."""
    
    system_prompt = """
    당신은 SK 조직의 협업 평가 전문가입니다.
    제공된 협업 분석 데이터를 바탕으로 해당 직원의 협업 스타일과 기여도를 포함한
    2-3줄의 간단한 종합 평가를 작성해주세요.

    포함 요소:
    - 협업 스타일 (리더형, 서포터형, 독립형 등)
    - 팀 기여도 (높음, 보통, 개선 필요 등)
    - 개선 제안 (필요시)
    """

    analysis_text = f"""
    팀 내 역할: {collaboration_analysis.get('team_role', '')}
    협업률: {collaboration_analysis.get('collaboration_rate', 0)}%
    핵심 협업자: {', '.join(collaboration_analysis.get('key_collaborators', []))}
    협업 편중도: {collaboration_analysis.get('collaboration_bias', '')}
    """

    human_prompt = f"""
    <협업 분석 데이터>
    이름: {emp_name}({emp_no})
    {analysis_text}
    </협업 분석 데이터>

    종합 평가 (2-3줄):
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        evaluation = _get_llm_content(response).strip()
        return evaluation if evaluation else f"{emp_name}({emp_no})님의 협업 평가를 완료하지 못했습니다."
        
    except Exception as e:
        print(f"LLM 종합 평가 오류: {e}")
        return f"{emp_name}({emp_no})님의 협업 분석에 오류가 발생했습니다."

def call_llm_for_team_summary(collaboration_matrix: List[Dict], team_summary_stats: Dict) -> str:
    """팀 협업 매트릭스 전체를 기반으로 종합적인 팀 요약 코멘트를 생성합니다."""
    
    system_prompt = """
    당신은 SK조직의 팀 협업 분석 전문가입니다.
    제공된 팀의 협업 분석 데이터를 종합하여, 팀의 전반적인 협업 현황에 대한 심층적인 분석 코멘트를 생성해주세요.
    단순히 수치를 나열하는 것을 넘어, 팀의 협업 패턴, 강점, 약점, 그리고 개선점을 구체적으로 제시해야 합니다.
    긍정적인 부분과 부정적인 부분을 균형있게 다루어 주세요.
    결과는 2-3 문장의 완성된 코멘트 형태로 응답해주세요.
    """

    matrix_summary_list = []
    for member in collaboration_matrix:
        summary_line = (
            f"- {member['name']}: "
            f"역할({member['team_role']}), "
            f"협업률({member['collaboration_rate']}), "
            f"기여도({member['avg_contribution_score']}), "
            f"편중도({member['collaboration_bias']})"
        )
        matrix_summary_list.append(summary_line)
    matrix_summary = "\\n".join(matrix_summary_list)


    human_prompt = f"""
    <팀 협업 분석 데이터>
    - 전체 평균 협업률: {team_summary_stats['avg_collaboration_rate']:.1f}%
    - 전체 평균 기여도: {team_summary_stats['avg_contribution_score']:.1f}점
    - 과의존 위험 멤버: {', '.join(team_summary_stats['high_bias_members']) if team_summary_stats['high_bias_members'] else '없음'}
    - 팀원별 분석 요약:
{matrix_summary}
    </팀 협업 분석 데이터>

    위 데이터를 바탕으로 팀 전체의 협업 상태를 분석하는 종합 코멘트를 작성해주세요.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        summary = _get_llm_content(response).strip()
        return summary if summary else "팀 협업 종합 분석 코멘트 생성에 실패했습니다."
        
    except Exception as e:
        print(f"LLM 팀 요약 생성 오류: {e}")
        return "팀 협업 종합 분석 중 오류가 발생했습니다."

# %%
# 서브모듈 1: 데이터 수집 서브모듈
def collaboration_data_collection_submodule(state: Module3AgentState) -> Module3AgentState:
    """
    협업 분석을 위한 기초 데이터를 수집합니다.
    - 동일 KPI 내 Task 그룹핑
    - Task Summary 데이터 수집
    """
    print("=== 모듈 3: 협업 데이터 수집 시작 ===")
    
    target_team_kpi_ids = state["target_team_kpi_ids"]
    period_id = state["period_id"]
    
    collaboration_relationships = []
    
    # 각 KPI별로 협업 관계 1차 구성
    for team_kpi_id in target_team_kpi_ids:
        kpi_tasks = fetch_collaboration_tasks_by_kpi(team_kpi_id, period_id)
        
        if len(kpi_tasks) > 1:  # 2개 이상 Task가 있어야 협업 가능
            for task in kpi_tasks:
                potential_collaborators = [t["emp_no"] for t in kpi_tasks if t["emp_no"] != task["emp_no"]]
                
                collaboration_relationships.append({
                    "task_id": task["task_id"],
                    "task_summary_id": task.get("task_summary_Id"),
                    "emp_no": task["emp_no"],
                    "emp_name": task["emp_name"],
                    "task_name": task["task_name"],
                    "task_summary": task.get("task_summary", ""),
                    "ai_contribution_score": task.get("ai_contribution_score", 0),
                    "team_kpi_id": team_kpi_id,
                    "potential_collaborators": potential_collaborators,
                    "start_date": task.get("start_date"),
                    "end_date": task.get("end_date"),
                    "collaboration_confirmed": False
                })
    
    print(f"총 {len(collaboration_relationships)}개 잠재적 협업 관계 수집 완료")
    
    new_state = state.copy()
    new_state["messages"] = [HumanMessage(content="모듈 3: 협업 데이터 수집 완료")]
    new_state["collaboration_relationships"] = collaboration_relationships
    return new_state

# %%
# 서브모듈 2: 개인 협업 분석 서브모듈
def individual_collaboration_analysis_submodule(state: Module3AgentState) -> Module3AgentState:
    """
    Task Summary를 LLM으로 분석하여 실제 협업 관계를 확인하고,
    개인별 협업 패턴을 분석합니다.
    """
    print("=== 모듈 3: 개인 협업 분석 시작 ===")
    
    collaboration_relationships = state["collaboration_relationships"]
    team_id = state["team_id"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    
    # 1. LLM으로 실제 협업 관계 확인
    confirmed_collaborations = []
    
    for relation in collaboration_relationships:
        task_summary = relation["task_summary"]
        potential_collaborators = relation["potential_collaborators"]
        
        if task_summary and potential_collaborators:
            llm_collaboration_result = call_llm_for_collaboration_detection(
                task_summary=task_summary,
                task_name=relation["task_name"],
                potential_collaborators=potential_collaborators,
                emp_name=relation["emp_name"]
            )
            
            if llm_collaboration_result.get("is_collaboration", False):
                confirmed_collaborations.append({
                    **relation,
                    "confirmed_collaborators": llm_collaboration_result.get("collaborators", []),
                    "collaboration_description": llm_collaboration_result.get("description", ""),
                    "collaboration_confirmed": True
                })
    
    # 2. 개인별 협업 패턴 분석
    team_members = fetch_team_members_with_tasks(team_id, period_id)
    individual_analysis = {}
    
    for member in team_members:
        emp_no = member["emp_no"]
        emp_name = member["emp_name"]
        role = member["role"]
        total_tasks = member["total_task_count"]
        avg_contribution_score = member.get("avg_contribution_score", 0)
        
        # 팀장(MANAGER) 제외
        if role == "MANAGER":
            print(f"팀장 {emp_name}({emp_no}) 분석에서 제외")
            continue
        
        # 해당 개인의 협업 Task들 필터링
        member_collaborations = [c for c in confirmed_collaborations if c["emp_no"] == emp_no]
        collaboration_task_count = len(member_collaborations)
        
        # 협업자별 카운트
        collaborator_counts = {}
        all_collaborators = []
        for collab in member_collaborations:
            for collaborator in collab["confirmed_collaborators"]:
                collaborator_counts[collaborator] = collaborator_counts.get(collaborator, 0) + 1
                all_collaborators.extend(collab["confirmed_collaborators"])
        
        # 핵심 협업자 (상위 2-3명)
        sorted_collaborators = sorted(collaborator_counts.items(), key=lambda x: x[1], reverse=True)
        key_collaborators = [collab[0] for collab in sorted_collaborators[:3]]
        
        # 협업 편중도 계산을 위한 의존도 지표
        dependency_metrics = {
            "collaboration_concentration": (max(collaborator_counts.values()) / sum(collaborator_counts.values()) * 100) if collaborator_counts else 0,
            "unique_collaborators": len(collaborator_counts),
            "total_collaborations": sum(collaborator_counts.values())
        }
        
        # Task Summary들 수집 (역할 분석용)
        member_task_summaries = [c["task_summary"] for c in member_collaborations if c["task_summary"]]
        
        individual_analysis[emp_no] = {
            "emp_name": emp_name,
            "total_tasks": total_tasks,
            "collaboration_tasks": collaboration_task_count,
            "collaboration_rate": (collaboration_task_count / total_tasks * 100) if total_tasks > 0 else 0,
            "avg_contribution_score": avg_contribution_score or 0,
            "collaborator_counts": collaborator_counts,
            "key_collaborators": key_collaborators,
            "dependency_metrics": dependency_metrics,
            "task_summaries": member_task_summaries
        }
    
    print(f"개인별 협업 분석 완료: {len(individual_analysis)}명")
    
    new_state = state.copy()
    new_state["messages"] = [HumanMessage(content="모듈 3: 개인 협업 분석 완료")]
    new_state["individual_collaboration_analysis"] = individual_analysis
    return new_state

# %%
# 서브모듈 3: 팀 협업 네트워크 서브모듈
def team_collaboration_network_submodule(state: Module3AgentState) -> Module3AgentState:
    """
    팀 전체 협업 네트워크 매트릭스를 생성합니다.
    """
    print("=== 모듈 3: 팀 협업 네트워크 분석 시작 ===")
    
    individual_analysis = state["individual_collaboration_analysis"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    
    collaboration_matrix = []
    
    for emp_no, analysis in individual_analysis.items():
        emp_name = analysis["emp_name"]
        
        print(f"처리 중: {emp_name}({emp_no})")
        
        # 1. 팀 내 역할 분석
        team_role_result = call_llm_for_team_role_analysis(
            task_summaries=analysis["task_summaries"],
            emp_name=emp_name,
            emp_no=emp_no
        )
        
        # 2. 핵심 협업자 이름 매핑
        key_collaborators_with_names = []
        for collaborator_emp_no in analysis["key_collaborators"]:
            if collaborator_emp_no in individual_analysis:
                collaborator_name = individual_analysis[collaborator_emp_no]["emp_name"]
                key_collaborators_with_names.append(f"{collaborator_name}({collaborator_emp_no})")
            else:
                key_collaborators_with_names.append(f"({collaborator_emp_no})")
        
        # 3. Peer Talk 요약
        peer_talk_content = fetch_peer_talk_summary(emp_no, period_id, report_type)
        peer_talk_summary = "동료평가 없음"
        if peer_talk_content:
            peer_talk_summary = call_llm_for_peer_talk_summary(peer_talk_content, emp_name)
        
        # 4. 협업 편중도 분석 (신뢰성 있는 방법)
        collaboration_data = {
            "total_tasks": analysis["total_tasks"],
            "collaboration_tasks": analysis["collaboration_tasks"],
            "collaborator_counts": analysis["collaborator_counts"],
            "dependency_metrics": analysis["dependency_metrics"]
        }
        
        bias_result = call_llm_for_collaboration_bias_analysis(collaboration_data, emp_name, emp_no)
        collaboration_bias = f"{bias_result['bias_level']}"
        if bias_result['bias_level'] == "높음":
            collaboration_bias += "(과의존 위험)"
        elif bias_result['bias_level'] == "낮음":
            collaboration_bias += "(협업 부족)"
        else:
            collaboration_bias += "(적절)"
        
        # 5. 종합 평가
        collaboration_analysis_summary = {
            "team_role": team_role_result["team_role"],
            "collaboration_rate": analysis["collaboration_rate"],
            "key_collaborators": key_collaborators_with_names,
            "collaboration_bias": collaboration_bias
        }
        
        overall_evaluation = call_llm_for_overall_evaluation(
            collaboration_analysis_summary, emp_name, emp_no
        )
        
        # 매트릭스 항목 구성
        matrix_item = {
            "emp_no": emp_no,
            "name": f"{emp_name}({emp_no})",
            "total_tasks": analysis["total_tasks"],
            "collaboration_rate": round(analysis["collaboration_rate"], 1),
            "avg_contribution_score": round(analysis["avg_contribution_score"], 1),
            "team_role": team_role_result["team_role"],
            "key_collaborators": key_collaborators_with_names,
            "peer_talk_summary": peer_talk_summary,
            "collaboration_bias": collaboration_bias,
            "overall_evaluation": overall_evaluation
        }
        
        collaboration_matrix.append(matrix_item)

    # 팀 전체 요약
    total_members = len(collaboration_matrix)
    avg_collaboration_rate = sum([item["collaboration_rate"] for item in collaboration_matrix]) / total_members if total_members > 0 else 0
    avg_contribution_score = sum([item["avg_contribution_score"] for item in collaboration_matrix]) / total_members if total_members > 0 else 0
    high_bias_members = [item["name"] for item in collaboration_matrix if "과의존 위험" in item["collaboration_bias"]]
    
    team_summary_stats = {
        "avg_collaboration_rate": avg_collaboration_rate,
        "avg_contribution_score": avg_contribution_score,
        "high_bias_members": high_bias_members,
        "total_members": total_members
    }
    team_summary = call_llm_for_team_summary(collaboration_matrix, team_summary_stats)
    
    # 최종 결과 구성
    team_collaboration_matrix = {
        "collaboration_matrix": collaboration_matrix,
        "team_summary": team_summary,
        "analysis_period": period_id,
        "analysis_date": "2024-12-19",  # 실제로는 현재 날짜
        "total_members": total_members,
        "avg_collaboration_rate": round(avg_collaboration_rate, 1),
        "avg_contribution_score": round(avg_contribution_score, 1)
    }
    
    print(f"팀 협업 네트워크 매트릭스 생성 완료: {total_members}명")
    
    new_state = state.copy()
    new_state["messages"] = [HumanMessage(content="모듈 3: 팀 협업 네트워크 분석 완료")]
    new_state["team_collaboration_matrix"] = team_collaboration_matrix
    return new_state

# %%
# 서브모듈 4: 협업 기여도 종합 분석 서브모듈 (최종 전 중간평가 - 제외)
def collaboration_comprehensive_analysis_submodule(state: Module3AgentState) -> Module3AgentState:
    """
    협업 기여도 종합 분석 (현재는 팀장용만 처리하므로 패스)
    """
    print("=== 모듈 3: 협업 기여도 종합 분석 (스킵) ===")
    
    new_state = state.copy()
    new_state["messages"] = [HumanMessage(content="모듈 3: 협업 기여도 종합 분석 스킵")]
    return new_state

# %%
# 서브모듈 5: 포맷터 서브모듈
def collaboration_formatter_submodule(state: Module3AgentState) -> Module3AgentState:
    """
    협업 매트릭스를 DB에 저장합니다.
    """
    print("=== 모듈 3: 협업 매트릭스 DB 저장 시작 ===")
    
    team_collaboration_matrix = state["team_collaboration_matrix"]
    team_evaluation_id = state["team_evaluation_id"]
    
    # DB에 저장
    success = save_collaboration_matrix_to_db(team_evaluation_id, team_collaboration_matrix)
    
    if success:
        print(f"협업 매트릭스 저장 성공: team_evaluation_id={team_evaluation_id}")
        status_message = "모듈 3: 협업 매트릭스 DB 저장 완료"
    else:
        print(f"협업 매트릭스 저장 실패: team_evaluation_id={team_evaluation_id}")
        status_message = "모듈 3: 협업 매트릭스 DB 저장 실패"
    
    new_state = state.copy()
    new_state["messages"] = [HumanMessage(content=status_message)]
    return new_state

# %%
# 워크플로우 생성
def create_module3_graph():
    """모듈 3 그래프 생성 및 반환"""
    module3_workflow = StateGraph(Module3AgentState)
    
    # 노드 추가 (State 키와 겹치지 않도록 이름 수정)
    module3_workflow.add_node("data_collection", collaboration_data_collection_submodule)
    module3_workflow.add_node("individual_analysis", individual_collaboration_analysis_submodule)
    module3_workflow.add_node("team_network", team_collaboration_network_submodule)
    module3_workflow.add_node("comprehensive_analysis", collaboration_comprehensive_analysis_submodule)
    module3_workflow.add_node("formatter", collaboration_formatter_submodule)
    
    # 엣지 정의
    module3_workflow.add_edge(START, "data_collection")
    module3_workflow.add_edge("data_collection", "individual_analysis")
    module3_workflow.add_edge("individual_analysis", "team_network")
    module3_workflow.add_edge("team_network", "comprehensive_analysis")
    module3_workflow.add_edge("comprehensive_analysis", "formatter")
    module3_workflow.add_edge("formatter", END)
    
    return module3_workflow.compile()



# # %%
# # 실행 함수
# def run_module3_quarterly():
#     """모듈 3 분기별 실행"""
    
#     # State 정의 (딕셔너리로 생성)
#     state = {
#         "messages": [HumanMessage(content="모듈 3 협업 분석 시작")],
#         "report_type": "quarterly",
#         "team_id": 1,
#         "period_id": 2,
#         "target_task_summary_ids": [1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30],
#         "target_team_kpi_ids": [1, 2, 3],
#         "team_evaluation_id": 101,
#         "collaboration_relationships": [],
#         "individual_collaboration_analysis": {},
#         "team_collaboration_matrix": {}
#     }

#     # 그래프 생성 및 실행
#     print("모듈 3 실행 시작...")
#     module3_graph = create_module3_graph()
#     result = module3_graph.invoke(state)
#     print("모듈 3 실행 완료!")
    
#     # 결과 출력
#     for message in result['messages']:
#         print(f"- {message.content}")
    
#     # 최종 협업 매트릭스 확인
#     if result.get('team_collaboration_matrix'):
#         print("\n=== 생성된 협업 매트릭스 ===")
#         matrix = result['team_collaboration_matrix']
#         print(f"팀 요약: {matrix['team_summary']}")
#         print(f"분석 인원: {matrix['total_members']}명")
#         print(f"평균 협업률: {matrix['avg_collaboration_rate']}%")
#         print(f"평균 기여도: {matrix['avg_contribution_score']}점")
        
#         print("\n팀원별 협업 분석:")
#         for member in matrix['collaboration_matrix']:
#             print(f"- {member['name']}: {member['team_role']}, 협업률 {member['collaboration_rate']}%, 기여도 {member['avg_contribution_score']}점")
    
#     return result

# # %%
# # 테스트 실행
# if __name__ == "__main__":
#     result = run_module3_quarterly()


# 연말 실행 함수
def run_module3_annual():
    """모듈 3 연말 실행"""
    
    state = {
        "messages": [HumanMessage(content="모듈 3 협업 분석 시작")],
        "report_type": "annual",
        "team_id": 1,
        "period_id": 4,
        "target_task_summary_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        "target_team_kpi_ids": [1, 2, 3, 4],
        "team_evaluation_id": 104,
        "collaboration_relationships": [],
        "individual_collaboration_analysis": {},
        "team_collaboration_matrix": {}
    }

    module3_graph = create_module3_graph()
    result = module3_graph.invoke(state)
    return result

# 실행
if __name__ == "__main__":
    result = run_module3_annual()


def run_module3_quarterly_q2():
    """모듈 3 2분기(2024년 2분기) 실행"""
    
    # State 정의 (딕셔너리로 생성)
    state = {
        "messages": [HumanMessage(content="모듈 3 협업 분석 시작 (2분기)")],
        "report_type": "quarterly",
        "team_id": 1,
        "period_id": 2,  # 2분기
        "target_task_summary_ids": [2, 6, 10, 14, 18, 22, 26, 30],  # 2분기 task_summary_id만 추출
        "target_team_kpi_ids": [1, 2, 3],  # 필요시 조정
        "team_evaluation_id": 102,  # 2분기 team_evaluation_id
        "collaboration_relationships": [],
        "individual_collaboration_analysis": {},
        "team_collaboration_matrix": {}
    }

    # 그래프 생성 및 실행
    print("모듈 3 2분기 실행 시작...")
    module3_graph = create_module3_graph()
    result = module3_graph.invoke(state)
    print("모듈 3 2분기 실행 완료!")
    
    # 결과 출력
    for message in result['messages']:
        print(f"- {message.content}")
    
    # 최종 협업 매트릭스 확인
    if result.get('team_collaboration_matrix'):
        print("\n=== 생성된 협업 매트릭스 ===")
        matrix = result['team_collaboration_matrix']
        print(f"팀 요약: {matrix['team_summary']}")
        print(f"분석 인원: {matrix['total_members']}명")
        print(f"평균 협업률: {matrix['avg_collaboration_rate']}%")
        print(f"평균 기여도: {matrix['avg_contribution_score']}점")

run_module3_quarterly_q2()