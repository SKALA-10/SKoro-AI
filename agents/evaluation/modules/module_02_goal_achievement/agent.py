from db_utils import *
from llm_utils import *

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Literal, TypedDict, Dict
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END


# --- Module2AgentState 정의 ---
class Module2AgentState(TypedDict):
    """
    모듈 2 (목표달성도 분석 모듈)의 내부 상태를 정의합니다.
    이 상태는 모듈 2 내의 모든 서브모듈이 공유하고 업데이트합니다.
    """
    messages: Annotated[List[HumanMessage], operator.add] 

    report_type: Literal["quarterly", "annual"] 
    team_id: int 
    period_id: int 
    
    target_task_summary_ids: List[int] 
    target_team_kpi_ids: List[int] 

    updated_task_ids: List[int]
    updated_team_kpi_ids: List[int]
    
    kpi_individual_relative_contributions: List[Dict] = [] 

    feedback_report_id: int = None 
    team_evaluation_id: int = None 
    final_evaluation_report_id: int = None 
    updated_temp_evaluation_ids_list: List[int] = [] 

# --- 서브모듈 함수 정의 ---

# 1. 데이터 수집 서브모듈
def data_collection_submodule(state: Module2AgentState) -> Module2AgentState:
    messages = state.get("messages", []) + [HumanMessage(content="모듈 2: 데이터 수집 ID 초기화 완료")] 
    return {"messages": messages}


# 2. 개인 기여도 계산 서브모듈
def calculate_individual_contribution_submodule(state: Module2AgentState) -> Module2AgentState:
    report_type = state["report_type"] 
    target_task_summary_ids = state["target_task_summary_ids"] 
    
    updated_task_ids_list = [] 

    for task_summary_id in target_task_summary_ids: 
        task_data = fetch_task_summary_by_id(task_summary_id) 
        if not task_data: 
            print(f"Warning: Task data not found for task_summary_id {task_summary_id}.") 
            continue 
        
        task_id = task_data["task_id"] 
        
        llm_results = {} 
        update_data = {} 

        # --- 분기별 로직: 기여도만 계산 ---
        if report_type == "quarterly": 
            task_summary_text = task_data.get("task_summary", "") 
            if task_summary_text: 
                llm_results = call_llm_for_task_contribution(task_summary_text) 
                update_data = {
                    "ai_contribution_score": llm_results.get("score"), 
                    "ai_analysis_comment_task": llm_results.get("comment") 
                }
            else: 
                print(f"Warning: No task_summary found for task_id {task_id} in {report_type} report.") 
                continue 
        # --- 연말 로직: 달성률/등급 및 기여도 계산 ---
        elif report_type == "annual": 
            target_level = task_data.get("target_level", "") 
            task_performance = task_data.get("task_performance", "") 
            task_summary_text_q4 = task_data.get("task_summary", "") 

            grade_definitions = fetch_grade_definitions_from_db() 
            
            if target_level and task_performance: 
                llm_achievement_results = call_llm_for_task_achievement(target_level, task_performance, grade_definitions) 
                
                llm_contribution_results = call_llm_for_task_contribution(task_summary_text_q4) 
                
                update_data = {
                    "ai_contribution_score": llm_contribution_results.get("score"), 
                    "ai_achievement_rate": llm_achievement_results.get("rate"), 
                    "ai_assessed_grade": llm_achievement_results.get("grade"), 
                    "ai_analysis_comment_task": llm_achievement_results.get("analysis") 
                }
            else: 
                print(f"Warning: target_level or task_performance missing for task_id {task_id} in {report_type} report.") 
                continue 

        if update_task_ai_results_in_db(task_id, update_data): 
            updated_task_ids_list.append(task_id) 
        else: 
            print(f"Failed to update AI results for task_id: {task_id}") 

    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 개인 Task 기여도/달성률 계산 및 DB 업데이트 완료 ({len(updated_task_ids_list)}건)")] 
    return {"messages": messages, "updated_task_ids": updated_task_ids_list}



# 3. 팀 목표 분석 서브모듈 (수정: KPI 내 개인 상대 기여도 계산 및 LLM 요청 후 tasks 업데이트)
def analyze_team_goals_submodule(state: Module2AgentState) -> Module2AgentState:
    report_type = state["report_type"] 
    target_team_kpi_ids = state["target_team_kpi_ids"] 
    period_id = state["period_id"] 

    updated_team_kpi_ids_list = [] 
    kpi_individual_relative_contributions_for_state = [] 

    for team_kpi_id in target_team_kpi_ids: 
        kpi_data = fetch_kpi_data_by_id(team_kpi_id) 
        if not kpi_data: 
            print(f"Warning: Team KPI data not found for team_kpi_id {team_kpi_id}.") 
            continue 

        tasks_in_this_kpi = fetch_tasks_for_kpi(team_kpi_id, period_id) 
        
        llm_input_for_kpi_analysis = {
            "kpi_goal": kpi_data.get("kpi_name"), 
            "kpi_description": kpi_data.get("kpi_description"), 
            "team_members_tasks": [
                {
                    "emp_no": task.get("emp_no"), 
                    "task_id": task.get("task_id"), 
                    "task_name": task.get("task_name"), 
                    "task_summary": task.get("task_summary"), 
                    "ai_contribution_score_from_individual_analysis": task.get("ai_contribution_score") 
                } for task in tasks_in_this_kpi
            ]
        }

        llm_kpi_analysis_results = call_llm_for_kpi_relative_contribution(llm_input_for_kpi_analysis) 

        update_data_kpi = { 
            "ai_kpi_overall_progress_rate": llm_kpi_analysis_results.get("kpi_overall_rate"), 
            "ai_kpi_analysis_comment": llm_kpi_analysis_results.get("kpi_analysis_comment") 
        }

        if update_team_kpi_ai_results_in_db(team_kpi_id, update_data_kpi): 
            updated_team_kpi_ids_list.append(team_kpi_id) 
            
            if "individual_relative_contributions_in_kpi" in llm_kpi_analysis_results: 
                relative_contributions_by_emp = llm_kpi_analysis_results["individual_relative_contributions_in_kpi"]
                kpi_individual_relative_contributions_for_state.append({ 
                    "team_kpi_id": team_kpi_id,
                    "relative_contributions": relative_contributions_by_emp
                }) 

                # --- 수정된 부분: tasks 테이블 ai_contribution_score 및 ai_analysis_comment_task 업데이트 ---
                for task in tasks_in_this_kpi: # 현재 KPI에 속한 Task들을 다시 순회
                    emp_no_task = task.get("emp_no")
                    task_id_current = task.get("task_id")
                    
                    if emp_no_task in relative_contributions_by_emp:
                        new_contribution_score = relative_contributions_by_emp[emp_no_task]
                        
                        # LLM 호출을 위한 Task 상세 정보 준비
                        # task_data에는 emp_name도 포함될 수 있도록 fetch_task_summary_by_id 쿼리 확인
                        # (py의 fetch_employees_by_team_id도 emp_name을 가져옴)
                        task_data_for_comment = fetch_task_summary_by_id(task.get("task_summary_Id")) # task_summary_Id 필요
                        if not task_data_for_comment:
                            print(f"Warning: Task data for comment generation not found for task_summary_Id {task.get('task_summary_Id')}.")
                            continue

                        # 새로운 LLM 호출: Task별 상세 기여도 근거 코멘트 생성
                        reason_llm_results = call_llm_for_individual_contribution_reason_comment(
                            task_data_for_comment, # Task 상세 정보
                            float(new_contribution_score), # 조정된 점수 (LLM에 전달 시 float으로 변환)
                            kpi_data.get('kpi_name', ''), # KPI 목표
                            llm_kpi_analysis_results.get('kpi_analysis_comment', '') # KPI 전체 분석 코멘트
                        )
                        adjusted_comment = reason_llm_results.get("comment", f"AI 근거 생성 실패: {emp_no_task}의 Task {task_id_current}에 대한 근거를 생성할 수 없습니다.")
                        
                        update_data_task = {
                            "ai_contribution_score": new_contribution_score,
                            "ai_analysis_comment_task": adjusted_comment
                        }
                        
                        if not update_task_ai_results_in_db(task_id_current, update_data_task):
                            print(f"Warning: Failed to update ai_contribution_score for task_id {task_id_current} (emp_no: {emp_no_task}) with new relative contribution.")
                    else:
                        print(f"Warning: Emp_no {emp_no_task} from task_id {task_id_current} not found in LLM's relative contributions for KPI {team_kpi_id}. ai_contribution_score not updated for this task.")

        else: 
            print(f"Failed to update AI results for team_kpi_id: {team_kpi_id}") 

    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 팀 목표 분석 및 DB 업데이트 완료 ({len(updated_team_kpi_ids_list)}건)")] 
    
    return {"messages": messages, "updated_team_kpi_ids": updated_team_kpi_ids_list,
            "kpi_individual_relative_contributions": kpi_individual_relative_contributions_for_state}



# 4. 모듈 2 관련 레포트 테이블 데이터 생성/업데이트 서브모듈
def generate_module2_report_data_submodule(state: Module2AgentState) -> Module2AgentState:
    report_type = state["report_type"] 
    team_id = state["team_id"] 
    period_id = state["period_id"] 
    
    kpi_individual_relative_contributions = state.get("kpi_individual_relative_contributions", []) 
    
    updated_ids_for_state = {} 

    # 개인의 팀 전체 기여도 계산 (KPI별 상대 기여도 기반)
    emp_overall_relative_contributions = {} 
    
    for kpi_result in kpi_individual_relative_contributions: 
        for emp_no, relative_score in kpi_result["relative_contributions"].items(): 
            if emp_no not in emp_overall_relative_contributions: 
                emp_overall_relative_contributions[emp_no] = 0 
            emp_overall_relative_contributions[emp_no] += relative_score 

    # --- 팀 전체 기여도 합계 100%로 정규화 ---
    total_sum_of_relative_contributions = sum(emp_overall_relative_contributions.values())
    if total_sum_of_relative_contributions > 0:
        adjustment_factor = 100.0 / total_sum_of_relative_contributions
        for emp_no, score in emp_overall_relative_contributions.items():
            emp_overall_relative_contributions[emp_no] = round(score * adjustment_factor, 2)
    # ----------------------------------------

    # 모든 개인 Task 결과는 여전히 필요 
    all_individual_task_results_raw = [] 
    for task_summary_id in state["target_task_summary_ids"]: 
        task_data = fetch_task_summary_by_id(task_summary_id) 
        if task_data: 
            all_individual_task_results_raw.append(task_data) 


    # 개인용 분기별 피드백 레포트 (feedback_reports)
    if report_type == "quarterly": 
        # 1. 해당 팀의 모든 emp_no 조회 (피드백 레포트는 팀원용)
        all_team_members_in_db = fetch_employees_by_team_id(team_id)

        for member_info in all_team_members_in_db:
            emp_no_current_member = member_info["emp_no"]
            emp_name_current_member = member_info["emp_name"] # 직원 이름 추가

            # 팀장(MANAGER) 역할은 피드백 레포트를 직접 생성하지 않으므로 건너뜁니다.
            if member_info.get("role") == "MANAGER": 
                print(f"Info: Skipping feedback_reports for manager {emp_no_current_member}.")
                continue

            # 해당 팀원에게 해당하는 Task Summaries 필터링
            individual_tasks_for_report = [
                task for task in all_individual_task_results_raw 
                if task.get("emp_no") == emp_no_current_member and task.get("period_id") <= period_id 
            ]

            if not individual_tasks_for_report: 
                print(f"Warning: No individual tasks found for emp_no {emp_no_current_member} in period {period_id}. Skipping feedback_reports save for this member.") 
                continue 

            # LLM 호출 시 emp_name, emp_no 전달
            individual_overall_results = call_llm_for_overall_contribution_summary(
                individual_tasks_for_report, emp_name_current_member, emp_no_current_member
            ) 
            calculated_individual_quarterly_contribution = emp_overall_relative_contributions.get(emp_no_current_member, 0) 

            team_evaluation_id_for_report = state.get("team_evaluation_id") 
            if team_evaluation_id_for_report is None: 
                print(f"Warning: team_evaluation_id for team_id={team_id}, period_id={period_id} is missing in state. Cannot save feedback_reports for {emp_no_current_member}. (앞단 Agent에서 생성 필요)") 
            else: 
                actual_team_eval_id_in_db = fetch_team_evaluation_id_by_team_and_period(team_id, period_id) 
                if actual_team_eval_id_in_db != team_evaluation_id_for_report: 
                     print(f"Warning: team_evaluation_id {team_evaluation_id_for_report} from state does not match existing ID in DB for team={team_id}, period={period_id}. Skipping feedback_reports save for {emp_no_current_member}.") 
                else: 
                    # --- INSERT 또는 UPDATE 로직 (ON DUPLICATE KEY UPDATE 사용) ---
                    feedback_report_id = save_feedback_report_module2_results_to_db(
                        emp_no_current_member, team_evaluation_id_for_report, 
                        {
                            "ai_individual_total_contribution_quarterly": calculated_individual_quarterly_contribution, 
                            "ai_overall_contribution_summary_comment": individual_overall_results.get("comment") 
                        }
                    )
                    updated_ids_for_state["feedback_report_id"] = feedback_report_id 
                    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 개인 {emp_no_current_member} 분기별 레포트 내용 생성/업데이트 및 feedback_reports 저장 완료 (ID: {feedback_report_id})")] 

    # 팀장용 분기별/연말 팀 전체 평가 레포트 (team_evaluations)
    team_evaluation_id = state.get("team_evaluation_id") 
    if team_evaluation_id is None: 
        print(f"Warning: team_evaluation_id for team_id={team_id}, period_id={period_id} is missing in state. Cannot update team_evaluations. (앞단 Agent에서 생성 필요)") 
    else: 
        actual_team_eval_id_in_db = fetch_team_evaluation_id_by_team_and_period(team_id, period_id) 
        if actual_team_eval_id_in_db != team_evaluation_id: 
             print(f"Warning: team_evaluation_id {team_evaluation_id} from state does not match existing ID in DB for team={team_id}, period={period_id}. Skipping team_evaluations update.") 
        else: 
            all_team_kpis_results = [fetch_kpi_data_by_id(kpi_id) for kpi_id in state["target_team_kpi_ids"] if fetch_kpi_data_by_id(kpi_id)] 
            team_overall_results = call_llm_for_team_overall_analysis(all_team_kpis_results) 
            
            update_data = {
                "ai_team_overall_achievement_rate": team_overall_results.get("overall_rate"), 
                "ai_team_overall_analysis_comment": team_overall_results.get("comment") 
            }
            update_team_evaluations_module2_results_in_db(team_evaluation_id, update_data) 
            updated_ids_for_state["team_evaluation_id"] = team_evaluation_id 
            messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 팀 전체 분석 코멘트 생성 및 team_evaluations 업데이트 완료 (ID: {team_evaluation_id})")] 


    # 개인용 연말 최종 평가 레포트 (final_evaluation_reports)
    if report_type == "annual": 
        all_team_members_in_db = fetch_employees_by_team_id(team_id)

        for member_info in all_team_members_in_db:
            emp_no_current_member = member_info["emp_no"]
            emp_name_current_member = member_info["emp_name"] # 직원 이름 추가

            # 팀장(MANAGER) 역할은 최종 평가 레포트의 직접 대상이 아니므로 건너뜁니다.
            if member_info.get("role") == "MANAGER": 
                print(f"Info: Skipping final_evaluation_reports for manager {emp_no_current_member}.")
                continue

            # 해당 팀원에게 해당하는 Task Summaries 필터링
            individual_tasks_for_annual_report = [
                task for task in all_individual_task_results_raw 
                if task.get("emp_no") == emp_no_current_member and task.get("period_id") <= period_id
            ]
            if not individual_tasks_for_annual_report: 
                print(f"Warning: No individual tasks found for emp_no {emp_no_current_member} in period {period_id}. Skipping final_evaluation_reports save for this member.") 
                continue 

            # LLM 호출 시 emp_name, emp_no 전달
            annual_individual_summary_results = call_llm_for_overall_contribution_summary(
                individual_tasks_for_annual_report, emp_name_current_member, emp_no_current_member
            ) 
            
            calculated_annual_individual_total_contribution = emp_overall_relative_contributions.get(emp_no_current_member, 0) 
            
            final_team_evaluation_id_example = state.get("team_evaluation_id") 
            if final_team_evaluation_id_example is None: 
                print(f"Warning: team_evaluation_id for team_id={team_id}, period_id={period_id} is missing in state. Cannot save final_evaluation_reports for {emp_no_current_member}. (앞단 Agent에서 생성 필요)") 
            else: 
                actual_team_eval_id_in_db = fetch_team_evaluation_id_by_team_and_period(team_id, period_id) 
                if actual_team_eval_id_in_db != final_team_evaluation_id_example: 
                     print(f"Warning: team_evaluation_id {final_team_evaluation_id_example} from state does not match existing ID in DB for team={team_id}, period={period_id}. Skipping final_evaluation_reports save for {emp_no_current_member}.") 
                else: 
                    final_report_id = save_final_evaluation_report_module2_results_to_db(
                        emp_no_current_member, final_team_evaluation_id_example, 
                        {
                            "ai_annual_individual_total_contribution": calculated_annual_individual_total_contribution, 
                            "ai_annual_achievement_rate": annual_individual_summary_results.get("average_rate"), 
                            "ai_annual_performance_summary_comment": annual_individual_summary_results.get("comment") 
                        }
                    )
                    updated_ids_for_state["final_report_id"] = final_report_id 
                    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 개인 {emp_no_current_member} 연말 최종 평가 레포트 내용 생성 및 final_evaluation_reports 저장 완료 (ID: {final_report_id})")] 
    
    # 최종 전 중간 평가 자료 (temp_evaluations)
    if report_type == "annual": 
        all_team_members = fetch_employees_by_team_id(team_id) 

        for member in all_team_members:
            emp_no_current_member = member["emp_no"]
            emp_name_current_member = member["emp_name"] # 직원 이름 추가

            # 팀장(MANAGER) 역할도 temp_evaluations에는 포함될 수 있으므로 (참고 자료)
            # 여기서는 MANAGER 역할도 포함하여 처리합니다.
            
            # 해당 팀원에게 해당하는 Task Summaries 필터링
            individual_tasks_for_temp_eval = [
                task for task in all_individual_task_results_raw
                if task.get("emp_no") == emp_no_current_member and task.get("period_id") <= period_id
            ]
            
            if not individual_tasks_for_temp_eval:
                print(f"Warning: No individual tasks found for emp_no {emp_no_current_member} in period {period_id}. Skipping temp_evaluations update for this member.") 
                continue 

            # LLM 호출 시 emp_name, emp_no 전달
            key_performance_summary_results = call_llm_for_overall_contribution_summary(
                individual_tasks_for_temp_eval, emp_name_current_member, emp_no_current_member
            )
            
            temp_eval_id_for_member = fetch_temp_evaluation_id_by_emp_and_period(emp_no_current_member, period_id) 

            if temp_eval_id_for_member is None: 
                print(f"Warning: temp_evaluation_id for emp_no={emp_no_current_member}, period_id={period_id} is missing in DB. Cannot update temp_evaluations. (앞단 Agent에서 생성 필요)") 
            else: 
                update_temp_evaluations_module2_results_in_db(
                    temp_eval_id_for_member,
                    {
                        "ai_annual_key_performance_contribution_summary": key_performance_summary_results.get("comment")
                    }
                )
                if "updated_temp_evaluation_ids_list" not in updated_ids_for_state: 
                     updated_ids_for_state["updated_temp_evaluation_ids_list"] = [] 
                updated_ids_for_state["updated_temp_evaluation_ids_list"].append(temp_eval_id_for_member) 
                
                messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 팀원 {emp_no_current_member} 연간 핵심 성과 기여도 요약 생성 및 temp_evaluations 업데이트 완료 (ID: {temp_eval_id_for_member})")] 

    return {"messages": messages, **updated_ids_for_state}


# 5. 포맷터 서브모듈
def formatter_submodule(state: Module2AgentState) -> Module2AgentState:
    messages = state.get("messages", []) + [HumanMessage(content="모듈 2: 포맷팅 완료")]
    return {"messages": messages}


# 워크플로우 생성
def create_module2_graph():
    """모듈 2 그래프 생성 및 반환"""
    module2_workflow = StateGraph(Module2AgentState)
    
    # 노드 추가
    module2_workflow.add_node("data_collection", data_collection_submodule)
    module2_workflow.add_node("calculate_individual_contribution", calculate_individual_contribution_submodule)
    module2_workflow.add_node("analyze_team_goals", analyze_team_goals_submodule)
    module2_workflow.add_node("generate_module2_report_data", generate_module2_report_data_submodule)
    module2_workflow.add_node("formatter", formatter_submodule)
    
    # 엣지 정의
    module2_workflow.add_edge(START, "data_collection")
    module2_workflow.add_edge("data_collection", "calculate_individual_contribution")
    module2_workflow.add_edge("calculate_individual_contribution", "analyze_team_goals")
    module2_workflow.add_edge("analyze_team_goals", "generate_module2_report_data")
    module2_workflow.add_edge("generate_module2_report_data", "formatter")
    module2_workflow.add_edge("formatter", END)
    
    return module2_workflow.compile()