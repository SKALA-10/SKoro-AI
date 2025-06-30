from agents.evaluation.modules.module_06_4p_evaluation.agent import *
import time

# 
def run_single_evaluation(state: dict):
    """지정된 state에 대해 단일 평가를 실행하고 결과를 출력합니다."""
    print(f"--- {state['emp_no']}에 대한 {state['report_type']} 평가 시작 ---")
    
    # 그래프 생성
    graph = create_module6_graph_efficient()

    # 그래프 실행
    try:
        result = graph.invoke(state)

        # 결과 출력
        print("\n=== 최종 결과 ===")
        for msg in result.get("messages", []):
            print(msg)
        print("\n통합 4P 평가 결과:")
        print(result.get("integrated_data", {}))
        print(f"--- {state['emp_no']} 평가 완료 ---\n")

    except Exception as e:
        print(f"!!! {state['emp_no']} 평가 중 오류 발생: {e} !!!\n")


def run_evaluations_for_period(period_name: str, report_type: str, period_id: int, team_id: int, employee_reports: list):
    """특정 기간에 대해 모든 직원의 평가를 실행합니다."""
    print(f"=================================================")
    print(f"  {period_name} 4P 평가 전체 실행 시작")
    print(f"=================================================\n")

    for report_info in employee_reports:
        state = {
            "report_type": report_type,
            "team_id": team_id,
            "period_id": period_id,
            "emp_no": report_info["emp_no"],
            "feedback_report_id": report_info.get("feedback_report_id"),
            "final_evaluation_report_id": report_info.get("final_evaluation_report_id"),
            "messages": [],
            "evaluation_criteria": {},
            "evaluation_results": {},
            "integrated_data": {},
            "raw_evaluation_criteria": "",
        }
        run_single_evaluation(state)
        time.sleep(1)  # API 호출 간에 약간의 지연 시간 추가

    print(f"=================================================")
    print(f"  {period_name} 4P 평가 전체 실행 완료")
    print(f"=================================================\n")


if __name__ == "__main__":
    # --- 2분기 평가 실행 ---
    # `feedback_reports` 테이블 참고 (team_evaluation_id=102, period_id=2)
    q2_employee_reports = [
        {"emp_no": "SK0002", "feedback_report_id": 2001},
        {"emp_no": "SK0003", "feedback_report_id": 2002},
        {"emp_no": "SK0004", "feedback_report_id": 2003},
    ]
    run_evaluations_for_period(
        period_name="2024년 2분기",
        report_type="quarterly",  # 분기 평가
        period_id=2,
        team_id=1,
        employee_reports=q2_employee_reports
    )

    # --- 4분기(연말) 평가 실행 ---
    # `final_evaluation_reports` 테이블 참고 (team_evaluation_id=104, period_id=4)
    annual_employee_reports = [
        {"emp_no": "SK0002", "final_evaluation_report_id": 4001},
        {"emp_no": "SK0003", "final_evaluation_report_id": 4002},
        {"emp_no": "SK0004", "final_evaluation_report_id": 4003},
    ]
    run_evaluations_for_period(
        period_name="2024년 4분기(연말)",
        report_type="annual",  # 연말 평가
        period_id=4,
        team_id=1,
        employee_reports=annual_employee_reports
    )

