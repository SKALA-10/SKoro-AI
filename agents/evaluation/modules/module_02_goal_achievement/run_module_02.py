# 실행 스크립트 (간단함)
from agent import create_module2_graph, Module2AgentState
from langchain_core.messages import HumanMessage

def run_module2_quarterly():
    """모듈 2 분기별 실행"""
    # State 정의
    state = Module2AgentState(
        messages=[HumanMessage(content="모듈 2 분기별 평가 시작")],
        report_type="quarterly",
        team_id=1,
        period_id=2,
        target_task_summary_ids=[1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30],
        target_team_kpi_ids=[1, 2, 3],
        team_evaluation_id=101,
        updated_task_ids=None,
        updated_team_kpi_ids=None,
        kpi_individual_relative_contributions=None
    )

    # 그래프 생성 및 실행
    print("모듈 2 실행 시작...")
    module2_graph = create_module2_graph()
    result = module2_graph.invoke(state)
    print("모듈 2 실행 완료!")
    print(f"최종 메시지: {result['messages'][-1].content}")
    return result

if __name__ == "__main__":
    run_module2_quarterly()