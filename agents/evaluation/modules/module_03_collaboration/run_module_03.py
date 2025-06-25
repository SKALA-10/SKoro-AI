# ================================================================
# run_module3.py - 모듈 3 실행 파일
# ================================================================

from langchain_core.messages import HumanMessage
from agent import Module3AgentState, create_module3_graph

# ================================================================
# 실행 함수들
# ================================================================

def run_module3_quarterly():
    """모듈 3 분기별 실행"""
    
    # State 정의 (딕셔너리로 생성)
    state = {
        "messages": [HumanMessage(content="모듈 3 협업 분석 시작")],
        "report_type": "quarterly",
        "team_id": 1,
        "period_id": 2,
        "target_task_summary_ids": [1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30],
        "target_team_kpi_ids": [1, 2, 3],
        "team_evaluation_id": 101,
        "collaboration_relationships": [],
        "individual_collaboration_analysis": {},
        "team_collaboration_matrix": {}
    }

    # 그래프 생성 및 실행
    print("모듈 3 실행 시작...")
    module3_graph = create_module3_graph()
    result = module3_graph.invoke(state)
    print("모듈 3 실행 완료!")
    
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
        
        print("\n팀원별 협업 분석:")
        for member in matrix['collaboration_matrix']:
            print(f"- {member['name']}: {member['team_role']}, 협업률 {member['collaboration_rate']}%, 기여도 {member['avg_contribution_score']}점")
    
    return result

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

    print("모듈 3 연말 실행 시작...")
    module3_graph = create_module3_graph()
    result = module3_graph.invoke(state)
    print("모듈 3 연말 실행 완료!")
    
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
        
        print("\n팀원별 협업 분석:")
        for member in matrix['collaboration_matrix']:
            print(f"- {member['name']}: {member['team_role']}, 협업률 {member['collaboration_rate']}%, 기여도 {member['avg_contribution_score']}점")
    
    return result

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

        print("\n팀원별 협업 분석:")
        for member in matrix['collaboration_matrix']:
            print(f"- {member['name']}: {member['team_role']}, 협업률 {member['collaboration_rate']}%, 기여도 {member['avg_contribution_score']}점")
    
    return result

# ================================================================
# 메인 실행 부분
# ================================================================

if __name__ == "__main__":
    # 테스트 실행 - 2분기
    print("🧪 모듈 3 테스트 실행 - 2분기")
    result_q2 = run_module3_quarterly_q2()
    
    print("\n" + "="*60)
    
    # 테스트 실행 - 연말
    print("🧪 모듈 3 테스트 실행 - 연말")
    result_annual = run_module3_annual()