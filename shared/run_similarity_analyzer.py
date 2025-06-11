# run_similarity_analysis.py
# SimilarityAnalyzer 실행 스크립트

from similarity_analyzer import SimilarityAnalyzer


def run_similarity_analysis(period_id=2):
    """팀 + 개인 유사도 분석 통합 실행"""
    print("=== 유사도 분석 시작 ===\n")
    
    analyzer = SimilarityAnalyzer()
    
    # 1. 팀 유사도 분석
    print("1. 팀 유사도 분석")
    team_success = analyzer.analyze_teams()
    
    # 2. 개인 유사도 분석  
    print("\n2. 개인 유사도 분석")
    individual_success = analyzer.analyze_individuals(period_id)
    
    print("\n=== 유사도 분석 완료 ===")
    
    if team_success and individual_success:
        return analyzer
    else:
        return None


def test_similarity_results(analyzer):
    """유사도 분석 결과 테스트"""
    if not analyzer:
        print("분석기가 없습니다.")
        return
    
    print("\n=== 결과 테스트 ===")
    
    # 팀 유사도 테스트
    print("\n--- 팀 유사도 테스트: 팀 ID 1 ---")
    similar_teams = analyzer.get_similar_teams(1, include_scores=True)
    if similar_teams:
        for team in similar_teams:
            print(f"팀 {team['team_id']}: {team['team_name']} (유사도: {team['similarity_score']})")
    else:
        print("유사한 팀을 찾을 수 없습니다.")
    
    # 개인 유사도 테스트
    print("\n--- 개인 유사도 테스트: E002 ---")
    similar_individuals = analyzer.get_similar_individuals('E002', include_scores=True)
    if similar_individuals:
        for individual in similar_individuals:
            print(f"{individual['emp_no']} ({individual['emp_name']}) - {individual['position']} - {individual['team_name']} (유사도: {individual['similarity_score']})")
    else:
        print("유사한 개인을 찾을 수 없습니다.")


if __name__ == "__main__":
    # 분석 실행
    analyzer = run_similarity_analysis(period_id=2)
    
    # 결과 테스트
    test_similarity_results(analyzer)
    
    # 추가 테스트 (필요시)
    if analyzer:
        print("\n=== 추가 테스트 가능 ===")
        print("analyzer.get_similar_teams(team_id, include_scores=True)")
        print("analyzer.get_similar_individuals(emp_no, include_scores=True)")