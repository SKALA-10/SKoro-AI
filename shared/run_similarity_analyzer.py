# run_similarity_analysis.py
# SimilarityAnalyzer 실행 스크립트

from similarity_analyzer import SimilarityAnalyzer


def run_similarity_analysis(period_id=2, use_cache=True, save_to_cache=True):
    """팀 + 개인 유사도 분석 통합 실행"""
    print("=== 유사도 분석 시작 ===\n")
    
    analyzer = SimilarityAnalyzer()
    
    # 캐시 상태 확인
    cache_status = analyzer.get_cache_status()
    print(f"캐시 상태: {cache_status}")
    
    # 1. 팀 유사도 분석
    print("\n1. 팀 유사도 분석")
    if use_cache and cache_status['teams_cached']:
        print("기존 팀 캐시를 사용합니다.")
        team_success = True
    else:
        print("새로운 팀 분석을 수행합니다.")
        team_success = analyzer.analyze_teams(save_to_cache=save_to_cache)
    
    # 2. 개인 유사도 분석  
    print("\n2. 개인 유사도 분석")
    quarter = f"Q{period_id}"
    if use_cache and quarter in cache_status['individuals_quarters']:
        print(f"기존 개인 캐시를 사용합니다 ({quarter}).")
        individual_success = True
    else:
        print(f"새로운 개인 분석을 수행합니다 ({quarter}).")
        individual_success = analyzer.analyze_individuals(period_id, save_to_cache=save_to_cache)
    
    print("\n=== 유사도 분석 완료 ===")
    
    if team_success and individual_success:
        return analyzer
    else:
        return None


def test_similarity_results(analyzer, period_id=2):
    """유사도 분석 결과 테스트"""
    if not analyzer:
        print("분석기가 없습니다.")
        return
    
    print("\n=== 결과 테스트 ===")
    
    # 팀 유사도 테스트 (캐시 우선)
    print("\n--- 팀 유사도 테스트: 팀 ID 1 (캐시 사용) ---")
    similar_teams = analyzer.get_similar_teams(1, use_cache=True)
    if similar_teams:
        print(f"유사한 팀 ID들: {similar_teams}")
    else:
        print("유사한 팀을 찾을 수 없습니다.")
    
    # 개인 유사도 테스트 (캐시 우선)
    print(f"\n--- 개인 유사도 테스트: E002 (캐시 사용, Q{period_id}) ---")
    similar_individuals = analyzer.get_similar_individuals('E002', period_id=period_id, cl=2, use_cache=True)
    if similar_individuals:
        print(f"유사한 개인 emp_no들: {similar_individuals}")
    else:
        print("유사한 개인을 찾을 수 없습니다.")
    
    # 실시간 분석 결과와 비교 (캐시와 다른지 확인)
    print(f"\n--- 실시간 분석 결과 비교 ---")
    if hasattr(analyzer.team_analyzer, 'team_data') and analyzer.team_analyzer.team_data:
        realtime_teams = analyzer.get_similar_teams(1, use_cache=False, include_scores=True)
        if realtime_teams:
            print("실시간 팀 유사도 (상세):")
            for team in realtime_teams:
                print(f"  팀 {team['team_id']}: {team['team_name']} (유사도: {team['similarity_score']})")
    
    if hasattr(analyzer.individual_analyzer, 'cluster_results') and analyzer.individual_analyzer.cluster_results:
        realtime_individuals = analyzer.get_similar_individuals('E002', period_id=period_id, cl=2, use_cache=False, include_scores=True)
        if realtime_individuals:
            print("실시간 개인 유사도 (상세):")
            for individual in realtime_individuals:
                print(f"  {individual['emp_no']} ({individual['emp_name']}) - {individual['position']} - {individual['team_name']} (유사도: {individual['similarity_score']})")


def show_cache_info(analyzer):
    """캐시 정보 상세 출력"""
    print("\n=== 캐시 정보 ===")
    status = analyzer.get_cache_status()
    
    print(f"캐시 파일 존재: {status['cache_file_exists']}")
    print(f"팀 데이터 캐시: {status['teams_cached']}")
    print(f"개인 데이터 분기: {status['individuals_quarters']}")
    
    if status['metadata']:
        metadata = status['metadata']
        print(f"\n메타데이터:")
        print(f"  생성일: {metadata.get('created_at', 'N/A')}")
        print(f"  마지막 업데이트: {metadata.get('last_updated', 'N/A')}")
        print(f"  팀 업데이트: {metadata.get('teams_updated_at', 'N/A')}")
        print(f"  개인 마지막 분기: {metadata.get('individuals_last_quarter', 'N/A')}")


if __name__ == "__main__":
    # 기본 실행 (캐시 사용)
    print("=== 기본 실행 (캐시 우선) ===")
    analyzer = run_similarity_analysis(period_id=2, use_cache=True, save_to_cache=True)
    
    if analyzer:
        # 캐시 정보 출력
        show_cache_info(analyzer)
        
        # 결과 테스트
        test_similarity_results(analyzer, period_id=2)
        
        print("\n=== 사용 가능한 메서드 ===")
        print("analyzer.get_similar_teams(team_id, use_cache=True)")
        print("analyzer.get_similar_individuals(emp_no, period_id, cl, use_cache=True)")
        print("analyzer.get_cache_status()")
        
        # 강제 재분석 예시
        print("\n=== 강제 재분석이 필요한 경우 ===")
        print("run_similarity_analysis(period_id=2, use_cache=False, save_to_cache=True)")
    else:
        print("분석 실패!")