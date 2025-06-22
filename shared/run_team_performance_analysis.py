# run_team_performance_analysis.py
# 팀 성과 비교 분석 실행 스크립트

from team_performance_comparator import TeamPerformanceComparator


def run_team_performance_analysis(team_id=1, period_id=2, force_recalculate=False):
    """팀 성과 비교 분석 통합 실행"""
    print("=== 팀 성과 비교 분석 시작 ===\n")
    
    comparator = TeamPerformanceComparator()
    
    # 클러스터 상태 확인
    print("1. 클러스터 상태 확인")
    status = comparator.get_cluster_status(period_id)
    print(f"캐시 파일 존재: {status['cache_file_exists']}")
    if status['cache_file_exists']:
        print(f"클러스터 수: {status.get('total_clusters', 'Unknown')}")
        print(f"전체 팀 수: {status.get('total_teams', 'Unknown')}")
        print(f"생성일: {status.get('created_at', 'Unknown')}")
        if 'cluster_distribution' in status:
            print("클러스터별 분포:")
            for cluster_id, team_count in status['cluster_distribution'].items():
                print(f"  클러스터 {cluster_id}: {team_count}개 팀")
    else:
        print("캐시 파일 없음 - 새로 생성됩니다.")
    
    # 팀 성과 분석 실행
    print(f"\n2. 팀 {team_id} 성과 분석 (Q{period_id})")
    result = comparator.analyze_team_cluster_performance(team_id, period_id, force_recalculate)
    
    if result["success"]:
        team_info = result["team_cluster_info"]
        print(f"\n✅ 분석 완료!")
        print(f"팀 {team_id} → 클러스터 {team_info['cluster_id']}")
        print(f"유사팀 {len(team_info['similar_teams'])}개: {team_info['similar_teams']}")
        print(f"클러스터 통계: 평균 {team_info['cluster_stats']['avg_rate']}% "
              f"(±{team_info['cluster_stats']['std_rate']}%)")
        print(f"비교 신뢰도: {team_info['reliability']}")
        
        return comparator
    else:
        print(f"\n❌ 분석 실패: {result['error']}")
        return None


def test_multiple_teams(team_ids=[1, 2, 3], period_id=2):
    """여러 팀 성과 분석 테스트"""
    print(f"=== 여러 팀 성과 분석 테스트 (Q{period_id}) ===\n")
    
    comparator = TeamPerformanceComparator()
    results = {}
    
    for team_id in team_ids:
        print(f"\n--- 팀 {team_id} 분석 ---")
        result = comparator.analyze_team_cluster_performance(team_id, period_id)
        results[team_id] = result
        
        if result["success"]:
            team_info = result["team_cluster_info"]
            print(f"클러스터: {team_info['cluster_id']}, "
                  f"유사팀: {len(team_info['similar_teams'])}개, "
                  f"신뢰도: {team_info['reliability']}")
        else:
            print(f"실패: {result['error']}")
    
    return results


def show_cluster_details(period_id=2):
    """클러스터 상세 정보 출력"""
    print(f"=== 클러스터 상세 정보 (Q{period_id}) ===\n")
    
    comparator = TeamPerformanceComparator()
    status = comparator.get_cluster_status(period_id)
    
    if not status['cache_file_exists']:
        print("클러스터 캐시 파일이 없습니다.")
        return
    
    # 클러스터 통계 로드
    cluster_stats = comparator.stats_manager.load_cluster_stats(period_id)
    
    print(f"총 {len(cluster_stats)}개 클러스터:")
    
    for cluster_id, stats in cluster_stats.items():
        print(f"\n클러스터 {cluster_id}:")
        print(f"  팀 수: {stats['team_count']}개")
        print(f"  팀 ID: {stats['teams']}")
        print(f"  평균 달성률: {stats['overall_stats']['avg_rate']}%")
        print(f"  표준편차: {stats['overall_stats']['std_rate']}%")
        print(f"  범위: {stats['overall_stats']['min_rate']}% ~ {stats['overall_stats']['max_rate']}%")
        
        # 신뢰도 표시
        reliability = comparator.stats_manager.get_comparison_reliability(stats['team_count'])
        print(f"  비교 신뢰도: {reliability}")


def force_recalculate_clusters(period_id=2):
    """클러스터 강제 재계산"""
    print(f"=== 클러스터 강제 재계산 (Q{period_id}) ===\n")
    
    comparator = TeamPerformanceComparator()
    
    print("기존 캐시를 무시하고 새로 계산합니다...")
    cluster_stats = comparator.stats_manager.calculate_cluster_performance_stats(period_id, force_recalculate=True)
    
    print(f"\n재계산 완료: {len(cluster_stats)}개 클러스터")
    
    return cluster_stats


if __name__ == "__main__":
    # 기본 실행
    print("=== 기본 실행 ===")
    comparator = run_team_performance_analysis(team_id=1, period_id=2)
    
    if comparator:
        print("\n=== 사용 가능한 함수 ===")
        print("run_team_performance_analysis(team_id, period_id, force_recalculate=False)")
        print("test_multiple_teams([1,2,3], period_id)")
        print("show_cluster_details(period_id)")
        print("force_recalculate_clusters(period_id)")
        
        # 클러스터 상세 정보 출력
        print("\n")
        show_cluster_details(period_id=2)
        
        # 예시: 여러 팀 테스트 (주석 처리)
        # print("\n")
        # test_multiple_teams([1, 2, 3], period_id=2)
        
        # 예시: 강제 재계산 (주석 처리)
        # print("\n")
        # force_recalculate_clusters(period_id=2)
    else:
        print("기본 실행 실패!")