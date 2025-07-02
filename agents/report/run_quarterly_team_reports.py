# =====================================
# 팀 평가 리포트 생성기 실행 스크립트
# =====================================
# 목적: AI 기반 팀 평가 데이터를 한국어 JSON 리포트로 변환하여 DB에 저장
# - team_evaluations 테이블의 AI 분석 결과를 한국어 리포트로 변환
# - 팀별 KPI, 협업 매트릭스, 코칭 제안, 리스크 분석 등을 포함
# - 생성된 리포트는 team_evaluations.report 컬럼에 JSON 형태로 저장
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 팀 평가 리포트 생성:
#   python agents/report/run_quarterly_team_reports.py
# 특정 분기만 리포트 생성 (예: 분기 2):
#   python agents/report/run_quarterly_team_reports.py --period-id 2
# 특정 팀만 리포트 생성 (예: 팀 1,2):
#   python agents/report/run_quarterly_team_reports.py --teams 1,2
# 특정 팀의 특정 분기만 리포트 생성:
#   python agents/report/run_quarterly_team_reports.py --period-id 2 --teams 1,2
# =====================================
# 생성되는 리포트 구조
# =====================================
# - 기본_정보: 팀명, 팀장명, 업무수행기간
# - 팀_종합_평가: 평균달성률, 유사팀비교, 성과분석
# - 팀_업무_목표_및_달성률: KPI별 달성률 및 분석
# - 팀원_성과_분석: 팀원별 기여도 및 순위
# - 협업_네트워크: 협업 매트릭스 및 협업률 분석
# - 팀원별_코칭_제안: 일반코칭 및 집중코칭 제안
# - 리스크_및_향후_운영_제안: 주요 리스크 및 개선 전략
# - 총평: 종합 인사이트 및 평가
# =====================================

#!/usr/bin/env python3
"""
팀 평가 리포트 생성기 실행 파일

이 스크립트는 팀 평가 리포트를 한국어 JSON 형태로 생성하고 
team_evaluations.report 컬럼에 저장합니다.

사용법:
    python agents/report/run_quarterly_team_reports.py                    # 모든 팀 처리
    python agents/report/run_quarterly_team_reports.py --period-id 2     # 특정 분기만 처리
    python agents/report/run_quarterly_team_reports.py --teams 1,2       # 특정 팀만 처리
    python agents/report/run_quarterly_team_reports.py --period-id 2 --teams 1,2  # 특정 팀의 특정 분기만 처리
"""

import sys
import os
import argparse

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from agents.report.quarterly_team_reports import main

def parse_arguments():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="팀 평가 리포트 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python agents/report/run_quarterly_team_reports.py                    # 모든 팀 처리
  python agents/report/run_quarterly_team_reports.py --period-id 2     # 특정 분기만 처리
  python agents/report/run_quarterly_team_reports.py --teams 1,2       # 특정 팀만 처리 (모든 분기)
  python agents/report/run_quarterly_team_reports.py --period-id 2 --teams 1,2  # 특정 팀의 특정 분기만 처리
        """
    )
    
    parser.add_argument(
        "--period-id",
        type=int,
        help="처리할 특정 분기 ID (예: 1, 2, 3, 4). 생략하면 모든 분기를 처리합니다."
    )
    
    parser.add_argument(
        "--teams",
        type=str,
        help="처리할 특정 팀 ID들 (예: 1,2,3). 생략하면 모든 팀을 처리합니다."
    )
    
    return parser.parse_args()

def parse_teams(teams_str: str) -> list:
    """팀 문자열을 리스트로 변환합니다."""
    if not teams_str:
        return []
    return [int(team_id.strip()) for team_id in teams_str.split(',')]

def main_wrapper():
    """메인 실행 래퍼 함수"""
    try:
        args = parse_arguments()
        
        print("🚀 팀 평가 리포트 생성기 시작")
        print("=" * 60)
        
        teams = parse_teams(args.teams) if args.teams else None
        
        if args.period_id and teams:
            print(f"🎯 특정 팀 {teams}, 분기 {args.period_id} 처리 모드")
            main(period_id=args.period_id, teams=teams)
        elif args.period_id:
            print(f"🎯 특정 분기 처리 모드: 분기 ID {args.period_id}")
            main(period_id=args.period_id)
        elif teams:
            print(f"🎯 특정 팀 처리 모드: 팀 {teams} (모든 분기)")
            main(teams=teams)
        else:
            print("📊 전체 팀 처리 모드")
            main()
            
        print("=" * 60)
        print("✅ 팀 평가 리포트 생성기 완료")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main_wrapper() 