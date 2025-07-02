#!/usr/bin/env python3
"""
개인 평가 리포트 생성기 실행 파일

이 스크립트는 개인 평가 리포트를 한국어 JSON 형태로 생성하고 
feedback_reports.report 컬럼에 저장합니다.

사용법:
    python agents/report/run_quarterly_individual_reports.py                    # 모든 직원 처리
    python agents/report/run_quarterly_individual_reports.py --emp_no SK0002   # 특정 직원만 처리
    python agents/report/run_quarterly_individual_reports.py --emp_no SK0002 --period-id 2
"""

import sys
import os
import argparse

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from agents.report.quarterly_individual_reports import main

def parse_arguments():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="개인 평가 리포트 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python agents/report/run_quarterly_individual_reports.py                    # 모든 직원 처리
  python agents/report/run_quarterly_individual_reports.py --emp_no SK0002   # 특정 직원만 처리 (모든 분기)
  python agents/report/run_quarterly_individual_reports.py --emp_no SK0002 --period-id 2  # 특정 직원의 특정 분기만 처리
  python agents/report/run_quarterly_individual_reports.py --teams 1 2 3     # 특정 팀들의 직원만 처리
  python agents/report/run_quarterly_individual_reports.py --teams 1 2 --period-id 2  # 특정 팀들의 특정 분기만 처리
        """
    )
    
    parser.add_argument(
        "--emp_no",
        type=str,
        help="처리할 특정 직원 번호 (예: SK0002). 생략하면 모든 직원을 처리합니다."
    )
    
    parser.add_argument(
        "--period-id",
        type=int,
        help="처리할 특정 분기 ID (예: 1, 2, 3, 4). 생략하면 모든 분기를 처리합니다."
    )
    
    parser.add_argument(
        "--teams",
        type=int,
        nargs='+',
        help="처리할 특정 팀 ID들 (예: --teams 1 2 3). 생략하면 모든 팀을 처리합니다."
    )
    
    return parser.parse_args()

def main_wrapper():
    """메인 실행 래퍼 함수"""
    try:
        args = parse_arguments()
        
        print("🚀 개인 평가 리포트 생성기 시작")
        print("=" * 60)
        
        if args.emp_no and args.period_id:
            print(f"🎯 특정 직원의 특정 분기 처리 모드: {args.emp_no}님 (분기 ID: {args.period_id})")
            main(emp_no=args.emp_no, period_id=args.period_id)
        elif args.emp_no:
            print(f"🎯 특정 직원 처리 모드: {args.emp_no}님 (모든 분기)")
            main(emp_no=args.emp_no)
        elif args.teams and args.period_id:
            print(f"🎯 특정 팀들의 특정 분기 처리 모드: 팀 {args.teams} (분기 ID: {args.period_id})")
            main(teams=args.teams, period_id=args.period_id)
        elif args.teams:
            print(f"🎯 특정 팀들 처리 모드: 팀 {args.teams} (모든 분기)")
            main(teams=args.teams)
        else:
            print("📊 전체 직원 처리 모드")
            main()
            
        print("=" * 60)
        print("✅ 개인 평가 리포트 생성기 완료")
        
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