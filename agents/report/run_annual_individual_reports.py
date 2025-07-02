# =====================================
# 연말 개인 최종 평가 리포트 생성기 실행 스크립트
# =====================================
# 목적: 연말 개인 최종 평가 리포트를 생성하여 DB에 저장
# - final_evaluation_reports 테이블의 AI 분석 결과를 한국어 리포트로 변환
# - 생성된 리포트는 final_evaluation_reports.report 컬럼에 JSON 형태로 저장
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 직원 연말 개인 평가 리포트 생성:
#   python agents/report/run_annual_individual_reports.py
# 특정 직원만 리포트 생성 (예: SK0002):
#   python agents/report/run_annual_individual_reports.py --emp_no SK0002
# 특정 분기만 리포트 생성 (예: 분기 4):
#   python agents/report/run_annual_individual_reports.py --period-id 4
# 특정 팀만 리포트 생성 (예: 팀 1,2):
#   python agents/report/run_annual_individual_reports.py --teams 1,2
# 특정 팀의 특정 분기만 리포트 생성:
#   python agents/report/run_annual_individual_reports.py --period-id 4 --teams 1,2
# =====================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def parse_teams(teams_str: str) -> list:
    """팀 문자열을 리스트로 변환합니다."""
    if not teams_str:
        return []
    return [int(team_id.strip()) for team_id in teams_str.split(',')]

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='연말 개인 최종 평가 리포트 생성기')
    parser.add_argument('--emp_no', type=str, 
        help='특정 직원 번호 (예: SK0002). 입력하지 않으면 모든 직원을 처리합니다.')
    parser.add_argument('--period-id', type=int,
        help='처리할 특정 분기 ID (예: 4). 입력하지 않으면 모든 분기를 처리합니다.')
    parser.add_argument('--teams', type=str,
        help='처리할 특정 팀 ID들 (예: 1,2,3). 입력하지 않으면 모든 팀을 처리합니다.')
    
    args = parser.parse_args()
    
    print("🚀 연말 개인 최종 평가 리포트 생성기 실행 시작")
    print("=" * 60)
    
    try:
        from agents.report.annual_individual_reports import main as run_final_reports
        
        teams = parse_teams(args.teams) if args.teams else None
        
        if args.emp_no:
            print(f"🎯 특정 직원 처리 모드: {args.emp_no}님")
            run_final_reports(emp_no=args.emp_no)
        elif args.period_id and teams:
            print(f"🎯 특정 팀 {teams}, 분기 {args.period_id} 처리 모드")
            run_final_reports(period_id=args.period_id, teams=teams)
        elif args.period_id:
            print(f"🎯 특정 분기 처리 모드: 분기 ID {args.period_id}")
            run_final_reports(period_id=args.period_id)
        elif teams:
            print(f"🎯 특정 팀 처리 모드: 팀 {teams} (모든 분기)")
            run_final_reports(teams=teams)
        else:
            print("📊 전체 직원 처리 모드")
            run_final_reports()
            
        print("\n" + "=" * 60)
        print("✅ 연말 개인 최종 평가 리포트 생성 완료!")
    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print("프로젝트 루트 디렉토리에서 실행하거나 Python 경로를 확인해주세요.")
        return 1
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 