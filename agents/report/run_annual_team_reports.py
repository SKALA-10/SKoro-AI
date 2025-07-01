# =====================================
# 연간 팀 평가 리포트 생성기 실행 스크립트
# =====================================
# 목적: Period 4 전용 최종 팀 평가 리포트를 생성하여 DB에 저장
# - team_evaluations 테이블의 AI 분석 결과를 한국어 리포트로 변환
# - 팀 종합 평가, 업무 목표 달성률, 팀 성과 요약, 리스크 분석 등을 포함
# - 생성된 리포트는 team_evaluations.report 컬럼에 JSON 형태로 저장
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 Period 4 팀 평가 리포트 생성:
#   python agents/report/run_annual_team_reports.py
# 특정 팀 평가 ID만 리포트 생성 (예: ID 104):
#   python agents/report/run_annual_team_reports.py --team_evaluation_id 104
# =====================================
# 생성되는 리포트 구조
# =====================================
# - 기본_정보: 팀명, 팀장명, 업무수행기간, 평가구분
# - 팀_종합_평가: 평균달성률, 유사팀비교, 성과분석, 전분기대비추이
# - 팀_업무_목표_및_달성률: KPI별 달성률 및 분석
# - 팀_성과_요약: 업적, SK Values, 성과요약
# - 팀_조직력_및_리스크_요인: 주요 리스크 및 영향 예측
# - 다음_연도_운영_제안: 핵심 인력 운용 전략, 협업 구조 개선
# - 총평: 종합 의견
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

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='연간 팀 평가 리포트 생성기 (Period 4 전용)')
    parser.add_argument('--team_evaluation_id', type=int, 
        help='특정 팀 평가 ID (입력하지 않으면 모든 Period 4 팀 평가 처리)')
    
    args = parser.parse_args()
    
    print("🚀 연간 팀 평가 리포트 생성기 실행 시작")
    print("=" * 60)
    
    try:
        # annual_team_reports 모듈 임포트
        from agents.report.annual_team_reports import main as run_annual_reports
        
        # 리포트 생성 실행
        run_annual_reports(args.team_evaluation_id)
        
        print("\n" + "=" * 60)
        print("✅ 연간 팀 평가 리포트 생성 완료!")
        
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