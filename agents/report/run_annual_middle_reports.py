# =====================================
# 연간 중간평가 리포트 생성기 실행 스크립트
# =====================================
# 목적: Period 4 전용 팀장용 최종전 중간평가 리포트를 생성하여 DB에 저장
# - team_evaluations 테이블의 AI 분석 결과를 한국어 리포트로 변환
# - 팀원별 평가 요약표, 협업 네트워크, 팀원별 평가 근거 등을 포함
# - 생성된 리포트는 team_evaluations.middle_report 컬럼에 JSON 형태로 저장
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 Period 4 팀 평가 중간평가 리포트 생성:
#   python agents/report/run_annual_middle_reports.py
# 특정 팀 평가 ID만 중간평가 리포트 생성 (예: ID 104):
#   python agents/report/run_annual_middle_reports.py --team_evaluation_id 104
# =====================================
# 생성되는 리포트 구조
# =====================================
# - 기본_정보: 팀명, 팀장명, 업무수행기간
# - 팀원_평가_요약표: 팀원별 AI 추천 점수 및 핵심 기여 요약
# - 협업_네트워크: 협업 매트릭스 및 팀 협업 요약
# - 팀원별_평가_근거: 팀원별 상세 평가 근거 및 Peer Talk
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
    parser = argparse.ArgumentParser(description='연간 중간평가 리포트 생성기 (Period 4 전용)')
    parser.add_argument('--team_evaluation_id', type=int, 
        help='특정 팀 평가 ID (입력하지 않으면 모든 Period 4 팀 평가 처리)')
    
    args = parser.parse_args()
    
    print("🚀 연간 중간평가 리포트 생성기 실행 시작")
    print("=" * 60)
    
    try:
        # annual_middle_reports 모듈 임포트
        from annual_middle_reports import main as run_annual_reports
        
        # 리포트 생성 실행
        run_annual_reports(args.team_evaluation_id)
        
        print("\n" + "=" * 60)
        print("✅ 연간 중간평가 리포트 생성 완료!")
        
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