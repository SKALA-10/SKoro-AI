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
# 특정 팀 평가 ID만 리포트 생성 (예: ID 102):
#   python agents/report/run_quarterly_team_reports.py --team_evaluation_id 102
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
    parser = argparse.ArgumentParser(description='팀 평가 리포트 생성기')
    parser.add_argument('--team_evaluation_id', type=int, 
        help='특정 팀 평가 ID (입력하지 않으면 모든 팀 평가 처리)')
    
    args = parser.parse_args()
    
    print("🚀 팀 평가 리포트 생성기 실행 시작")
    print("=" * 60)
    
    try:
        # quarterly_team_reports 모듈 임포트
        from quarterly_team_reports import main as run_quarterly_reports
        
        # 리포트 생성 실행
        run_quarterly_reports(args.team_evaluation_id)
        
        print("\n" + "=" * 60)
        print("✅ 팀 평가 리포트 생성 완료!")
        
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