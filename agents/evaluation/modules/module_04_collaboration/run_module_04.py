# ================================================================
# run_module3.py - 모듈 4 실행 파일
# ================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
from typing import Literal, Optional

from agents.evaluation.modules.module_04_collaboration.agent import *
from agents.evaluation.modules.module_04_collaboration.db_utils import *

# ================================================================
# 실행 함수들
# ================================================================

def run_module3_for_team_period(team_id: int, period_id: int, report_type: str, target_team_kpi_ids: Optional[list] = None):
    """
    모듈 4 전체 워크플로우 실행 함수
    Args:
        team_id: 팀 ID
        period_id: 평가 기간 ID (분기)
        report_type: 'quarterly' 또는 'annual'
        target_team_kpi_ids: 평가 대상 KPI ID 리스트 (None이면 자동 조회)
    """
    print(f"\n============================")
    print(f"[모듈 4] 팀 {team_id}, 기간 {period_id} ({'연말' if report_type == 'annual' else '분기'}) 협업 분석 실행")
    print(f"============================\n")
    
    # KPI ID 자동 조회 (지정되지 않은 경우)
    if target_team_kpi_ids is None:
        _, kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
    else:
        kpi_ids = target_team_kpi_ids
    
    # State 생성
    state = Module4State(
        report_type=report_type if report_type == "annual" else "quarterly",  # Literal 타입 보장
        team_id=team_id,
        period_id=period_id,
        target_team_kpi_ids=kpi_ids,
        # Optional 필드들을 명시적으로 초기화
        collaboration_relationships=None,
        individual_collaboration_analysis=None,
        team_collaboration_matrix=None,
        team_evaluation_id=None,
        messages=None
    )
    
    # 워크플로우 실행
    module3_graph = create_module4_graph()
    result = module3_graph.invoke(state)
    print(f"\n[완료] 팀 {team_id}, 기간 {period_id} 협업 분석 종료\n")
    return result

# ================================================================
# 메인 실행 부분
# ================================================================

if __name__ == "__main__":
    # 분기별 period_id와 report_type 매핑 (예시)
    period_map = {
        1: {"period_id": 1, "report_type": "quarterly"},
        2: {"period_id": 2, "report_type": "quarterly"},
        3: {"period_id": 3, "report_type": "quarterly"},
        4: {"period_id": 4, "report_type": "annual"},
    }
    
    parser = argparse.ArgumentParser(description="Module3 Collaboration Runner")
    parser.add_argument("--quarter", type=int, choices=[1,2,3,4], required=False, default=2, help="실행할 분기 (1,2,3,4). 기본값: 2")
    args = parser.parse_args()
    
    team_id = 1
    period_info = period_map[args.quarter]
    period_id = period_info["period_id"]
    report_type = period_info["report_type"]
    
    # KPI ID 자동 조회
    _, kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
    
    run_module3_for_team_period(
        team_id=team_id,
        period_id=period_id,
        report_type=report_type,
        target_team_kpi_ids=kpi_ids
    )