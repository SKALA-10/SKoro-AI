# =====================================
# 연말 1단계 평가 워크플로우
# =====================================
# 목적: AI 기반 팀별 평가 수행 (모듈2,3,6,4,7) + 연말 중간평가 리포트 생성 및 톤 조정
# Phase 1: 팀별 평가 (모듈2,3,6)
# - 모듈2: 목표달성도 분석
# - 모듈3: Peer Talk 분석  
# - 모듈6: 4P BARS 평가
# Phase 2: 팀별 평가 (모듈4)
# - 모듈4: 협업 분석
# Phase 3: 팀별 평가 (모듈7)
# - 모듈7: 종합평가 점수 산정
# Phase 4: 연말 중간평가 리포트 생성 및 톤 조정
# - 4.1: 연말 중간평가 리포트 생성
# - 4.2: 팀 중간평가 리포트 톤 조정
# 완료 후: 팀장이 프론트엔드에서 수정 및 제출 필요
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 팀 자동 실행 (연말 1단계):
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --auto
# 특정 팀만 자동 실행 (예: 팀 1):
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --auto --teams 1
# 여러 팀 지정 (예: 팀 1, 3, 5):
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --auto --teams 1,3,5
# 특정 단계만 실행:
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --phase 1 --teams 1
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --phase 2 --teams 1
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --phase 3 --teams 1
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --phase 4 --teams 1
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --phase 4.1 --teams 1
#   python agents/workflow/annual_phase1_workflow.py --period-id 4 --phase 4.2 --teams 1
# =====================================

import argparse
import logging
from agents.workflow.workflow_utils import (
    get_target_teams, run_team_module_with_retry, check_all_teams_phase_completed, update_team_status, parse_teams
)
from agents.evaluation.modules.module_02_goal_achievement.agent import create_module2_graph
from agents.evaluation.modules.module_03_peer_talk.agent import create_module3_graph
from agents.evaluation.modules.module_04_collaboration.agent import create_module4_graph
from agents.evaluation.modules.module_06_4p_evaluation.agent import create_module6_graph_efficient
from agents.evaluation.modules.module_07_final_evaluation.agent import create_team_module7_graph
from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_tasks_and_kpis, fetch_team_members
import sys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

# Phase 1: 모듈2,3,6 (팀별)
def run_phase1_all_teams(teams, period_id):
    logging.info("Phase1: 모듈2,3,6 실행 시작")
    for idx, team_id in enumerate(teams, 1):
        logging.info(f"[Phase1] 팀 {team_id} ({idx}/{len(teams)}) 시작")
        # 1. 모듈2 (목표달성도)
        def module2_func(team_id, period_id):
            task_ids, kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
            state = {
                "report_type": "annual",
                "team_id": team_id,
                "period_id": period_id,
                "target_task_ids": task_ids,
                "target_team_kpi_ids": kpi_ids,
                # 연말 평가에서 필요한 키들 추가
                "feedback_report_ids": [],
                "final_evaluation_report_ids": [],
                "updated_task_ids": [],
                "updated_team_kpi_ids": [],
                "team_evaluation_id": None,
                "team_context_guide": {},
                "messages": []
            }
            graph = create_module2_graph()
            graph.invoke(state)
            return True
        run_team_module_with_retry(team_id, module2_func, period_id)
        # 2. 모듈3 (Peer Talk)
        def module3_func(team_id, period_id):
            members = fetch_team_members(team_id)
            for member in members:
                if member.get('role') == 'MANAGER':
                    continue
                state = {
                    "team_id": team_id,  # team_id 추가
                    "period_id": period_id,
                    "target_emp_no": member['emp_no'],
                    "peer_evaluation_ids": [],
                    "evaluator_emp_nos": [],
                    "evaluation_weights": [],
                    "keyword_collections": [],
                    "task_summaries": [],
                    "peer_evaluation_summary_sentences": [],
                    "strengths": [],
                    "concerns": [],
                    "collaboration_observations": [],
                    "weighted_analysis_result": {},
                    "feedback_report_id": None,
                    "final_evaluation_report_id": None,
                    "messages": []
                }
                graph = create_module3_graph()
                graph.invoke(state)
            return True
        run_team_module_with_retry(team_id, module3_func, period_id)
        # 3. 모듈6 (4P BARS)
        def module6_func(team_id, period_id):
            members = fetch_team_members(team_id)
            for member in members:
                if member.get('role') == 'MANAGER':
                    continue
                state = {
                    "report_type": "annual",
                    "team_id": team_id,
                    "period_id": period_id,
                    "emp_no": member['emp_no'],
                    "feedback_report_id": None,
                    "final_evaluation_report_id": None,
                    "raw_evaluation_criteria": "",
                    "evaluation_criteria": {},
                    "evaluation_results": {},
                    "integrated_data": {},
                    "messages": []
                }
                graph = create_module6_graph_efficient()
                graph.invoke(state)
            return True
        run_team_module_with_retry(team_id, module6_func, period_id)
        update_team_status(team_id, period_id, "AI_PHASE1_COMPLETED")
        logging.info(f"[Phase1] 팀 {team_id} 완료")

# Phase 2: 모듈4 (팀별)
def run_phase2_all_teams(teams, period_id):
    logging.info("Phase2: 모듈4 실행 시작")
    for idx, team_id in enumerate(teams, 1):
        logging.info(f"[Phase2] 팀 {team_id} ({idx}/{len(teams)}) 시작")
        def module4_func(team_id, period_id):
            _, kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
            state = {
                "report_type": "annual",
                "team_id": team_id,
                "period_id": period_id,
                "target_team_kpi_ids": kpi_ids,
                "collaboration_relationships": None,
                "individual_collaboration_analysis": None,
                "team_collaboration_matrix": None,
                "team_evaluation_id": None,
                "messages": None
            }
            graph = create_module4_graph()
            graph.invoke(state)
            return True
        run_team_module_with_retry(team_id, module4_func, period_id)
        update_team_status(team_id, period_id, "AI_PHASE2_COMPLETED")
        logging.info(f"[Phase2] 팀 {team_id} 완료")

# Phase 3: 모듈7 (팀별)
def run_phase3_all_teams(teams, period_id):
    logging.info("Phase3: 모듈7 실행 시작")
    for idx, team_id in enumerate(teams, 1):
        logging.info(f"[Phase3] 팀 {team_id} ({idx}/{len(teams)}) 시작")
        def module7_func(team_id, period_id):
            state = {
                "report_type": "annual",
                "team_id": team_id,
                "period_id": period_id,
                "messages": []
            }
            graph = create_team_module7_graph()
            graph.invoke(state)
            return True
        run_team_module_with_retry(team_id, module7_func, period_id)
        update_team_status(team_id, period_id, "AI_PHASE3_COMPLETED")
        logging.info(f"[Phase3] 팀 {team_id} 완료")

# Phase 4.1: 연말 중간평가 리포트 생성
def run_phase4_1_middle_reports(period_id: int, teams):
    """
    Phase4.1: 연말 중간평가 리포트 생성
    """
    logging.info("Phase4.1: 연말 중간평가 리포트 생성 시작")
    
    from agents.report.annual_middle_reports import main as generate_middle_reports
    generate_middle_reports(period_id=period_id, teams=teams)
    
    logging.info("Phase4.1: 연말 중간평가 리포트 생성 완료!")

def run_phase4_2_team_interim_tone_adjustment(period_id: int, teams):
    """
    Phase4.2: 팀 중간평가 리포트 톤 조정 및 DB 업데이트
    """
    logging.info("Phase4.2: 팀 중간평가 톤 조정 시작")
    
    try:
        # LLM 클라이언트 초기화
        from langchain_openai import ChatOpenAI
        llm_client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        # 팀 중간평가 톤 조정 실행
        from agents.tone_adjustment.team_tone_adjustment import TeamLeaderToneAdjustmentAgent
        from agents.tone_adjustment.run_team_tone_adjustment import run_team_tone_adjustment_for_teams
        
        result = run_team_tone_adjustment_for_teams(period_id, teams, llm_client, report_type="team_interim_evaluation")
        
        logging.info("Phase4.2: 팀 중간평가 톤 조정 완료!")
        return result
        
    except Exception as e:
        logging.error(f"❌ 팀 중간평가 톤 조정 실행 중 오류: {e}")
        return {"status": "error", "message": str(e)}

def run_phase4_reports(period_id: int, teams):
    """
    Phase4: 연말 중간평가 리포트 생성 → 톤 조정 순서로 실행
    """
    logging.info("Phase4: 전체 리포트 프로세스 시작")
    
    # 4.1: 연말 중간평가 리포트 생성
    try:
        logging.info("[Phase4] 중간평가 리포트 생성 시작")
        run_phase4_1_middle_reports(period_id, teams)
        logging.info("[Phase4] 중간평가 리포트 생성 완료")
    except Exception as e:
        logging.error(f"[Phase4] 중간평가 리포트 생성 실패: {e}")
    
    # 4.2: 팀 중간평가 톤 조정
    try:
        logging.info("[Phase4] 팀 중간평가 톤 조정 시작")
        tone_result = run_phase4_2_team_interim_tone_adjustment(period_id, teams)
        logging.info("[Phase4] 팀 중간평가 톤 조정 완료")
    except Exception as e:
        logging.error(f"[Phase4] 팀 중간평가 톤 조정 실패: {e}")
        tone_result = None
    
    logging.info("Phase4: 전체 리포트 프로세스 완료!")
    return tone_result

def run_auto_workflow(period_id: int, specific_teams=None):
    """
    --auto 옵션: Phase1/2/3(팀별 평가) → 동기화 체크 → Phase4(연말 중간평가 리포트 생성)까지 자동 실행
    """
    logging.info("[AUTO] 연말 1단계 평가 자동 실행 시작")
    teams = get_target_teams(period_id, specific_teams)
    logging.info(f"[AUTO] 평가 대상 팀: {teams}")

    # Phase1/2/3: 팀별 평가
    run_phase1_all_teams(teams, period_id)
    run_phase2_all_teams(teams, period_id)
    run_phase3_all_teams(teams, period_id)

    # 동기화 체크
    all_completed = check_all_teams_phase_completed(teams, period_id, 'AI_PHASE3_COMPLETED')
    if not all_completed:
        logging.warning(f"[AUTO] 일부 팀이 아직 Phase3를 완료하지 못했습니다.")
    else:
        logging.info("[AUTO] 모든 팀이 Phase3를 완료했습니다.")

    # Phase4: 연말 중간평가 리포트 생성
    run_phase4_reports(period_id, teams)
    
    logging.info("[AUTO] 연말 1단계 평가 자동 실행 완료!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period-id', type=int, required=True)
    parser.add_argument('--teams', help='1,2,3 or all', required=False, default=None)
    parser.add_argument('--auto', action='store_true', help='모든 단계 자동 실행')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', '4', '4.1', '4.2'], help='특정 Phase만 실행')
    args = parser.parse_args()

    # 팀 목록 파싱
    if args.teams and args.teams != 'all':
        team_list = parse_teams(args.teams)
    else:
        team_list = None

    # --phase 옵션: 특정 Phase만 실행
    if args.phase:
        teams = get_target_teams(args.period_id, team_list)
        logging.info(f"[Phase{args.phase}] {len(teams)}개 팀 실행")
        
        if args.phase == '1':
            run_phase1_all_teams(teams, args.period_id)
        elif args.phase == '2':
            run_phase2_all_teams(teams, args.period_id)
        elif args.phase == '3':
            run_phase3_all_teams(teams, args.period_id)
        elif args.phase == '4':
            run_phase4_reports(args.period_id, teams)
        elif args.phase == '4.1':
            run_phase4_1_middle_reports(args.period_id, teams)
        elif args.phase == '4.2':
            run_phase4_2_team_interim_tone_adjustment(args.period_id, teams)
        
        logging.info(f"[Phase{args.phase}] 완료!")
        sys.exit(0)

    # --auto 옵션: 전체 or 특정 팀만 자동 실행
    if args.auto:
        if team_list:
            logging.info(f"[AUTO] 지정된 팀만 자동 실행: {team_list}")
        else:
            logging.info("[AUTO] 전체 팀 자동 실행")
        run_auto_workflow(args.period_id, team_list)
        sys.exit(0)

    teams = get_target_teams(args.period_id, team_list)
    logging.info(f"🚀 연말 1단계 평가 시작: {len(teams)}개 팀")

    run_phase1_all_teams(teams, args.period_id)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE1_COMPLETED"):
        logging.warning("일부 팀이 Phase1을 완료하지 못했습니다. 중단합니다.")
        return
    run_phase2_all_teams(teams, args.period_id)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE2_COMPLETED"):
        logging.warning("일부 팀이 Phase2를 완료하지 못했습니다. 중단합니다.")
        return
    run_phase3_all_teams(teams, args.period_id)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE3_COMPLETED"):
        logging.warning("일부 팀이 Phase3를 완료하지 못했습니다. 중단합니다.")
        return
    
    # Phase4: 연말 중간평가 리포트 생성
    run_phase4_reports(args.period_id, teams)
    
    logging.info("연말 1단계 평가 워크플로우 완료! 팀장 수정 및 제출을 기다립니다.")

if __name__ == "__main__":
    main() 