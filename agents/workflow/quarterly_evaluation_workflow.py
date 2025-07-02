# =====================================
# 분기별 평가 워크플로우
# =====================================
# 목적: 분기별 AI 평가 수행 (모듈2,3,6,4,8,10,11)
# Phase 1: 팀별 평가 (모듈2,3,6)
# - 모듈2: 목표달성도 분석
# - 모듈3: Peer Talk 분석
# - 모듈6: 4P BARS 평가
# Phase 2: 팀별 평가 (모듈4)
# - 모듈4: 협업 분석
# Phase 3: 전사 모듈 (모듈8,10,11)
# - 모듈8: 팀 성과 비교
# - 모듈10: 개인 성장 코칭
# - 모듈11: 팀 운영 리스크 분석
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 팀 자동 실행 (분기 2):
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto
# 특정 팀만 자동 실행 (예: 팀 1):
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto --teams 1
# 여러 팀 지정 (예: 팀 1, 3, 5):
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto --teams 1,3,5
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
from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_tasks_and_kpis, fetch_team_members
from agents.evaluation.modules.module_08_team_comparision.agent import create_module8_graph
from agents.evaluation.modules.module_10_growth_coaching.agent import create_module10_graph
from agents.evaluation.modules.module_11_team_coaching.agent import Module11TeamRiskManagementAgent
from agents.evaluation.modules.module_11_team_coaching.db_utils import Module11DataAccess, SQLAlchemyDBWrapper, engine
import asyncio
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
                "report_type": "quarterly",
                "team_id": team_id,
                "period_id": period_id,
                "target_task_ids": task_ids,
                "target_team_kpi_ids": kpi_ids,
                # 분기별 평가에서 필요한 키들 추가
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
                    "report_type": "quarterly",
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
                "report_type": "quarterly",
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

# Phase 3: 전사 모듈8,10,11 (구조만)
def run_phase3_all_modules(period_id: int, teams):
    """
    Phase3: 전사 모듈8(팀 성과 비교), 10(개인 성장 코칭), 11(팀 리스크) 실행
    """
    logging.info("Phase3: 전사 모듈8,10,11 실행 시작")
    logging.info(f"[Phase3] 전체 대상 팀: {teams}")

    # 1. 모듈8: 팀 성과 비교 (팀 단위)
    for team_id in teams:
        try:
            logging.info(f"[Phase3][모듈8] 팀 {team_id} 실행 시작")
            module8_graph = create_module8_graph()
            state8 = {
                "team_id": team_id,
                "period_id": period_id,
                "report_type": "quarterly",
                "messages": []
            }
            module8_graph.invoke(state8)
            logging.info(f"[Phase3][모듈8] 팀 {team_id} 완료")
        except Exception as e:
            logging.error(f"[Phase3][모듈8] 팀 {team_id} 실패: {e}")

    # 2. 모듈10: 개인 성장 코칭 (팀원별)
    for team_id in teams:
        try:
            members = fetch_team_members(team_id)
            for member in members:
                # 팀장 제외
                if member.get('role') == 'MANAGER':
                    continue
                emp_no = member["emp_no"]
                logging.info(f"[Phase3][모듈10] 팀 {team_id} - {emp_no} 실행 시작")
                module10_graph = create_module10_graph()
                state10 = {
                    "emp_no": emp_no,
                    "period_id": period_id,
                    "report_type": "quarterly",
                    "messages": [],
                    "basic_info": {},
                    "performance_data": {},
                    "peer_talk_data": {},
                    "fourp_data": {},
                    "collaboration_data": {},
                    "module7_score_data": {},
                    "module9_final_data": {},
                    "growth_analysis": {},
                    "focus_coaching_needed": False,
                    "focus_coaching_analysis": {},
                    "individual_growth_result": {},
                    "manager_coaching_result": {},
                    "overall_comment": "",
                    "storage_result": {},
                    "processing_status": "",
                    "error_messages": []
                }
                module10_graph.invoke(state10)
                logging.info(f"[Phase3][모듈10] 팀 {team_id} - {emp_no} 완료")
        except Exception as e:
            logging.error(f"[Phase3][모듈10] 팀 {team_id} 실패: {e}")

    # 3. 모듈11: 팀 리스크 (팀 단위, async)
    async def run_module11_for_all_teams():
        db_wrapper = SQLAlchemyDBWrapper(engine)
        data_access = Module11DataAccess(db_wrapper)
        agent11 = Module11TeamRiskManagementAgent(data_access)
        tasks = []
        for team_id in teams:
            try:
                from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_evaluation_id
                team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
                if not team_evaluation_id:
                    logging.error(f"[Phase3][모듈11] 팀 {team_id} team_evaluation_id 없음")
                    continue
                tasks.append(agent11.execute(team_id, period_id, team_evaluation_id))
            except Exception as e:
                logging.error(f"[Phase3][모듈11] 팀 {team_id} 실패: {e}")
        await asyncio.gather(*tasks)
        logging.info("[Phase3][모듈11] 전체 완료")

    asyncio.run(run_module11_for_all_teams())
    logging.info("Phase3: 완료")

def run_phase4_reports(period_id: int, teams):
    """
    Phase4: 개인별 리포트 → 팀별 리포트 순서로 생성
    """
    logging.info("Phase4: 리포트 생성 시작")
    
    # 1. 개인별 리포트 생성
    logging.info("[Phase4] 개인별 리포트 생성 시작")
    from agents.report.quarterly_individual_reports import main as generate_individual_reports
    generate_individual_reports(period_id=period_id, teams=teams)
    
    # 2. 팀별 리포트 생성
    logging.info("[Phase4] 팀별 리포트 생성 시작")
    from agents.report.quarterly_team_reports import main as generate_team_reports
    generate_team_reports(period_id=period_id, teams=teams)
    
    logging.info("Phase4: 리포트 생성 완료!")

def run_auto_workflow(period_id: int, specific_teams=None):
    """
    --auto 옵션: Phase1/2(팀별 평가) → 동기화 체크 → Phase3(전사 모듈) → Phase4(리포트 생성)까지 자동 실행
    """
    logging.info("[AUTO] 전체 평가 자동 실행 시작")
    teams = get_target_teams(period_id, specific_teams)
    logging.info(f"[AUTO] 평가 대상 팀: {teams}")

    # Phase1/2: 팀별 평가
    run_phase1_all_teams(teams, period_id)
    run_phase2_all_teams(teams, period_id)

    # 동기화 체크
    all_completed = check_all_teams_phase_completed(teams, period_id, 'AI_PHASE2_COMPLETED')
    if not all_completed:
        logging.warning(f"[AUTO] 일부 팀이 아직 Phase2를 완료하지 못했습니다. (계속 진행)")
    else:
        logging.info("[AUTO] 모든 팀이 Phase2를 완료했습니다.")

    # Phase3: 전사 모듈 자동 실행
    run_phase3_all_modules(period_id, teams)

    # Phase4: 리포트 생성 (개인별 + 팀별)
    run_phase4_reports(period_id, teams)
    logging.info("[AUTO] 전체 평가 자동 실행 완료!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period-id', type=int, required=True)
    parser.add_argument('--teams', help='1,2,3 or all', required=False, default=None)
    parser.add_argument('--auto', action='store_true', help='모든 단계 자동 실행')
    parser.add_argument('--phase4-only', action='store_true', help='Phase4(개인별 리포트 생성)만 실행')
    args = parser.parse_args()

    # --phase4-only 옵션: Phase4만 실행
    if args.phase4_only:
        if args.teams and args.teams != 'all':
            team_list = parse_teams(args.teams)
            logging.info(f"[Phase4-Only] 지정된 팀만 Phase4 실행: {team_list}")
        else:
            team_list = None
            logging.info("[Phase4-Only] 전체 팀 Phase4 실행")
        run_phase4_reports(args.period_id, team_list)
        sys.exit(0)

    # --auto 옵션: 전체 or 특정 팀만 자동 실행
    if args.auto:
        if args.teams and args.teams != 'all':
            team_list = parse_teams(args.teams)
            logging.info(f"[AUTO] 지정된 팀만 자동 실행: {team_list}")
        else:
            team_list = None
            logging.info("[AUTO] 전체 팀 자동 실행")
        run_auto_workflow(args.period_id, team_list)
        sys.exit(0)

    teams = get_target_teams(args.period_id, parse_teams(args.teams))
    logging.info(f"🚀 평가 시작: {len(teams)}개 팀")

    run_phase1_all_teams(teams, args.period_id)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE1_COMPLETED"):
        logging.warning("일부 팀이 Phase1을 완료하지 못했습니다. 중단합니다.")
        return
    run_phase2_all_teams(teams, args.period_id)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE2_COMPLETED"):
        logging.warning("일부 팀이 Phase2를 완료하지 못했습니다. 중단합니다.")
        return
    run_phase3_all_modules(args.period_id, teams)
    # Phase4: 리포트 생성 (개인별 + 팀별)
    run_phase4_reports(args.period_id, teams)
    logging.info("분기별 평가 워크플로우 전체 완료!")

if __name__ == "__main__":
    main() 