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
                "target_team_kpi_ids": kpi_ids
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
def run_phase3_global_modules(teams, period_id):
    logging.info("Phase3: (예시) 전사 모듈8,10,11 실행 (구조만)")
    # 실제 구현 필요시 모듈8,10,11 import 및 실행
    logging.info("Phase3: 완료")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period-id', type=int, required=True)
    parser.add_argument('--teams', help='1,2,3 or all', required=True)
    args = parser.parse_args()

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
    run_phase3_global_modules(teams, args.period_id)
    logging.info("분기별 평가 워크플로우 전체 완료!")

if __name__ == "__main__":
    main() 