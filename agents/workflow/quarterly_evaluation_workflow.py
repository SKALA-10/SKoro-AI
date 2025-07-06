# =====================================
# 분기별 평가 워크플로우
# =====================================
# 목적: 분기별 AI 평가 수행 (모듈2,3,4,6,8,10,11)
# Phase 1: 팀별 평가 (모듈2,3,4,6 순차 실행)
# - 모듈2: 목표달성도 분석
# - 모듈3: Peer Talk 분석  
# - 모듈4: 협업 분석
# - 모듈6: 4P BARS 평가
# Phase 2: 전사 모듈 (모듈8,10,11 순차 실행)
# - 모듈8: 팀 성과 비교
# - 모듈10: 개인 성장 코칭
# - 모듈11: 팀 운영 리스크 분석
# Phase 3: 리포트 생성 + 톤 조정
# - 개인별 리포트 생성
# - 팀별 리포트 생성
# - 개인별 톤 조정
# - 팀별 톤 조정
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 팀 자동 실행 (분기 2):
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto
# 특정 팀만 자동 실행 (예: 팀 1):
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto --teams 1
# 여러 팀 지정 (예: 팀 1, 3, 5):
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto --teams 1,3,5
# 특정 단계만 실행:
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 1 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 2 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 3 --teams 1
# 특정 모듈만 실행:
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 2 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 3 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 4 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 6 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 8 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 10 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 11 --teams 1
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

# Phase 1: 모듈2,3,4,6 순차 실행 (팀별)
def run_phase1_all_teams(teams, period_id):
    logging.info("Phase1: 모듈2,3,4,6 순차 실행 시작")
    
    for idx, team_id in enumerate(teams, 1):
        logging.info(f"[Phase1] 팀 {team_id} ({idx}/{len(teams)}) 시작")
        
        # 1. 모듈2 (목표달성도)
        logging.info(f"[Phase1][모듈2] 팀 {team_id} 실행")
        def module2_func(team_id, period_id):
            task_ids, kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
            state = {
                "report_type": "quarterly",
                "team_id": team_id,
                "period_id": period_id,
                "target_task_ids": task_ids,
                "target_team_kpi_ids": kpi_ids,
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
        logging.info(f"[Phase1][모듈3] 팀 {team_id} 실행")
        def module3_func(team_id, period_id):
            members = fetch_team_members(team_id)
            for member in members:
                if member.get('role') == 'MANAGER':
                    continue
                state = {
                    "team_id": team_id,
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
        
        # 3. 모듈4 (협업 분석)
        logging.info(f"[Phase1][모듈4] 팀 {team_id} 실행")
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
        
        # 4. 모듈6 (4P BARS)
        logging.info(f"[Phase1][모듈6] 팀 {team_id} 실행")
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

# Phase 2: 전사 모듈8,10,11 순차 실행
def run_phase2_all_modules(period_id: int, teams):
    """
    Phase2: 전사 모듈8(팀 성과 비교), 10(개인 성장 코칭), 11(팀 리스크) 순차 실행
    """
    logging.info("Phase2: 전사 모듈8,10,11 순차 실행 시작")
    logging.info(f"[Phase2] 전체 대상 팀: {teams}")

    # 1. 모듈8: 팀 성과 비교 (팀 단위)
    logging.info("[Phase2][모듈8] 팀 성과 비교 시작")
    for team_id in teams:
        try:
            logging.info(f"[Phase2][모듈8] 팀 {team_id} 실행")
            module8_graph = create_module8_graph()
            state8 = {
                "team_id": team_id,
                "period_id": period_id,
                "report_type": "quarterly",
                "messages": []
            }
            module8_graph.invoke(state8)
            logging.info(f"[Phase2][모듈8] 팀 {team_id} 완료")
        except Exception as e:
            logging.error(f"[Phase2][모듈8] 팀 {team_id} 실패: {e}")

    # 2. 모듈10: 개인 성장 코칭 (팀원별)
    logging.info("[Phase2][모듈10] 개인 성장 코칭 시작")
    for team_id in teams:
        try:
            members = fetch_team_members(team_id)
            for member in members:
                # 팀장 제외
                if member.get('role') == 'MANAGER':
                    continue
                emp_no = member["emp_no"]
                logging.info(f"[Phase2][모듈10] 팀 {team_id} - {emp_no} 실행")
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
                logging.info(f"[Phase2][모듈10] 팀 {team_id} - {emp_no} 완료")
        except Exception as e:
            logging.error(f"[Phase2][모듈10] 팀 {team_id} 실패: {e}")

    # 3. 모듈11: 팀 리스크 (팀 단위, async)
    logging.info("[Phase2][모듈11] 팀 리스크 분석 시작")
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
                    logging.error(f"[Phase2][모듈11] 팀 {team_id} team_evaluation_id 없음")
                    continue
                tasks.append(agent11.execute(team_id, period_id, team_evaluation_id))
            except Exception as e:
                logging.error(f"[Phase2][모듈11] 팀 {team_id} 실패: {e}")
        await asyncio.gather(*tasks)
        logging.info("[Phase2][모듈11] 전체 완료")

    asyncio.run(run_module11_for_all_teams())
    
    # Phase2 완료 후 상태 업데이트
    for team_id in teams:
        update_team_status(team_id, period_id, "AI_PHASE2_COMPLETED")
    
    logging.info("Phase2: 완료")

# Phase 3: 리포트 생성 + 톤 조정
def run_phase3_reports_and_tone(period_id: int, teams):
    """
    Phase3: 개인별 리포트 생성 → 팀별 리포트 생성 → 개인별 톤 조정 → 팀별 톤 조정
    """
    logging.info("Phase3: 리포트 생성 + 톤 조정 시작")
    
    # 1. 개인별 리포트 생성
    logging.info("[Phase3] 개인별 리포트 생성 시작")
    from agents.report.quarterly_individual_reports import main as generate_individual_reports
    generated_reports = generate_individual_reports(period_id=period_id, teams=teams, return_json=True)
    
    # 2. 팀별 리포트 생성
    logging.info("[Phase3] 팀별 리포트 생성 시작")
    from agents.report.quarterly_team_reports import main as generate_team_reports
    generate_team_reports(period_id=period_id, teams=teams)
    
    # 3. 개인별 + 팀별 톤 조정
    logging.info("[Phase3] 톤 조정 시작")
    completed_teams = []
    
    for team_id in teams:
        logging.info(f"[Phase3] 팀 {team_id} 톤 조정 시작")
        
        # 개인별 톤 조정
        individual_success = False
        try:
            logging.info(f"[Phase3] 팀 {team_id} 개인별 톤 조정 시작")
            from agents.tone_adjustment.run_individual_tone_adjustment import main as run_individual_tone_adjustment
            individual_result = run_individual_tone_adjustment(period_id=period_id, teams=[team_id])
            individual_success = True
            logging.info(f"[Phase3] 팀 {team_id} 개인별 톤 조정 완료")
        except Exception as e:
            logging.error(f"[Phase3] 팀 {team_id} 개인별 톤 조정 실패: {e}")
            individual_result = None
        
        # 팀별 톤 조정
        team_success = False
        try:
            logging.info(f"[Phase3] 팀 {team_id} 팀별 톤 조정 시작")
            import agents.tone_adjustment.run_team_tone_adjustment as team_tone_module
            team_result = team_tone_module.main(period_id=period_id, teams=[team_id])
            team_success = True
            logging.info(f"[Phase3] 팀 {team_id} 팀별 톤 조정 완료")
        except Exception as e:
            logging.error(f"[Phase3] 팀 {team_id} 팀별 톤 조정 실패: {e}")
            team_result = None
        
        # 둘 다 성공한 팀만 COMPLETED 업데이트
        if individual_success and team_success:
            update_team_status(team_id, period_id, "COMPLETED")
            completed_teams.append(team_id)
            logging.info(f"[Phase3] 팀 {team_id} 최종 완료 상태 업데이트")
        else:
            logging.warning(f"[Phase3] 팀 {team_id} 톤 조정 실패로 COMPLETED 상태 업데이트 건너뜀")
    
    logging.info(f"Phase3: 완료된 팀 {len(completed_teams)}/{len(teams)}")
    logging.info("Phase3: 전체 완료!")

def run_auto_workflow(period_id: int, specific_teams=None):
    """
    --auto 옵션: Phase1 → Phase2 → Phase3까지 자동 실행
    """
    logging.info("[AUTO] 전체 평가 자동 실행 시작")
    teams = get_target_teams(period_id, specific_teams)
    logging.info(f"[AUTO] 평가 대상 팀: {teams}")

    # Phase1: 팀별 평가 (모듈2,3,4,6)
    run_phase1_all_teams(teams, period_id)
    
    # Phase1 완료 체크
    all_completed = check_all_teams_phase_completed(teams, period_id, 'AI_PHASE1_COMPLETED')
    if not all_completed:
        logging.error(f"[AUTO] 일부 팀이 Phase1을 완료하지 못했습니다. Phase2를 실행할 수 없습니다.")
        return
    else:
        logging.info("[AUTO] 모든 팀이 Phase1을 완료했습니다.")

    # Phase2: 전사 모듈 (모듈8,10,11)
    run_phase2_all_modules(period_id, teams)
    
    # Phase2 완료 체크
    all_completed = check_all_teams_phase_completed(teams, period_id, 'AI_PHASE2_COMPLETED')
    if not all_completed:
        logging.error(f"[AUTO] 일부 팀이 Phase2를 완료하지 못했습니다. Phase3를 실행할 수 없습니다.")
        return
    else:
        logging.info("[AUTO] 모든 팀이 Phase2를 완료했습니다.")

    # Phase3: 리포트 생성 + 톤 조정
    run_phase3_reports_and_tone(period_id, teams)
    
    logging.info("[AUTO] 전체 평가 자동 실행 완료!")

def main():
    parser = argparse.ArgumentParser(
        description="분기별 평가 워크플로우",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 자동 실행
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto
  
  # 특정 팀만 자동 실행
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --auto --teams 1
  
  # 특정 단계만 실행
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 1 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 2 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 3 --teams 1
  
  # 특정 모듈만 실행
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 2 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 3 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 4 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 6 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 8 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 10 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --module 11 --teams 1
        """
    )
    parser.add_argument('--period-id', type=int, required=True, help='분기 ID (예: 2)')
    parser.add_argument('--teams', help='팀 ID (예: 1,2,3 또는 all)', required=False, default=None)
    parser.add_argument('--auto', action='store_true', help='모든 단계 자동 실행')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3'], help='특정 Phase만 실행')
    parser.add_argument('--module', type=int, choices=[2, 3, 4, 6, 8, 10, 11], help='특정 모듈만 실행')
    args = parser.parse_args()

    # 팀 목록 파싱
    if args.teams and args.teams != 'all':
        team_list = parse_teams(args.teams)
    else:
        team_list = None

    # --auto 옵션: 전체 자동 실행
    if args.auto:
        if team_list:
            logging.info(f"[AUTO] 지정된 팀만 자동 실행: {team_list}")
        else:
            logging.info("[AUTO] 전체 팀 자동 실행")
        run_auto_workflow(args.period_id, team_list)
        sys.exit(0)

    # --phase 옵션: 특정 Phase만 실행
    if args.phase:
        teams = get_target_teams(args.period_id, team_list)
        logging.info(f"[Phase{args.phase}] {len(teams)}개 팀 실행")
        
        if args.phase == '1':
            run_phase1_all_teams(teams, args.period_id)
        elif args.phase == '2':
            # Phase1 완료 체크
            if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE1_COMPLETED"):
                logging.error("일부 팀이 Phase1을 완료하지 못했습니다.")
                return
            run_phase2_all_modules(args.period_id, teams)
        elif args.phase == '3':
            # Phase2 완료 체크
            if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE2_COMPLETED"):
                logging.error("일부 팀이 Phase2를 완료하지 못했습니다.")
                return
            run_phase3_reports_and_tone(args.period_id, teams)
        
        logging.info(f"[Phase{args.phase}] 완료!")
        sys.exit(0)

    # --module 옵션: 특정 모듈만 실행
    if args.module:
        teams = get_target_teams(args.period_id, team_list)
        logging.info(f"[Module{args.module}] {len(teams)}개 팀 실행")
        
        if args.module == 2:
            # 모듈2: 목표달성도 분석
            for team_id in teams:
                logging.info(f"[Module2] 팀 {team_id} 실행")
                task_ids, kpi_ids = fetch_team_tasks_and_kpis(team_id, args.period_id)
                state = {
                    "report_type": "quarterly",
                    "team_id": team_id,
                    "period_id": args.period_id,
                    "target_task_ids": task_ids,
                    "target_team_kpi_ids": kpi_ids,
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
        
        elif args.module == 3:
            # 모듈3: Peer Talk 분석
            for team_id in teams:
                logging.info(f"[Module3] 팀 {team_id} 실행")
                members = fetch_team_members(team_id)
                for member in members:
                    if member.get('role') == 'MANAGER':
                        continue
                    state = {
                        "team_id": team_id,
                        "period_id": args.period_id,
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
        
        elif args.module == 4:
            # 모듈4: 협업 분석
            for team_id in teams:
                logging.info(f"[Module4] 팀 {team_id} 실행")
                _, kpi_ids = fetch_team_tasks_and_kpis(team_id, args.period_id)
                state = {
                    "report_type": "quarterly",
                    "team_id": team_id,
                    "period_id": args.period_id,
                    "target_team_kpi_ids": kpi_ids,
                    "collaboration_relationships": None,
                    "individual_collaboration_analysis": None,
                    "team_collaboration_matrix": None,
                    "team_evaluation_id": None,
                    "messages": None
                }
                graph = create_module4_graph()
                graph.invoke(state)
        
        elif args.module == 6:
            # 모듈6: 4P BARS 평가
            for team_id in teams:
                logging.info(f"[Module6] 팀 {team_id} 실행")
                members = fetch_team_members(team_id)
                for member in members:
                    if member.get('role') == 'MANAGER':
                        continue
                    state = {
                        "report_type": "quarterly",
                        "team_id": team_id,
                        "period_id": args.period_id,
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
        
        elif args.module == 8:
            # 모듈8: 팀 성과 비교
            for team_id in teams:
                logging.info(f"[Module8] 팀 {team_id} 실행")
                module8_graph = create_module8_graph()
                state8 = {
                    "team_id": team_id,
                    "period_id": args.period_id,
                    "report_type": "quarterly",
                    "messages": []
                }
                module8_graph.invoke(state8)
        
        elif args.module == 10:
            # 모듈10: 개인 성장 코칭
            for team_id in teams:
                logging.info(f"[Module10] 팀 {team_id} 실행")
                members = fetch_team_members(team_id)
                for member in members:
                    if member.get('role') == 'MANAGER':
                        continue
                    emp_no = member["emp_no"]
                    logging.info(f"[Module10] {emp_no} 실행")
                    module10_graph = create_module10_graph()
                    state10 = {
                        "emp_no": emp_no,
                        "period_id": args.period_id,
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
        
        elif args.module == 11:
            # 모듈11: 팀 리스크 분석
            async def run_module11():
                db_wrapper = SQLAlchemyDBWrapper(engine)
                data_access = Module11DataAccess(db_wrapper)
                agent11 = Module11TeamRiskManagementAgent(data_access)
                tasks = []
                for team_id in teams:
                    logging.info(f"[Module11] 팀 {team_id} 실행")
                    try:
                        from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_evaluation_id
                        team_evaluation_id = fetch_team_evaluation_id(team_id, args.period_id)
                        if not team_evaluation_id:
                            logging.error(f"[Module11] 팀 {team_id} team_evaluation_id 없음")
                            continue
                        tasks.append(agent11.execute(team_id, args.period_id, team_evaluation_id))
                    except Exception as e:
                        logging.error(f"[Module11] 팀 {team_id} 실패: {e}")
                await asyncio.gather(*tasks)
            
            asyncio.run(run_module11())
        
        logging.info(f"[Module{args.module}] 완료!")
        sys.exit(0)

    # 기본 실행: 모든 단계 순차 실행
    teams = get_target_teams(args.period_id, team_list)
    logging.info(f"🚀 평가 시작: {len(teams)}개 팀")

    # Phase1: 팀별 평가 (모듈2,3,4,6)
    run_phase1_all_teams(teams, args.period_id)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE1_COMPLETED"):
        logging.warning("일부 팀이 Phase1을 완료하지 못했습니다. 중단합니다.")
        return
    
    # Phase2: 전사 모듈 (모듈8,10,11)
    run_phase2_all_modules(args.period_id, teams)
    if not check_all_teams_phase_completed(teams, args.period_id, "AI_PHASE2_COMPLETED"):
        logging.warning("일부 팀이 Phase2를 완료하지 못했습니다. 중단합니다.")
        return
    
    # Phase3: 리포트 생성 + 톤 조정
    run_phase3_reports_and_tone(args.period_id, teams)
    
    logging.info("분기별 평가 워크플로우 전체 완료!")

if __name__ == "__main__":
    main() 