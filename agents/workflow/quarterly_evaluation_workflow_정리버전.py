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
# Phase 4: 리포트 생성
# - 개인별 리포트 생성
# - 팀별 리포트 생성
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
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 4 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 4.1 --teams 1
#   python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 4.2 --teams 1
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

# =====================================
# 모듈 실행 함수들
# =====================================

def run_module2_for_team(team_id: int, period_id: int):
    """모듈2: 목표달성도 분석"""
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

def run_module3_for_team(team_id: int, period_id: int):
    """모듈3: Peer Talk 분석"""
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

def run_module4_for_team(team_id: int, period_id: int):
    """모듈4: 협업 분석"""
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

def run_module6_for_team(team_id: int, period_id: int):
    """모듈6: 4P BARS 평가"""
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

def run_module8_for_team(team_id: int, period_id: int):
    """모듈8: 팀 성과 비교"""
    module8_graph = create_module8_graph()
    state8 = {
        "team_id": team_id,
        "period_id": period_id,
        "report_type": "quarterly",
        "messages": []
    }
    module8_graph.invoke(state8)

def run_module10_for_team(team_id: int, period_id: int):
    """모듈10: 개인 성장 코칭"""
    members = fetch_team_members(team_id)
    for member in members:
        if member.get('role') == 'MANAGER':
            continue
        emp_no = member["emp_no"]
        logging.info(f"[Module10] 팀 {team_id} - {emp_no} 실행")
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

async def run_module11_for_team(team_id: int, period_id: int):
    """모듈11: 팀 리스크 분석"""
    from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_evaluation_id
    team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
    if not team_evaluation_id:
        logging.error(f"[Module11] 팀 {team_id} team_evaluation_id 없음")
        return
    
    db_wrapper = SQLAlchemyDBWrapper(engine)
    data_access = Module11DataAccess(db_wrapper)
    agent11 = Module11TeamRiskManagementAgent(data_access)
    await agent11.execute(team_id, period_id, team_evaluation_id)

# =====================================
# Phase 실행 함수들
# =====================================

def run_phase1_all_teams(teams, period_id):
    """Phase1: 모듈2,3,6 (팀별)"""
    logging.info("Phase1: 모듈2,3,6 실행 시작")
    for idx, team_id in enumerate(teams, 1):
        logging.info(f"[Phase1] 팀 {team_id} ({idx}/{len(teams)}) 시작")
        
        # 모듈2 실행
        run_team_module_with_retry(team_id, run_module2_for_team, period_id)
        
        # 모듈3 실행
        run_team_module_with_retry(team_id, run_module3_for_team, period_id)
        
        # 모듈6 실행
        run_team_module_with_retry(team_id, run_module6_for_team, period_id)
        
        update_team_status(team_id, period_id, "AI_PHASE1_COMPLETED")
        logging.info(f"[Phase1] 팀 {team_id} 완료")

def run_phase2_all_teams(teams, period_id):
    """Phase2: 모듈4 (팀별)"""
    logging.info("Phase2: 모듈4 실행 시작")
    for idx, team_id in enumerate(teams, 1):
        logging.info(f"[Phase2] 팀 {team_id} ({idx}/{len(teams)}) 시작")
        run_team_module_with_retry(team_id, run_module4_for_team, period_id)
        update_team_status(team_id, period_id, "AI_PHASE2_COMPLETED")
        logging.info(f"[Phase2] 팀 {team_id} 완료")

def run_phase3_all_modules(period_id: int, teams):
    """Phase3: 전사 모듈8,10,11"""
    logging.info("Phase3: 전사 모듈8,10,11 실행 시작")
    logging.info(f"[Phase3] 전체 대상 팀: {teams}")

    # 모듈8: 팀 성과 비교 (팀 단위)
    for team_id in teams:
        try:
            logging.info(f"[Phase3][모듈8] 팀 {team_id} 실행 시작")
            run_module8_for_team(team_id, period_id)
            logging.info(f"[Phase3][모듈8] 팀 {team_id} 완료")
        except Exception as e:
            logging.error(f"[Phase3][모듈8] 팀 {team_id} 실패: {e}")

    # 모듈10: 개인 성장 코칭 (팀원별)
    for team_id in teams:
        try:
            run_module10_for_team(team_id, period_id)
        except Exception as e:
            logging.error(f"[Phase3][모듈10] 팀 {team_id} 실패: {e}")

    # 모듈11: 팀 리스크 (팀 단위, async)
    async def run_module11_for_all_teams():
        tasks = []
        for team_id in teams:
            try:
                tasks.append(run_module11_for_team(team_id, period_id))
            except Exception as e:
                logging.error(f"[Phase3][모듈11] 팀 {team_id} 실패: {e}")
        await asyncio.gather(*tasks)
        logging.info("[Phase3][모듈11] 전체 완료")

    asyncio.run(run_module11_for_all_teams())
    logging.info("Phase3: 완료")

def run_phase4_1_generate_reports(period_id: int, teams):
    """Phase4.1: 개인별 리포트 생성 및 DB 저장"""
    logging.info("Phase4.1: 개인별 리포트 생성 시작")
    
    # 개인별 리포트 생성 (JSON 반환)
    from agents.report.quarterly_individual_reports import main as generate_individual_reports
    generated_reports = generate_individual_reports(period_id=period_id, teams=teams, return_json=True)
    
    # 팀별 리포트 생성
    logging.info("[Phase4.1] 팀별 리포트 생성 시작")
    from agents.report.quarterly_team_reports import main as generate_team_reports
    generate_team_reports(period_id=period_id, teams=teams)
    
    logging.info("Phase4.1: 리포트 생성 완료!")
    return generated_reports

def run_phase4_2_tone_adjustment(period_id: int, teams):
    """Phase4.2: 개인별 톤 조정 + 팀별 톤 조정 및 DB 업데이트"""
    logging.info("Phase4.2: 개인별 + 팀별 톤 조정 시작")
    
    # 1. 개인별 톤 조정
    try:
        logging.info("[Phase4.2] 개인별 톤 조정 시작")
        from agents.tone_adjustment.run_individual_tone_adjustment import main as run_individual_tone_adjustment
        individual_result = run_individual_tone_adjustment(period_id=period_id, teams=teams)
        logging.info("[Phase4.2] 개인별 톤 조정 완료")
    except Exception as e:
        logging.error(f"[Phase4.2] 개인별 톤 조정 실패: {e}")
        individual_result = None
    
    # 2. 팀별 톤 조정
    try:
        logging.info("[Phase4.2] 팀별 톤 조정 시작")
        import agents.tone_adjustment.run_team_tone_adjustment as team_tone_module
        team_result = team_tone_module.main(period_id=period_id, teams=teams)
        logging.info("[Phase4.2] 팀별 톤 조정 완료")
    except Exception as e:
        logging.error(f"[Phase4.2] 팀별 톤 조정 실패: {e}")
        team_result = None
    
    logging.info("Phase4.2: 전체 톤 조정 완료!")
    return {
        "individual_result": individual_result,
        "team_result": team_result
    }

def run_phase4_reports(period_id: int, teams):
    """Phase4: 전체 리포트 프로세스"""
    logging.info("Phase4: 전체 리포트 프로세스 시작")
    
    # 4.1: 개인별 리포트 생성
    generated_reports = run_phase4_1_generate_reports(period_id, teams)
    
    # 4.2: 개인별 톤 조정
    tone_result = run_phase4_2_tone_adjustment(period_id, teams)
    
    logging.info("Phase4: 전체 리포트 프로세스 완료!")

def run_auto_workflow(period_id: int, specific_teams=None):
    """--auto 옵션: 전체 자동 실행"""
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

def run_single_module(module_id: int, teams: list, period_id: int):
    """단일 모듈 실행"""
    logging.info(f"[Module{module_id}] {len(teams)}개 팀 실행")
    
    module_functions = {
        2: run_module2_for_team,
        3: run_module3_for_team,
        4: run_module4_for_team,
        6: run_module6_for_team,
        8: run_module8_for_team,
        10: run_module10_for_team
    }
    
    if module_id == 11:
        # 모듈11은 async 함수
        async def run_module11():
            tasks = []
            for team_id in teams:
                logging.info(f"[Module11] 팀 {team_id} 실행")
                try:
                    tasks.append(run_module11_for_team(team_id, period_id))
                except Exception as e:
                    logging.error(f"[Module11] 팀 {team_id} 실패: {e}")
            await asyncio.gather(*tasks)
        
        asyncio.run(run_module11())
    else:
        # 일반 모듈들
        for team_id in teams:
            logging.info(f"[Module{module_id}] 팀 {team_id} 실행")
            try:
                module_functions[module_id](team_id, period_id)
            except Exception as e:
                logging.error(f"[Module{module_id}] 팀 {team_id} 실패: {e}")
    
    logging.info(f"[Module{module_id}] 완료!")

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
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 4 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 4.1 --teams 1
  python agents/workflow/quarterly_evaluation_workflow.py --period-id 2 --phase 4.2 --teams 1
  
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
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', '4', '4.1', '4.2'], help='특정 Phase만 실행')
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
        
        phase_functions = {
            '1': lambda: run_phase1_all_teams(teams, args.period_id),
            '2': lambda: run_phase2_all_teams(teams, args.period_id),
            '3': lambda: run_phase3_all_modules(args.period_id, teams),
            '4': lambda: run_phase4_reports(args.period_id, teams),
            '4.1': lambda: run_phase4_1_generate_reports(args.period_id, teams),
            '4.2': lambda: run_phase4_2_tone_adjustment(args.period_id, teams)
        }
        
        phase_functions[args.phase]()
        logging.info(f"[Phase{args.phase}] 완료!")
        sys.exit(0)

    # --module 옵션: 특정 모듈만 실행
    if args.module:
        teams = get_target_teams(args.period_id, team_list)
        run_single_module(args.module, teams, args.period_id)
        sys.exit(0)

    # 기본 실행: 모든 단계 순차 실행
    teams = get_target_teams(args.period_id, team_list)
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
    run_phase4_reports(args.period_id, teams)
    logging.info("분기별 평가 워크플로우 전체 완료!")

if __name__ == "__main__":
    main() 