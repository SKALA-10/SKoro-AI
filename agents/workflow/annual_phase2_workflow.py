# =====================================
# 연말 2단계 평가 워크플로우
# =====================================
# 목적: 전사 모듈 실행 (모듈8,9,10,11) + 연말 리포트 생성
# - 모듈8: 팀 성과 비교
# - 모듈9: 부문별 CL 정규화
# - 모듈10: 개인 성장 코칭
# - 모듈11: 팀 운영 리스크 분석
# - Phase5: 연말 개인/팀 리포트 생성
# 전제 조건: 모든 팀이 SUBMITTED 상태여야 함
# =====================================
# 사용 예시 (터미널 실행 명령어)
# =====================================
# 전체 팀 자동 실행 (연말 2단계):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --auto
# 특정 팀만 자동 실행 (예: 팀 1):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --auto --teams 1
# 여러 팀 지정 (예: 팀 1, 3, 5):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --auto --teams 1,3,5
# Phase5만 실행 (연말 리포트 생성):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase5-only
# =====================================

import argparse
import logging
from agents.workflow.workflow_utils import (
    get_target_teams, run_team_module_with_retry, update_team_status, parse_teams
)
from agents.evaluation.modules.module_08_team_comparision.agent import create_module8_graph
from agents.evaluation.modules.module_10_growth_coaching.agent import create_module10_graph
from agents.evaluation.modules.module_11_team_coaching.agent import Module11TeamRiskManagementAgent
from agents.evaluation.modules.module_11_team_coaching.db_utils import Module11DataAccess, SQLAlchemyDBWrapper, engine
from agents.evaluation.modules.module_09_cl_normalization.run_module_09 import run_headquarter_module9_evaluation
from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_members, fetch_team_evaluation_id
from agents.evaluation.modules.module_09_cl_normalization.db_utils import get_all_headquarters_info
import asyncio
import sys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

def check_all_teams_submitted(teams, period_id):
    """
    모든 팀이 SUBMITTED 상태인지 체크한다.
    """
    from agents.evaluation.modules.module_02_goal_achievement import db_utils
    from sqlalchemy import bindparam
    with db_utils.engine.connect() as connection:
        query = db_utils.text(
            """
            SELECT COUNT(*) FROM team_evaluations
            WHERE period_id = :period_id AND team_id IN :team_ids AND status != 'SUBMITTED'
            """
        ).bindparams(bindparam('team_ids', expanding=True))
        result = connection.execute(query, {"period_id": period_id, "team_ids": teams})
        count = result.scalar_one()
    logging.info(f"팀장 제출 상태 체크: {count}개 팀이 아직 SUBMITTED 미달성")
    return count == 0

# Phase 4: 전사 모듈8,9,10,11 (순차)
def run_phase4_all_modules(period_id: int, teams):
    """
    Phase4: 전사 모듈8(팀 성과 비교), 9(부문 정규화), 10(개인 성장 코칭), 11(팀 리스크) 실행
    """
    logging.info("Phase4: 전사 모듈8,9,10,11 실행 시작")
    logging.info(f"[Phase4] 전체 대상 팀: {teams}")

    # 1. 모듈8: 팀 성과 비교 (팀 단위)
    for team_id in teams:
        def module8_func(team_id, period_id):
            module8_graph = create_module8_graph()
            state8 = {
                "team_id": team_id,
                "period_id": period_id,
                "report_type": "annual",
                "messages": []
            }
            module8_graph.invoke(state8)
            return True
        run_team_module_with_retry(team_id, module8_func, period_id)

    # 2. 모듈9: 부문 정규화 (본부별)
    try:
        logging.info(f"[Phase4][모듈9] 본부별 CL 정규화 실행 시작")
        headquarters = get_all_headquarters_info()
        logging.info(f"[Phase4][모듈9] 대상 본부: {len(headquarters)}개")
        for hq in headquarters:
            headquarter_id = hq["headquarter_id"]
            headquarter_name = hq["headquarter_name"]
            try:
                logging.info(f"[Phase4][모듈9] 본부 {headquarter_id} ({headquarter_name}) 실행 시작")
                result = run_headquarter_module9_evaluation(headquarter_id, period_id)
                if result:
                    logging.info(f"[Phase4][모듈9] 본부 {headquarter_id} 완료: {result.get('processed_count', 0)}명 처리")
                else:
                    logging.error(f"[Phase4][모듈9] 본부 {headquarter_id} 실패")
            except Exception as e:
                logging.error(f"[Phase4][모듈9] 본부 {headquarter_id} 실패: {e}")
        logging.info(f"[Phase4][모듈9] 본부별 CL 정규화 완료")
    except Exception as e:
        logging.error(f"[Phase4][모듈9] 본부별 CL 정규화 실패: {e}")

    # 3. 모듈10: 개인 성장 코칭 (팀원별)
    for team_id in teams:
        def module10_func(team_id, period_id):
            members = fetch_team_members(team_id)
            for member in members:
                # 팀장 제외
                if member.get('role') == 'MANAGER':
                    continue
                emp_no = member["emp_no"]
                module10_graph = create_module10_graph()
                state10 = {
                    "emp_no": emp_no,
                    "period_id": period_id,
                    "report_type": "annual",
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
            return True
        run_team_module_with_retry(team_id, module10_func, period_id)

    # 4. 모듈11: 팀 리스크 (팀 단위, async)
    async def run_module11_for_all_teams():
        db_wrapper = SQLAlchemyDBWrapper(engine)
        data_access = Module11DataAccess(db_wrapper)
        agent11 = Module11TeamRiskManagementAgent(data_access)
        tasks = []
        for team_id in teams:
            try:
                team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
                if not team_evaluation_id:
                    logging.error(f"[Phase4][모듈11] 팀 {team_id} team_evaluation_id 없음")
                    continue
                tasks.append(agent11.execute(team_id, period_id, team_evaluation_id))
            except Exception as e:
                logging.error(f"[Phase4][모듈11] 팀 {team_id} 실패: {e}")
        await asyncio.gather(*tasks)
        logging.info("[Phase4][모듈11] 전체 완료")

    asyncio.run(run_module11_for_all_teams())
    logging.info("Phase4: 완료")

def run_phase5_reports(period_id: int, teams):
    """
    Phase5: 연말 개인별 리포트 → 팀별 리포트 순서로 생성
    """
    logging.info("Phase5: 연말 리포트 생성 시작")
    
    # 1. 개인별 리포트 생성
    logging.info("[Phase5] 연말 개인별 리포트 생성 시작")
    from agents.report.annual_individual_reports import main as generate_annual_individual_reports
    generate_annual_individual_reports(period_id=period_id, teams=teams)
    
    # 2. 팀별 리포트 생성 (나중에 추가 예정)
    # logging.info("[Phase5] 연말 팀별 리포트 생성 시작")
    # from agents.report.annual_team_reports import main as generate_annual_team_reports
    # generate_annual_team_reports(period_id=period_id, teams=teams)
    
    logging.info("Phase5: 연말 리포트 생성 완료!")

def run_auto_workflow(period_id: int, specific_teams=None):
    """
    --auto 옵션: Phase4(전사 모듈) → Phase5(연말 리포트 생성) 자동 실행
    """
    logging.info("[AUTO] 연말 2단계 평가 자동 실행 시작")
    teams = get_target_teams(period_id, specific_teams)
    logging.info(f"[AUTO] 평가 대상 팀: {teams}")

    # 팀장 제출 상태 확인
    all_submitted = check_all_teams_submitted(teams, period_id)
    if not all_submitted:
        logging.error(f"[AUTO] 일부 팀이 아직 SUBMITTED 상태가 아닙니다. 팀장 수정 및 제출을 완료해주세요.")
        return
    else:
        logging.info("[AUTO] 모든 팀이 SUBMITTED 상태입니다. Phase4를 시작합니다.")

    # Phase4: 전사 모듈 자동 실행
    run_phase4_all_modules(period_id, teams)
    
    # Phase5: 연말 리포트 생성
    run_phase5_reports(period_id, teams)
    
    # 완료 후 상태 업데이트
    for team_id in teams:
        update_team_status(team_id, period_id, "COMPLETED")
    
    logging.info("[AUTO] 연말 2단계 평가 자동 실행 완료!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period-id', type=int, required=True)
    parser.add_argument('--teams', help='1,2,3 or all', required=False, default=None)
    parser.add_argument('--auto', action='store_true', help='모든 단계 자동 실행')
    parser.add_argument('--phase5-only', action='store_true', help='Phase5(연말 리포트 생성)만 실행')
    args = parser.parse_args()

    # --phase5-only 옵션: Phase5만 실행
    if args.phase5_only:
        if args.teams and args.teams != 'all':
            team_list = parse_teams(args.teams)
            logging.info(f"[Phase5-Only] 지정된 팀만 Phase5 실행: {team_list}")
        else:
            team_list = None
            logging.info("[Phase5-Only] 전체 팀 Phase5 실행")
        run_phase5_reports(args.period_id, team_list)
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
    logging.info(f"🚀 연말 2단계 평가 시작: {len(teams)}개 팀")

    # 팀장 제출 상태 확인
    if not check_all_teams_submitted(teams, args.period_id):
        logging.error("일부 팀이 아직 SUBMITTED 상태가 아닙니다. 팀장 수정 및 제출을 완료해주세요.")
        return

    run_phase4_all_modules(args.period_id, teams)
    
    # Phase5: 연말 리포트 생성
    run_phase5_reports(args.period_id, teams)
    
    for team_id in teams:
        update_team_status(team_id, args.period_id, "COMPLETED")
    logging.info("연말 2단계 평가 워크플로우 완료!")

if __name__ == "__main__":
    main() 