# =====================================
# 연말 2단계 평가 워크플로우
# =====================================
# 목적: 전사 모듈 실행 (모듈8,9,10,11) + 연말 리포트 생성 + 개인 톤 조정
# - 모듈8: 팀 성과 비교
# - 모듈9: 부문별 CL 정규화
# - 모듈10: 개인 성장 코칭
# - 모듈11: 팀 운영 리스크 분석
# - Phase5.1: 연말 개인/팀 리포트 생성
# - Phase5.2: 개인별 톤 조정
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
# Phase5 전체 실행 (리포트 생성 + 톤 조정):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase 5
# Phase5.1만 실행 (연말 리포트 생성):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase 5.1
# Phase5.2만 실행 (개인 톤 조정):
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase 5.2
# 특정 모듈만 실행:
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 8 --teams 1
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 9
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 10 --teams 1
#   python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 11 --teams 1
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
from agents.evaluation.modules.module_09_cl_normalization.db_utils import get_all_headquarters_info
from agents.evaluation.modules.module_09_cl_normalization.run_module_09 import run_enhanced_module9_workflow_fixed
from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_members, fetch_team_evaluation_id

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

# 개별 모듈 실행 함수들
def run_module8(period_id: int, teams):
    """모듈8: 팀 성과 비교 실행"""
    logging.info("[Module8] 팀 성과 비교 실행 시작")
    for team_id in teams:
        try:
            logging.info(f"[Module8] 팀 {team_id} 실행 시작")
            module8_graph = create_module8_graph()
            state8 = {
                "team_id": team_id,
                "period_id": period_id,
                "report_type": "annual",
                "messages": []
            }
            module8_graph.invoke(state8)
            logging.info(f"[Module8] 팀 {team_id} 완료")
        except Exception as e:
            logging.error(f"[Module8] 팀 {team_id} 실패: {e}")
    logging.info("[Module8] 팀 성과 비교 완료")

def run_module9(period_id: int):
    """모듈9: 부문별 CL 정규화 실행 (향상된 버전)"""
    logging.info("[Module9] 부문별 CL 정규화 실행 시작 (향상된 버전)")
    try:
        headquarters = get_all_headquarters_info()
        logging.info(f"[Module9] 대상 본부: {len(headquarters)}개")
        for hq in headquarters:
            headquarter_id = hq["headquarter_id"]
            headquarter_name = hq["headquarter_name"]
            try:
                logging.info(f"[Module9] 본부 {headquarter_id} ({headquarter_name}) 실행 시작")
                # headquarter_id를 정수로 변환 (새로운 버전은 int를 요구함)
                headquarter_id_int = int(headquarter_id) if isinstance(headquarter_id, str) else headquarter_id
                result = run_enhanced_module9_workflow_fixed(headquarter_id_int, period_id)
                if result and result.get("success"):
                    logging.info(f"[Module9] 본부 {headquarter_id} 완료: {result.get('total_processed', 0)}명 처리")
                else:
                    logging.error(f"[Module9] 본부 {headquarter_id} 실패: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logging.error(f"[Module9] 본부 {headquarter_id} 실패: {e}")
        logging.info("[Module9] 부문별 CL 정규화 완료")
    except Exception as e:
        logging.error(f"[Module9] 부문별 CL 정규화 실패: {e}")

def run_module10(period_id: int, teams):
    """모듈10: 개인 성장 코칭 실행"""
    logging.info("[Module10] 개인 성장 코칭 실행 시작")
    for team_id in teams:
        try:
            members = fetch_team_members(team_id)
            for member in members:
                # 팀장 제외
                if member.get('role') == 'MANAGER':
                    continue
                emp_no = member["emp_no"]
                logging.info(f"[Module10] 팀 {team_id} - {emp_no} 실행 시작")
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
                logging.info(f"[Module10] 팀 {team_id} - {emp_no} 완료")
        except Exception as e:
            logging.error(f"[Module10] 팀 {team_id} 실패: {e}")
    logging.info("[Module10] 개인 성장 코칭 완료")

async def run_module11(period_id: int, teams):
    """모듈11: 팀 리스크 분석 실행"""
    logging.info("[Module11] 팀 리스크 분석 실행 시작")
    db_wrapper = SQLAlchemyDBWrapper(engine)
    data_access = Module11DataAccess(db_wrapper)
    agent11 = Module11TeamRiskManagementAgent(data_access)
    tasks = []
    for team_id in teams:
        try:
            team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
            if not team_evaluation_id:
                logging.error(f"[Module11] 팀 {team_id} team_evaluation_id 없음")
                continue
            logging.info(f"[Module11] 팀 {team_id} 실행 시작")
            tasks.append(agent11.execute(team_id, period_id, team_evaluation_id))
        except Exception as e:
            logging.error(f"[Module11] 팀 {team_id} 실패: {e}")
    await asyncio.gather(*tasks)
    logging.info("[Module11] 팀 리스크 분석 완료")

# Phase 4: 전사 모듈8,9,10,11 (순차)
def run_phase4_all_modules(period_id: int, teams):
    """
    Phase4: 전사 모듈8(팀 성과 비교), 9(부문 정규화), 10(개인 성장 코칭), 11(팀 리스크) 실행
    """
    logging.info("Phase4: 전사 모듈8,9,10,11 실행 시작")
    logging.info(f"[Phase4] 전체 대상 팀: {teams}")

    # 1. 모듈8: 팀 성과 비교 (팀 단위)
    run_module8(period_id, teams)

    # 2. 모듈9: 부문 정규화 (본부별)
    run_module9(period_id)

    # 3. 모듈10: 개인 성장 코칭 (팀원별)
    run_module10(period_id, teams)

    # 4. 모듈11: 팀 리스크 (팀 단위, async)
    asyncio.run(run_module11(period_id, teams))
    logging.info("Phase4: 완료")

def run_phase5_1_generate_reports(period_id: int, teams):
    """
    Phase5.1: 연말 개인별 리포트 → 팀별 리포트 순서로 생성
    """
    logging.info("Phase5.1: 연말 리포트 생성 시작")
    
    # 1. 개인별 리포트 생성
    logging.info("[Phase5.1] 연말 개인별 리포트 생성 시작")
    from agents.report.annual_individual_reports import main as generate_annual_individual_reports
    generate_annual_individual_reports(period_id=period_id, teams=teams)
    
    # 2. 팀별 리포트 생성
    logging.info("[Phase5.1] 연말 팀별 리포트 생성 시작")
    from agents.report.annual_team_reports import main as generate_annual_team_reports
    generate_annual_team_reports(period_id=period_id, teams=teams)
    
    logging.info("Phase5.1: 연말 리포트 생성 완료!")

def run_phase5_2_tone_adjustment(period_id: int, teams):
    """
    Phase5.2: 개인별 톤 조정 및 DB 업데이트
    """
    logging.info("Phase5.2: 개인별 톤 조정 시작")
    
    try:
        logging.info("[Phase5.2] 개인별 톤 조정 시작")
        from agents.tone_adjustment.run_individual_tone_adjustment import main as run_individual_tone_adjustment
        result = run_individual_tone_adjustment(period_id=period_id, teams=teams)
        logging.info("[Phase5.2] 개인별 톤 조정 완료")
        return result
    except Exception as e:
        logging.error(f"[Phase5.2] 개인별 톤 조정 실패: {e}")
        return None

def run_phase5_reports(period_id: int, teams):
    """
    Phase5: 연말 리포트 생성 → 개인별 톤 조정 순서로 실행
    """
    logging.info("Phase5: 전체 리포트 프로세스 시작")
    
    # 5.1: 연말 리포트 생성
    run_phase5_1_generate_reports(period_id, teams)
    
    # 5.2: 개인별 톤 조정
    tone_result = run_phase5_2_tone_adjustment(period_id, teams)
    
    logging.info("Phase5: 전체 리포트 프로세스 완료!")

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
    parser = argparse.ArgumentParser(
        description="연말 2단계 평가 워크플로우",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 자동 실행
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --auto
  
  # 특정 팀만 자동 실행
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --auto --teams 1
  
  # Phase5 전체 실행 (리포트 생성 + 톤 조정)
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase 5
  
  # Phase5.1만 실행 (연말 리포트 생성)
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase 5.1
  
  # Phase5.2만 실행 (개인 톤 조정)
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --phase 5.2
  
  # 특정 모듈만 실행
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 8 --teams 1
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 9
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 10 --teams 1
  python agents/workflow/annual_phase2_workflow.py --period-id 4 --module 11 --teams 1
        """
    )
    parser.add_argument('--period-id', type=int, required=True, help='연말 기간 ID (예: 4)')
    parser.add_argument('--teams', help='팀 ID (예: 1,2,3 또는 all)', required=False, default=None)
    parser.add_argument('--auto', action='store_true', help='모든 단계 자동 실행')
    parser.add_argument('--phase', type=str, choices=['5', '5.1', '5.2'], help='특정 Phase만 실행 (5: 전체, 5.1: 리포트 생성, 5.2: 톤 조정)')
    parser.add_argument('--module', type=int, choices=[8, 9, 10, 11], help='특정 모듈만 실행')
    args = parser.parse_args()

    # 팀 목록 파싱
    if args.teams and args.teams != 'all':
        team_list = parse_teams(args.teams)
    else:
        team_list = None

    # --phase 옵션: 특정 Phase만 실행
    if args.phase:
        teams = get_target_teams(args.period_id, team_list)
        if team_list:
            logging.info(f"[Phase{args.phase}] 지정된 팀만 실행: {team_list}")
        else:
            logging.info(f"[Phase{args.phase}] 전체 팀 실행")
        
        if args.phase == '5':
            run_phase5_reports(args.period_id, teams)
        elif args.phase == '5.1':
            run_phase5_1_generate_reports(args.period_id, teams)
        elif args.phase == '5.2':
            run_phase5_2_tone_adjustment(args.period_id, teams)
        
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

    # --module 옵션: 특정 모듈만 실행
    if args.module:
        teams = get_target_teams(args.period_id, team_list)
        logging.info(f"[Module{args.module}] {len(teams)}개 팀 실행")
        
        if args.module == 8:
            # 모듈8: 팀 성과 비교
            run_module8(args.period_id, teams)
        
        elif args.module == 9:
            # 모듈9: 부문별 CL 정규화 (팀 지정 불필요)
            run_module9(args.period_id)
        
        elif args.module == 10:
            # 모듈10: 개인 성장 코칭
            run_module10(args.period_id, teams)
        
        elif args.module == 11:
            # 모듈11: 팀 리스크 분석
            asyncio.run(run_module11(args.period_id, teams))
        
        logging.info(f"[Module{args.module}] 완료!")
        sys.exit(0)

    # 기본 실행: Phase4 + Phase5 순차 실행
    teams = get_target_teams(args.period_id, team_list)
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