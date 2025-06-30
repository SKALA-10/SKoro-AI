import logging
from typing import List, Optional
from agents.evaluation.modules.module_02_goal_achievement import db_utils
from sqlalchemy import bindparam

def get_target_teams(period_id: int, specific_teams: Optional[List[int]] = None) -> List[int]:
    """
    period_id 기준으로 평가 데이터가 있는 팀 목록을 조회한다.
    specific_teams가 주어지면 해당 팀만 필터링한다.
    """
    with db_utils.engine.connect() as connection:
        if specific_teams:
            query = db_utils.text(
                """
                SELECT DISTINCT team_id FROM team_evaluations
                WHERE period_id = :period_id AND team_id IN :team_ids
                """
            ).bindparams(bindparam('team_ids', expanding=True))
            result = connection.execute(query, {"period_id": period_id, "team_ids": specific_teams})
        else:
            query = db_utils.text(
                """
                SELECT DISTINCT team_id FROM team_evaluations
                WHERE period_id = :period_id
                """
            )
            result = connection.execute(query, {"period_id": period_id})
        teams = [row[0] for row in result]
    logging.info(f"조회된 팀 목록: {teams}")
    return teams

def run_team_module_with_retry(team_id: int, module_func, *args, **kwargs):
    """
    1회 재시도 후 실패시 스킵. module_func(team_id, ...) 형태로 호출.
    """
    try:
        return module_func(team_id, *args, **kwargs)
    except Exception as e:
        logging.warning(f"팀 {team_id} 첫 시도 실패: {e}")
        try:
            return module_func(team_id, *args, **kwargs)
        except Exception as e2:
            logging.error(f"팀 {team_id} 최종 실패, 스킵: {e2}")
            return False

def check_all_teams_phase_completed(teams: List[int], period_id: int, phase_status: str) -> bool:
    """
    모든 팀이 해당 phase_status에 도달했는지 체크한다.
    """
    with db_utils.engine.connect() as connection:
        query = db_utils.text(
            """
            SELECT COUNT(*) FROM team_evaluations
            WHERE period_id = :period_id AND team_id IN :team_ids AND status != :phase_status
            """
        ).bindparams(bindparam('team_ids', expanding=True))
        result = connection.execute(query, {"period_id": period_id, "team_ids": teams, "phase_status": phase_status})
        count = result.scalar_one()
    logging.info(f"동기화 체크: {count}개 팀이 아직 {phase_status} 미달성")
    return count == 0

def update_team_status(team_id: int, period_id: int, status: str):
    """
    team_evaluations의 status를 업데이트한다.
    """
    team_evaluation_id = db_utils.fetch_team_evaluation_id(team_id, period_id)
    if team_evaluation_id:
        db_utils.update_team_evaluations(team_evaluation_id, {"status": status})
        logging.info(f"팀 {team_id}의 status를 {status}로 업데이트")
    else:
        logging.warning(f"팀 {team_id}, period {period_id}에 해당하는 team_evaluation_id 없음")

def parse_teams(teams_arg: str) -> Optional[List[int]]:
    """
    '1,2,3' 또는 'all' 파싱
    """
    if teams_arg == "all":
        return None
    return [int(t) for t in teams_arg.split(",") if t.strip().isdigit()] 