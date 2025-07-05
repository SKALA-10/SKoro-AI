import json
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from langchain_openai import ChatOpenAI

from config.settings import DatabaseConfig
from agents.tone_adjustment.individual_tone_adjustment import IndividualToneAdjustmentAgent

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def get_db_engine() -> Engine:
    """데이터베이스 엔진 생성"""
    db_config = DatabaseConfig()
    engine = create_engine(db_config.DATABASE_URL, pool_pre_ping=True)
    return engine

def fetch_team_emp_nos(engine: Engine, teams: list, period_id: int) -> list:
    """특정 팀들의 직원 번호를 조회합니다. (팀장 제외) - 연말 리포트용"""
    placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
    query = text(f"""
        SELECT DISTINCT e.emp_no 
        FROM employees e
        JOIN final_evaluation_reports fer ON e.emp_no = fer.emp_no
        JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
        WHERE e.team_id IN ({placeholders}) 
        AND e.role != 'MANAGER'  # 팀장 제외
        AND te.period_id = :period_id
        AND fer.report IS NOT NULL  # 리포트가 있는 직원만
        ORDER BY e.emp_no;
    """)
    params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
    params['period_id'] = period_id
    
    with engine.connect() as connection:
        results = connection.execute(query, params).fetchall()
    emp_nos = [row[0] for row in results]
    logging.info(f"✅ 팀 {teams}의 직원 {len(emp_nos)}명을 처리합니다. 대상 직원: {emp_nos}")
    return emp_nos

def load_report_from_db(engine: Engine, emp_no: str, period_id: int) -> Optional[Dict[str, Any]]:
    """DB에서 직원의 리포트 JSON을 로드합니다. - 연말 리포트용"""
    try:
        query = text("""
            SELECT fer.report
            FROM final_evaluation_reports fer
            JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
            WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            AND fer.report IS NOT NULL
        """)
        
        with engine.connect() as connection:
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).first()
            
        if result and result[0]:
            return json.loads(result[0])
        else:
            logging.warning(f"⚠️ {emp_no}님의 리포트를 DB에서 찾을 수 없습니다.")
            return None
            
    except Exception as e:
        logging.error(f"❌ {emp_no}님 리포트 로드 중 오류: {e}")
        return None

def save_adjusted_report_to_db(engine: Engine, emp_no: str, adjusted_report: Dict[str, Any], period_id: int) -> bool:
    """조정된 리포트를 DB에 저장합니다. - 연말 리포트용"""
    try:
        json_content = json.dumps(adjusted_report, ensure_ascii=False, indent=2)
        
        query = text("""
            UPDATE final_evaluation_reports 
            SET report = :report_content 
            WHERE emp_no = :emp_no 
            AND team_evaluation_id IN (
                SELECT te.team_evaluation_id 
                FROM team_evaluations te 
                WHERE te.period_id = :period_id
            )
        """)
        
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query, {
                    "report_content": json_content, 
                    "emp_no": emp_no, 
                    "period_id": period_id
                })
                if result.rowcount > 0:
                    transaction.commit()
                    logging.info(f"✅ {emp_no}님의 조정된 리포트가 DB에 저장되었습니다.")
                    return True
                else:
                    transaction.rollback()
                    logging.warning(f"⚠️ {emp_no}님에 해당하는 레코드를 찾을 수 없습니다.")
                    return False
                    
    except Exception as e:
        logging.error(f"❌ {emp_no}님 리포트 저장 중 오류: {e}")
        return False

def run_tone_adjustment_for_emp(emp_no: str, period_id: int, teams: list, llm_client: ChatOpenAI) -> Dict[str, Any]:
    """특정 직원의 톤 조정을 실행합니다."""
    logging.info(f"🎨 {emp_no}님 톤 조정 시작")
    
    engine = get_db_engine()
    
    # 1. DB에서 리포트 로드
    original_report = load_report_from_db(engine, emp_no, period_id)
    if not original_report:
        return {"emp_no": emp_no, "status": "error", "message": "리포트 로드 실패"}
    
    # 2. 톤 조정 실행
    try:
        agent = IndividualToneAdjustmentAgent(llm_client)
        
        # final_evaluation_reports 타입으로 톤 조정
        adjusted_report = agent.process_report(original_report, "final_evaluation_reports")
        
        # 3. DB에 저장
        save_success = save_adjusted_report_to_db(engine, emp_no, adjusted_report, period_id)
        
        if save_success:
            return {
                "emp_no": emp_no, 
                "status": "success", 
                "original_length": len(json.dumps(original_report, ensure_ascii=False)),
                "adjusted_length": len(json.dumps(adjusted_report, ensure_ascii=False))
            }
        else:
            return {"emp_no": emp_no, "status": "error", "message": "DB 저장 실패"}
            
    except Exception as e:
        logging.error(f"❌ {emp_no}님 톤 조정 중 오류: {e}")
        return {"emp_no": emp_no, "status": "error", "message": str(e)}

def run_tone_adjustment_for_teams(period_id: int, teams: list, llm_client: ChatOpenAI) -> Dict[str, Any]:
    """팀별 직원들의 톤 조정을 실행합니다."""
    logging.info(f"🎨 Phase 4.2: 톤 조정 시작 - 팀 {teams}, 분기 {period_id}")
    
    engine = get_db_engine()
    
    # 대상 직원 목록 조회
    emp_nos = fetch_team_emp_nos(engine, teams, period_id)
    if not emp_nos:
        logging.warning("처리할 직원이 없습니다.")
        return {"status": "no_employees", "results": []}
    
    # 직원별 순차 처리
    results = []
    success_count = 0
    error_count = 0
    
    for idx, emp_no in enumerate(emp_nos, 1):
        logging.info(f"[{idx}/{len(emp_nos)}] {emp_no}님 처리 중...")
        
        result = run_tone_adjustment_for_emp(emp_no, period_id, teams, llm_client)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
        else:
            error_count += 1
    
    logging.info(f"🎉 Phase 4.2: 톤 조정 완료!")
    logging.info(f"✅ 성공: {success_count}개")
    logging.info(f"❌ 실패: {error_count}개")
    logging.info(f"📊 총 처리: {len(emp_nos)}개")
    
    return {
        "status": "completed",
        "total": len(emp_nos),
        "success": success_count,
        "error": error_count,
        "results": results
    }

def main(period_id: int, teams: list):
    """메인 실행 함수"""
    try:
        # LLM 클라이언트 초기화
        llm_client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        # 톤 조정 실행
        result = run_tone_adjustment_for_teams(period_id, teams, llm_client)
        
        return result
        
    except Exception as e:
        logging.error(f"❌ 톤 조정 실행 중 오류: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="개인별 톤 조정 실행")
    parser.add_argument('--period-id', type=int, required=True, help='분기 ID')
    parser.add_argument('--teams', required=True, help='팀 ID (예: 1,2,3)')
    
    args = parser.parse_args()
    
    # 팀 목록 파싱
    team_list = [int(t.strip()) for t in args.teams.split(',')]
    
    main(args.period_id, team_list)