"""
팀장용 레포트 데이터베이스 설정 및 연결 관리
Team manager report database configuration and connection management
"""

import sys
import os
from typing import Dict, List
from sqlalchemy import create_engine, text, Engine
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# DB 설정
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../..')))
from config.settings import DatabaseConfig

# 데이터베이스 설정
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row) -> Dict:
    """SQLAlchemy Row를 Dictionary로 변환"""
    return dict(row._mapping)

class ManagerDatabaseManager:
    """팀장용 레포트 데이터베이스 관리 클래스"""
    
    def __init__(self, engine: Engine):
        self.engine = engine
    
    def fetch_team_evaluation_reports(self) -> List[Dict]:
        """team_evaluations 테이블에서 팀장용 레포트 조회"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT team_evaluation_id as id, report
                FROM team_evaluations
                WHERE report IS NOT NULL
                ORDER BY team_evaluation_id
            """)
            results = connection.execute(query).fetchall()
            return [row_to_dict(row) for row in results]
    
    def update_team_evaluation_report(self, report_id: int, corrected_report: str) -> bool:
        """team_evaluations 테이블 업데이트"""
        try:
            with self.engine.connect() as connection:
                query = text("""
                    UPDATE team_evaluations 
                    SET report = :corrected_report 
                    WHERE team_evaluation_id = :report_id
                """)
                result = connection.execute(query, {
                    "corrected_report": corrected_report,
                    "report_id": report_id
                })
                connection.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"team_evaluations 업데이트 오류 (ID: {report_id}): {e}")
            return False