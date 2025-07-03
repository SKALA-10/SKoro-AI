"""
데이터베이스 설정 및 연결 관리
Database configuration and connection management
"""

import sys
import os
from typing import Dict, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
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

class DatabaseManager:
    """SQLAlchemy Core를 사용한 데이터베이스 관리 클래스"""
    
    def __init__(self, engine: Engine):
        self.engine = engine
    
    def fetch_employee_quarterly_reports(self) -> List[Dict]:
        """feedback_reports 테이블에서 팀원용 분기별 레포트 조회"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT feedback_report_id as id, report
                FROM feedback_reports
                WHERE report IS NOT NULL
                ORDER BY feedback_report_id
            """)
            results = connection.execute(query).fetchall()
            return [row_to_dict(row) for row in results]
    
    def fetch_employee_annual_reports(self) -> List[Dict]:
        """final_evaluation_reports 테이블에서 팀원용 연말 레포트 조회"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT final_evaluation_report_id as id, report
                FROM final_evaluation_reports
                WHERE report IS NOT NULL
                ORDER BY final_evaluation_report_id
            """)
            results = connection.execute(query).fetchall()
            return [row_to_dict(row) for row in results]
    
    def update_feedback_report(self, report_id: int, corrected_report: str) -> bool:
        """feedback_reports 테이블 업데이트"""
        try:
            with self.engine.connect() as connection:
                query = text("""
                    UPDATE feedback_reports 
                    SET report = :corrected_report 
                    WHERE feedback_report_id = :report_id
                """)
                result = connection.execute(query, {
                    "corrected_report": corrected_report,
                    "report_id": report_id
                })
                connection.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"feedback_reports 업데이트 오류 (ID: {report_id}): {e}")
            return False
    
    def update_final_evaluation_report(self, report_id: int, corrected_report: str) -> bool:
        """final_evaluation_reports 테이블 업데이트"""
        try:
            with self.engine.connect() as connection:
                query = text("""
                    UPDATE final_evaluation_reports 
                    SET report = :corrected_report 
                    WHERE final_evaluation_report_id = :report_id
                """)
                result = connection.execute(query, {
                    "corrected_report": corrected_report,
                    "report_id": report_id
                })
                connection.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"final_evaluation_reports 업데이트 오류 (ID: {report_id}): {e}")
            return False