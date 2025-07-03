"""
팀장용 레포트 데이터베이스 설정 및 연결 관리 (개선된 버전)
Team manager report database configuration and connection management (improved version)
"""

import sys
import os
import json
from typing import Dict, List, Any
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
            
            processed_results = []
            for row in results:
                row_dict = row_to_dict(row)
                
                # report 필드가 문자열인 경우 JSON으로 파싱 시도
                report_data = row_dict['report']
                if isinstance(report_data, str):
                    try:
                        report_data = json.loads(report_data)
                    except json.JSONDecodeError:
                        # 파싱 실패 시 원본 문자열 그대로 유지
                        print(f"⚠️ 레포트 ID {row_dict['id']}: JSON 파싱 실패, 원본 유지")
                
                row_dict['report'] = report_data
                processed_results.append(row_dict)
            
            return processed_results
    
    def update_team_evaluation_report(self, report_id: int, corrected_report: str) -> bool:
        """team_evaluations 테이블 업데이트"""
        try:
            with self.engine.connect() as connection:
                # 문자열이 이미 JSON 형태인지 확인
                try:
                    # JSON 유효성 검증
                    json.loads(corrected_report)
                    report_to_save = corrected_report
                except json.JSONDecodeError:
                    # JSON이 아닌 경우 에러 처리
                    print(f"⚠️ 잘못된 JSON 형식의 레포트 (ID: {report_id})")
                    return False
                
                query = text("""
                    UPDATE team_evaluations 
                    SET report = :corrected_report 
                    WHERE team_evaluation_id = :report_id
                """)
                result = connection.execute(query, {
                    "corrected_report": report_to_save,
                    "report_id": report_id
                })
                connection.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"team_evaluations 업데이트 오류 (ID: {report_id}): {e}")
            return False
    
    def validate_json_format(self, report_data: Any) -> bool:
        """레포트 데이터가 유효한 JSON 형식인지 검증"""
        if isinstance(report_data, dict):
            return True
        elif isinstance(report_data, str):
            try:
                json.loads(report_data)
                return True
            except json.JSONDecodeError:
                return False
        return False
    
    def get_report_statistics(self) -> Dict[str, int]:
        """레포트 통계 조회"""
        try:
            with self.engine.connect() as connection:
                query = text("""
                    SELECT 
                        COUNT(*) as total_reports,
                        COUNT(CASE WHEN report IS NOT NULL THEN 1 END) as reports_with_data,
                        COUNT(CASE WHEN report IS NULL THEN 1 END) as reports_without_data
                    FROM team_evaluations
                """)
                result = connection.execute(query).fetchone()
                
                if result:
                    return {
                        'total_reports': result.total_reports,
                        'reports_with_data': result.reports_with_data,
                        'reports_without_data': result.reports_without_data
                    }
                else:
                    return {
                        'total_reports': 0,
                        'reports_with_data': 0,
                        'reports_without_data': 0
                    }
        except Exception as e:
            print(f"통계 조회 오류: {e}")
            return {
                'total_reports': 0,
                'reports_with_data': 0,
                'reports_without_data': 0
            }