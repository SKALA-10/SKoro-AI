"""
중간 레포트 데이터베이스 설정 및 연결 관리
Middle report database configuration and connection management
"""

import sys
import os
from typing import Dict, List
from sqlalchemy import create_engine, text, Engine
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# DB 설정 (오류 처리 추가)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../..')))

try:
    from config.settings import DatabaseConfig
    db_config = DatabaseConfig()
    DATABASE_URL = db_config.DATABASE_URL
    print("✅ DatabaseConfig 로드 성공")
except Exception as e:
    print(f"⚠️ DatabaseConfig 로드 실패: {e}")
    # 환경변수에서 직접 가져오기 시도
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL 환경변수도 없음. 기본값 사용")
        DATABASE_URL = "sqlite:///./fallback.db"

# 데이터베이스 엔진 생성
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    print("✅ Database engine 생성 성공")
except Exception as e:
    print(f"❌ Database engine 생성 실패: {e}")
    engine = None

def row_to_dict(row) -> Dict:
    """SQLAlchemy Row를 Dictionary로 변환"""
    return dict(row._mapping)

class MiddleDatabaseManager:
    """중간 레포트 데이터베이스 관리 클래스"""
    
    def __init__(self, engine: Engine):
        self.engine = engine
    
    def fetch_middle_reports(self) -> List[Dict]:
        """team_evaluations 테이블에서 middle_report 컬럼 조회"""
        with self.engine.connect() as connection:
            query = text("""
                SELECT team_evaluation_id as id, middle_report
                FROM team_evaluations
                WHERE middle_report IS NOT NULL
                ORDER BY team_evaluation_id
            """)
            results = connection.execute(query).fetchall()
            return [row_to_dict(row) for row in results]
    
    def update_middle_report(self, report_id: int, corrected_report: str) -> bool:
        """team_evaluations 테이블의 middle_report 컬럼 업데이트"""
        try:
            with self.engine.connect() as connection:
                query = text("""
                    UPDATE team_evaluations 
                    SET middle_report = :corrected_report 
                    WHERE team_evaluation_id = :report_id
                """)
                result = connection.execute(query, {
                    "corrected_report": corrected_report,
                    "report_id": report_id
                })
                connection.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"middle_report 업데이트 오류 (ID: {report_id}): {e}")
            return False

# 디버그용
print(f"[DEBUG] db_utils.py 로드 완료 - engine: {engine is not None}")