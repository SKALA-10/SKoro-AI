"""
데이터베이스 관련 유틸리티 모듈
팀 피드백 요약 시스템 - 데이터베이스 연결 및 쿼리 함수
"""

import pandas as pd
from typing import List, Dict, Optional
from sqlalchemy import create_engine, text


class DatabaseManager:
    """데이터베이스 연결 및 쿼리 관리 클래스"""
    
    def __init__(self, database_url: str):
        """
        데이터베이스 매니저 초기화
        
        Args:
            database_url: 데이터베이스 연결 URL
        """
        self.engine = create_engine(database_url, pool_pre_ping=True)
    
    def test_connection(self) -> bool:
        """
        데이터베이스 연결 테스트
        
        Returns:
            bool: 연결 성공 시 True, 실패 시 False
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1 as test"))
                return True
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            return False
    
    def get_teams_to_summarize(self) -> List[Dict]:
        """
        요약이 필요한 팀 목록 조회
        
        Returns:
            List[Dict]: 요약 대상 팀 정보 리스트
        """
        query = text("""
        SELECT 
            ef.team_evaluation_id,
            ef.period_id,
            t.team_name,
            COUNT(ef.evaluation_feedback_id) as feedback_count
        FROM evaluation_feedbacks ef
        JOIN team_evaluations te ON ef.team_evaluation_id = te.team_evaluation_id
        JOIN teams t ON te.team_id = t.team_id
        WHERE ef.content IS NOT NULL 
        AND ef.content != ''
        AND NOT EXISTS (
            SELECT 1 FROM evaluation_feedback_summaries efs 
            WHERE efs.team_evaluation_id = ef.team_evaluation_id 
            AND efs.period_id = ef.period_id
        )
        GROUP BY ef.team_evaluation_id, ef.period_id, t.team_name
        ORDER BY t.team_name
        """)
        
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
                return df.to_dict('records')
        except Exception as e:
            print(f"❌ 팀 목록 조회 실패: {e}")
            return []
    
    def get_team_feedbacks(self, team_evaluation_id: int, period_id: int) -> List[Dict]:
        """
        특정 팀의 피드백 내용 조회
        
        Args:
            team_evaluation_id: 팀 평가 ID
            period_id: 기간 ID
            
        Returns:
            List[Dict]: 피드백 리스트
        """
        query = text("""
        SELECT 
            ef.content
        FROM evaluation_feedbacks ef
        WHERE ef.team_evaluation_id = :team_evaluation_id 
        AND ef.period_id = :period_id
        AND ef.content IS NOT NULL 
        AND ef.content != ''
        ORDER BY ef.evaluation_feedback_id
        """)
        
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, {
                    'team_evaluation_id': team_evaluation_id,
                    'period_id': period_id
                })
                feedbacks = [{'content': row.content, 'emp_name': '팀원'} for row in result]
                return feedbacks
        except Exception as e:
            print(f"❌ 피드백 조회 실패: {e}")
            return []
    
    def get_team_name(self, team_evaluation_id: int) -> Optional[str]:
        """
        팀 평가 ID로 팀 이름 조회
        
        Args:
            team_evaluation_id: 팀 평가 ID
            
        Returns:
            Optional[str]: 팀 이름 또는 None
        """
        query = text("""
        SELECT t.team_name
        FROM team_evaluations te
        JOIN teams t ON te.team_id = t.team_id
        WHERE te.team_evaluation_id = :team_evaluation_id
        """)
        
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, {'team_evaluation_id': team_evaluation_id})
                row = result.fetchone()
                return row._mapping['team_name'] if row else None
        except Exception as e:
            print(f"❌ 팀 이름 조회 실패: {e}")
            return None
    
    def save_summary(self, team_evaluation_id: int, period_id: int, summary_content: str) -> bool:
        """
        요약 결과를 데이터베이스에 저장
        
        Args:
            team_evaluation_id: 팀 평가 ID
            period_id: 기간 ID
            summary_content: 요약 내용
            
        Returns:
            bool: 저장 성공 시 True, 실패 시 False
        """
        query = text("""
        INSERT INTO evaluation_feedback_summaries 
        (team_evaluation_id, period_id, content)
        VALUES (:team_evaluation_id, :period_id, :content)
        ON DUPLICATE KEY UPDATE
        content = VALUES(content)
        """)
        
        try:
            with self.engine.connect() as connection:
                connection.execute(query, {
                    'team_evaluation_id': team_evaluation_id,
                    'period_id': period_id,
                    'content': summary_content
                })
                connection.commit()
                return True
        except Exception as e:
            print(f"❌ 요약 저장 실패: {e}")
            return False
    
    def get_summary_results(self) -> pd.DataFrame:
        """
        저장된 요약 결과 조회
        
        Returns:
            pd.DataFrame: 요약 결과 데이터프레임
        """
        query = text("""
        SELECT 
            t.team_name,
            p.period_name,
            LENGTH(efs.content) as summary_length,
            LEFT(efs.content, 100) as summary_preview
        FROM evaluation_feedback_summaries efs
        JOIN team_evaluations te ON efs.team_evaluation_id = te.team_evaluation_id
        JOIN teams t ON te.team_id = t.team_id
        LEFT JOIN periods p ON efs.period_id = p.period_id
        ORDER BY t.team_name
        """)
        
        try:
            with self.engine.connect() as connection:
                return pd.read_sql(query, connection)
        except Exception as e:
            print(f"❌ 결과 확인 실패: {e}")
            return pd.DataFrame()
    
    def get_detailed_summary(self, team_evaluation_id: int, period_id: int) -> Optional[Dict]:
        """
        특정 팀의 상세 요약 내용 조회
        
        Args:
            team_evaluation_id: 팀 평가 ID
            period_id: 기간 ID
            
        Returns:
            Optional[Dict]: 상세 요약 정보 또는 None
        """
        query = text("""
        SELECT 
            t.team_name,
            p.period_name,
            efs.content
        FROM evaluation_feedback_summaries efs
        JOIN team_evaluations te ON efs.team_evaluation_id = te.team_evaluation_id
        JOIN teams t ON te.team_id = t.team_id
        LEFT JOIN periods p ON efs.period_id = p.period_id
        WHERE efs.team_evaluation_id = :team_evaluation_id 
        AND efs.period_id = :period_id
        """)
        
        try:
            with self.engine.connect() as connection:
                result = connection.execute(query, {
                    'team_evaluation_id': team_evaluation_id,
                    'period_id': period_id
                })
                row = result.fetchone()
                
                if row:
                    return {
                        'team_name': row._mapping['team_name'],
                        'period_name': row._mapping['period_name'],
                        'content': row._mapping['content']
                    }
                return None
                
        except Exception as e:
            print(f"❌ 상세 요약 조회 실패: {e}")
            return None