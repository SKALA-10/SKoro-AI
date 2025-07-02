# db_utils.py
# 🔧 데이터베이스 설정 및 연결 관리
import sys
import os
import json
import logging
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from sqlalchemy.exc import OperationalError

# 기존 프로젝트의 DatabaseConfig 사용
from config.settings import DatabaseConfig

# 로깅 설정
logger = logging.getLogger(__name__)

# ====================================
# 예외 클래스
# ====================================

class Module11Error(Exception):
    """모듈 11 전용 예외"""
    pass

class DatabaseError(Module11Error):
    """데이터베이스 관련 예외"""
    pass



# ====================================
# 데이터베이스 설정 및 연결
# ====================================

# 기존 프로젝트의 DatabaseConfig 사용
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ====================================
# 기본 DB 연결 래퍼
# ====================================

class SQLAlchemyDBWrapper:
    """기본 DB 연결 래퍼 클래스"""
    
    def __init__(self, engine):
        self.engine = engine
    
    def fetch_one(self, query, params=None):
        """단일 행 조회"""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            row = result.fetchone()
            return row._asdict() if row else None
    
    def fetch_all(self, query, params=None):
        """전체 행 조회"""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            rows = result.fetchall()
            return [row._asdict() for row in rows] if rows else []
    
    def execute_update(self, query, params=None):
        """업데이트 실행"""
        with self.engine.begin() as conn:  # 자동 트랜잭션 관리
            result = conn.execute(text(query), params or {})
            affected_rows = result.rowcount
            
            # 명시적 검증
            if affected_rows == 0:
                logger.warning(f"업데이트된 행이 없음: query={query[:100]}...")
            else:
                logger.info(f"업데이트 완료: {affected_rows}행 영향")
            
            return affected_rows

# ====================================
# Module11 전용 데이터 액세스 계층
# ====================================

class Module11DataAccess:
    """Module11 전용 SQL 쿼리들이 모여있는 데이터 액세스 계층"""
    
    def __init__(self, db_wrapper: SQLAlchemyDBWrapper):
        self.db = db_wrapper
    
    def _parse_json_field(self, json_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """JSON 필드 안전 파싱"""
        if not json_str:
            return None
        
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON 파싱 실패: {str(e)[:100]}...")
            return None
    
    def get_period_info(self, period_id: int) -> Dict[str, Any]:
        """기간 정보 조회"""
        query = """
        SELECT period_id, year, period_name, order_in_year, is_final,
               start_date, end_date
        FROM periods 
        WHERE period_id = :period_id
        """
        
        result = self.db.fetch_one(query, {'period_id': period_id})
        if not result:
            raise DatabaseError(f"기간 정보를 찾을 수 없음: period_id={period_id}")
        
        return dict(result)

    def get_team_info(self, team_id: int) -> Dict[str, Any]:
        """팀 기본 정보 조회"""
        query = """
        SELECT t.team_id, t.team_name, t.team_description,
               h.headquarter_id, h.headquarter_name, 
               p.part_id, p.part_name
        FROM teams t
        JOIN headquarters h ON t.headquarter_id = h.headquarter_id
        JOIN parts p ON h.part_id = p.part_id
        WHERE t.team_id = :team_id
        """
        
        result = self.db.fetch_one(query, {'team_id': team_id})
        if not result:
            raise DatabaseError(f"팀 정보를 찾을 수 없음: team_id={team_id}")
        
        return dict(result)

    def get_team_members(self, team_id: int) -> List[Dict[str, Any]]:
        """팀원 목록 조회"""
        query = """
        SELECT emp_no, emp_name, email, cl, position, role, salary
        FROM employees 
        WHERE team_id = :team_id
        ORDER BY cl DESC, emp_no
        """
        
        results = self.db.fetch_all(query, {'team_id': team_id})
        if not results:
            logger.warning(f"팀원이 없음: team_id={team_id}")
        
        return [dict(row) for row in results]

    def get_team_kpis(self, team_id: int, year: int) -> List[Dict[str, Any]]:
        """팀 KPI 조회"""
        query = """
        SELECT team_kpi_id, kpi_name, kpi_description, weight, 
               ai_kpi_progress_rate, ai_kpi_analysis_comment
        FROM team_kpis 
        WHERE team_id = :team_id AND year = :year
        ORDER BY weight DESC, team_kpi_id
        """
        
        results = self.db.fetch_all(query, {'team_id': team_id, 'year': year})
        if not results:
            logger.warning(f"팀 KPI가 없음: team_id={team_id}, year={year}")
        
        return [dict(row) for row in results]

    def get_team_performance(self, team_id: int, period_id: int) -> Dict[str, Any]:
        """팀 성과 지표 조회"""
        query = """
        SELECT average_achievement_rate, relative_performance, year_over_year_growth,
               ai_team_overall_analysis_comment
        FROM team_evaluations 
        WHERE team_id = :team_id AND period_id = :period_id
        """
        
        result = self.db.fetch_one(query, {'team_id': team_id, 'period_id': period_id})
        if not result:
            logger.warning(f"팀 성과 데이터 없음: team_id={team_id}, period_id={period_id}")
            return {}
        
        return dict(result)

    def get_collaboration_data(self, team_evaluation_id: int) -> Dict[str, Any]:
        """협업 분석 데이터 조회"""
        query = """
        SELECT ai_collaboration_matrix, ai_team_comparison, ai_team_coaching
        FROM team_evaluations 
        WHERE team_evaluation_id = :team_evaluation_id
        """
        
        result = self.db.fetch_one(query, {'team_evaluation_id': team_evaluation_id})
        if not result:
            logger.warning(f"협업 분석 데이터 없음: team_evaluation_id={team_evaluation_id}")
            return {}
        
        return dict(result)

    def get_individual_risks(self, team_evaluation_id: int, is_final: bool) -> List[Dict[str, Any]]:
        """개인별 리스크 분석 결과 조회"""
        if is_final:
            # 연말: final_evaluation_reports
            query = """
            SELECT emp_no, ai_growth_coaching, score, contribution_rate,
                   ai_annual_performance_summary_comment, cl_reason
            FROM final_evaluation_reports 
            WHERE team_evaluation_id = :team_evaluation_id
            ORDER BY score DESC
            """
        else:
            # 분기: feedback_reports
            query = """
            SELECT emp_no, ai_growth_coaching, contribution_rate,
                   ai_overall_contribution_summary_comment, attitude
            FROM feedback_reports 
            WHERE team_evaluation_id = :team_evaluation_id
            ORDER BY contribution_rate DESC
            """
        
        results = self.db.fetch_all(query, {'team_evaluation_id': team_evaluation_id})
        
        # ai_growth_coaching JSON 파싱
        parsed_results = []
        for row in results:
            row_dict = dict(row)
            row_dict['ai_growth_coaching'] = self._parse_json_field(row.get('ai_growth_coaching'))
            parsed_results.append(row_dict)
        
        if not parsed_results:
            logger.warning(f"개인 평가 데이터 없음: team_evaluation_id={team_evaluation_id}")
        
        return parsed_results

    def get_previous_quarter_data(self, team_id: int, period_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """전분기 데이터 조회 (분기용)"""
        current_year = period_info['year']
        current_order = period_info['order_in_year']
        
        # 이전 분기 계산
        if current_order > 1:
            # 같은 해 이전 분기
            prev_year = current_year
            prev_order = current_order - 1
        else:
            # 전년도 4분기
            prev_year = current_year - 1
            prev_order = 4
        
        query = """
        SELECT te.average_achievement_rate, te.relative_performance, 
               te.ai_risk, te.overall_comment,
               p.period_name
        FROM team_evaluations te
        JOIN periods p ON te.period_id = p.period_id
        WHERE te.team_id = :team_id AND p.year = :year AND p.order_in_year = :order_in_year
        LIMIT 1
        """
        
        result = self.db.fetch_one(query, {
            'team_id': team_id, 
            'year': prev_year, 
            'order_in_year': prev_order
        })
        
        if result:
            result_dict = dict(result)
            result_dict['ai_risk'] = self._parse_json_field(result.get('ai_risk'))
            # overall_comment는 TEXT이므로 JSON 파싱하지 않음
            logger.info(f"전분기 데이터 조회 완료: {prev_year}년 {prev_order}분기")
            return result_dict
        
        logger.warning(f"전분기 데이터 없음: team_id={team_id}, {prev_year}년 {prev_order}분기")
        return None

    def get_temp_evaluations(self, team_id: int) -> List[Dict[str, Any]]:
        """중간평가 데이터 조회 (연말용)"""
        logger.info(f"DB에서 temp_evaluations 조회: team_id={team_id}")
        
        query = """
        SELECT te.temp_evaluation_id, te.ai_reason, te.score, te.raw_score,
               te.manager_score, te.comment, te.reason, te.status, te.emp_no
        FROM temp_evaluations te
        WHERE te.emp_no IN (
            SELECT emp_no FROM employees WHERE team_id = :team_id
        )
        ORDER BY te.score DESC
        """
        
        results = self.db.fetch_all(query, {'team_id': team_id})
        
        if not results:
            logger.warning(f"중간평가 데이터 없음: team_id={team_id}")
            return []
        
        logger.info(f"중간평가 데이터 조회 완료: {len(results)}건")
        return [dict(row) for row in results]

    def update_team_evaluations(self, team_evaluation_id: int, save_data: dict) -> int:
        """team_evaluations 테이블 업데이트"""
        
        # UPDATE 쿼리 동적 생성
        set_clauses = []
        params = {'team_evaluation_id': team_evaluation_id}
        
        for column, value in save_data.items():
            set_clauses.append(f"{column} = :{column}")
            params[column] = value
        
        query = f"""
        UPDATE team_evaluations 
        SET {', '.join(set_clauses)}
        WHERE team_evaluation_id = :team_evaluation_id
        """
        
        logger.info(f"실행할 쿼리: {query}")
        logger.info(f"파라미터 키: {list(params.keys())}")
        
        # 실제 업데이트 실행
        affected_rows = self.db.execute_update(query, params)
        
        return affected_rows

    def verify_team_evaluation_exists(self, team_evaluation_id: int) -> bool:
        """team_evaluation 레코드 존재 여부 확인"""
        query = "SELECT COUNT(*) as cnt FROM team_evaluations WHERE team_evaluation_id = :team_evaluation_id"
        result = self.db.fetch_one(query, {'team_evaluation_id': team_evaluation_id})
        
        exists = result['cnt'] > 0 if result else False
        logger.info(f"team_evaluation_id {team_evaluation_id} 존재 여부: {exists}")
        return exists

    def verify_save_success(self, team_evaluation_id: int, save_data: dict) -> None:
        """저장 성공 여부 검증"""
        logger.info("저장 결과 검증 시작...")
        
        # 저장된 데이터 다시 조회
        columns = list(save_data.keys())
        query = f"""
        SELECT {', '.join(columns)}
        FROM team_evaluations 
        WHERE team_evaluation_id = :team_evaluation_id
        """
        
        result = self.db.fetch_one(query, {'team_evaluation_id': team_evaluation_id})
        
        if not result:
            raise DatabaseError("저장 검증 실패: 데이터를 다시 조회할 수 없음")
        
        # 각 필드별 검증
        for column in columns:
            saved_value = result.get(column)
            if not saved_value:
                logger.warning(f"⚠️ {column} 필드가 비어있음")
            else:
                logger.info(f"✅ {column} 저장 확인: {len(str(saved_value))}자")
        
        logger.info("저장 결과 검증 완료")


# ====================================
# 초기화 함수
# ====================================

def init_database():
    """데이터베이스 연결 초기화"""
    try:
        # 기존 프로젝트의 DatabaseConfig와 engine 사용
        db_wrapper = SQLAlchemyDBWrapper(engine)
        data_access = Module11DataAccess(db_wrapper)
        
        logger.info("데이터베이스 연결 초기화 완료")
        return data_access
        
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {str(e)}")
        raise DatabaseError(f"DB 초기화 중 오류 발생: {str(e)}")