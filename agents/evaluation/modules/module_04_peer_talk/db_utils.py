# agents/evaluation/modules/module_04_peer_talk/db_utils.py

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from typing import Optional, List, Dict, Any
from collections import defaultdict

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# --- 도우미 함수: SQLAlchemy Row 객체를 딕셔너리로 변환 ---
def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환합니다."""
    if row is None:
        return {}
    return dict(row._mapping) if hasattr(row, '_mapping') else dict(row)


# --- 데이터 조회 함수들 ---

def fetch_peer_evaluations_for_target(engine, period_id: int, target_emp_no: str) -> List[Dict]:
    """
    특정 분기/평가받는 사번의 동료 평가 리스트 조회
    """
    with engine.connect() as conn:
        query = text("""
            SELECT 
                pe.peer_evaluation_id,
                te.period_id,
                pe.target_emp_no AS target_emp_no,
                pe.emp_no AS evaluator_emp_no,
                pe.progress AS weight
            FROM team_evaluations te
            JOIN peer_evaluations pe ON te.team_evaluation_id = pe.team_evaluation_id
            WHERE te.period_id = :period_id
              AND pe.target_emp_no = :target_emp_no
        """)
        results = conn.execute(query, {"period_id": period_id, "target_emp_no": target_emp_no}).fetchall()
        return [row_to_dict(row) for row in results]


def fetch_keywords_for_peer_evaluations(engine, peer_evaluation_ids: List[int]) -> Dict[int, List[str]]:
    """
    동료 평가 ID 리스트별 키워드(시스템/커스텀) 모음 조회
    """
    if not peer_evaluation_ids:
        return {}
    
    with engine.connect() as conn:
        # IN 절을 위한 파라미터 처리
        placeholders = ','.join([f':id_{i}' for i in range(len(peer_evaluation_ids))])
        params = {f'id_{i}': peer_id for i, peer_id in enumerate(peer_evaluation_ids)}
        
        query = text(f"""
            SELECT 
                pek.peer_evaluation_id,
                COALESCE(k.keyword_name, pek.custom_keyword) AS keyword
            FROM peer_evaluation_keywords pek
            LEFT JOIN keywords k ON pek.keyword_id = k.keyword_id
            WHERE pek.peer_evaluation_id IN ({placeholders})
        """)
        
        results = conn.execute(query, params).fetchall()
        keyword_map = defaultdict(list)
        for row in results:
            row_dict = row_to_dict(row)
            keyword_map[row_dict["peer_evaluation_id"]].append(row_dict["keyword"])
        return dict(keyword_map)


def fetch_tasks_for_peer_evaluations_fixed(engine, peer_evaluation_ids: List[int]) -> Dict[int, List[int]]:
    """
    동료 평가별 task_id 리스트 조회 (수정된 버전)
    peer_evaluations의 emp_no(평가자)와 target_emp_no(피평가자) 모두 고려
    """
    if not peer_evaluation_ids:
        return {}
    
    with engine.connect() as conn:
        placeholders = ','.join([f':id_{i}' for i in range(len(peer_evaluation_ids))])
        params = {f'id_{i}': peer_id for i, peer_id in enumerate(peer_evaluation_ids)}
        
        # 실제 tasks 테이블 구조에 맞는 쿼리
        # emp_no 컬럼을 사용하여 조인
        query = text(f"""
            SELECT DISTINCT
                pe.peer_evaluation_id,
                t.task_id
            FROM peer_evaluations pe
            LEFT JOIN tasks t ON t.emp_no = pe.target_emp_no
            WHERE pe.peer_evaluation_id IN ({placeholders})
              AND t.task_id IS NOT NULL
        """)
        
        results = conn.execute(query, params).fetchall()
        task_map = defaultdict(list)
        for row in results:
            row_dict = row_to_dict(row)
            task_map[row_dict["peer_evaluation_id"]].append(row_dict["task_id"])
        return dict(task_map)


def fetch_task_summaries_fixed(engine, period_id: int, task_ids: List[int]) -> Dict[int, str]:
    """
    task_summaries에서 업무 요약 조회 (수정된 버전)
    task_performance 컬럼 사용
    """
    if not task_ids:
        return {}
    
    with engine.connect() as conn:
        placeholders = ','.join([f':task_{i}' for i in range(len(task_ids))])
        params = {f'task_{i}': task_id for i, task_id in enumerate(task_ids)}
        
        # tasks 테이블에서 직접 task_performance 조회
        # (task_summaries 테이블이 비어있을 수 있으므로)
        query = text(f"""
            SELECT task_id, task_performance as summary
            FROM tasks
            WHERE task_id IN ({placeholders})
        """)
        
        results = conn.execute(query, params).fetchall()
        return {row_to_dict(row)["task_id"]: row_to_dict(row)["summary"] for row in results}


def get_all_employees_in_period(engine, period_id: int) -> List[str]:
    """특정 분기에 동료평가를 받은 모든 직원 조회"""
    
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT DISTINCT pe.target_emp_no
                FROM team_evaluations te
                JOIN peer_evaluations pe ON te.team_evaluation_id = pe.team_evaluation_id
                WHERE te.period_id = :period_id
                ORDER BY pe.target_emp_no
            """)
            
            results = conn.execute(query, {"period_id": period_id}).fetchall()
            employee_list = [row_to_dict(row)["target_emp_no"] for row in results]
            
            print(f"📊 {period_id}분기 동료평가 대상자: {len(employee_list)}명")
            for i, emp_no in enumerate(employee_list, 1):
                print(f"  {i}. {emp_no}")
                
            return employee_list
            
    except Exception as e:
        print(f"❌ 직원 목록 조회 실패: {str(e)}")
        return []


def get_team_evaluation_id(engine, period_id: int, emp_no: str) -> int:
    """
    해당 분기와 직원에 해당하는 team_evaluation_id 조회
    """
    try:
        with engine.connect() as conn:
            # 해당 직원이 속한 팀의 해당 분기 team_evaluation 조회
            query = text("""
                SELECT te.team_evaluation_id
                FROM team_evaluations te
                JOIN teams t ON te.team_id = t.team_id
                JOIN employees e ON e.team_id = t.team_id
                WHERE te.period_id = :period_id
                  AND e.emp_no = :emp_no
                LIMIT 1
            """)
            
            result = conn.execute(query, {
                "period_id": period_id,
                "emp_no": emp_no
            }).fetchone()
            
            if result:
                return row_to_dict(result)["team_evaluation_id"]
            else:
                print(f"[DatabaseStorageAgent] team_evaluation_id 조회 실패: period_id={period_id}, emp_no={emp_no}")
                return None
                
    except Exception as e:
        print(f"[DatabaseStorageAgent] team_evaluation_id 조회 오류: {str(e)}")
        return None


def check_feedback_reports(engine, period_id: int, emp_no: str) -> bool:
    """해당 직원의 피드백 보고서가 이미 존재하는지 확인"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT fr.feedback_report_id
                FROM feedback_reports fr
                JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                WHERE te.period_id = :period_id
                  AND fr.emp_no = :emp_no
                  AND fr.peer_review_result IS NOT NULL
                LIMIT 1
            """)
            
            result = conn.execute(query, {
                "period_id": period_id,
                "emp_no": emp_no
            }).fetchone()
            
            return result is not None
    except Exception as e:
        print(f"❌ 피드백 보고서 확인 실패: {str(e)}")
        return False


# --- 데이터 저장 함수들 ---

def save_peer_review_result_to_db(engine, period_id: int, emp_no: str, peer_review_result: str) -> bool:
    """
    동료평가 분석 결과를 feedback_reports 테이블에 저장
    """
    try:
        # 1. team_evaluation_id 조회
        team_evaluation_id = get_team_evaluation_id(engine, period_id, emp_no)
        
        if not team_evaluation_id:
            print(f"[DatabaseStorageAgent] {emp_no}: team_evaluation_id를 찾을 수 없음")
            return False
        
        # 2. 기존 데이터 확인 및 처리
        with engine.connect() as conn:
            # 기존 데이터 확인
            check_query = text("""
                SELECT feedback_report_id 
                FROM feedback_reports 
                WHERE team_evaluation_id = :team_evaluation_id 
                  AND emp_no = :emp_no
            """)
            existing = conn.execute(check_query, {
                "team_evaluation_id": team_evaluation_id,
                "emp_no": emp_no
            }).fetchone()
            
            if existing:
                # 기존 데이터 업데이트
                update_query = text("""
                    UPDATE feedback_reports 
                    SET peer_review_result = :peer_review_result
                    WHERE feedback_report_id = :feedback_report_id
                """)
                conn.execute(update_query, {
                    "peer_review_result": peer_review_result,
                    "feedback_report_id": row_to_dict(existing)["feedback_report_id"]
                })
                conn.commit()
                
                print(f"[DatabaseStorageAgent] {emp_no}: 기존 데이터 업데이트 완료")
                
            else:
                # 새 데이터 삽입
                insert_query = text("""
                    INSERT INTO feedback_reports 
                    (team_evaluation_id, emp_no, peer_review_result)
                    VALUES (:team_evaluation_id, :emp_no, :peer_review_result)
                """)
                result = conn.execute(insert_query, {
                    "team_evaluation_id": team_evaluation_id,
                    "emp_no": emp_no,
                    "peer_review_result": peer_review_result
                })
                conn.commit()
                
                feedback_report_id = result.lastrowid
                print(f"[DatabaseStorageAgent] {emp_no}: 새 데이터 삽입 완료 (ID: {feedback_report_id})")
        
        return True
        
    except Exception as e:
        print(f"[DatabaseStorageAgent] {emp_no}: DB 저장 실패 - {str(e)}")
        import traceback
        traceback.print_exc()
        return False