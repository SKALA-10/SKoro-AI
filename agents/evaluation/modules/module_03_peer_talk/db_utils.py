# ai-performance-management-system/shared/tools/py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from typing import Optional, List, Dict, Any
from collections import defaultdict
import json

from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# --- 도우미 함수: SQLAlchemy Row 객체를 딕셔너리로 변환 ---
def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환합니다."""
    if row is None:
        return {}
    return row._asdict() # ._asdict() 사용


# --- 모듈 3 전용 데이터 조회 함수들 ---

def fetch_peer_evaluations_for_target(period_id: int, target_emp_no: str) -> List[Dict]:
    """
    특정 분기/평가받는 사번의 동료 평가 리스트 조회
    """
    with engine.connect() as connection:
        query = text("""
            SELECT 
                pe.peer_evaluation_id,
                te.period_id,
                pe.target_emp_no AS target_emp_no,
                pe.emp_no AS evaluator_emp_no,
                pe.weight AS weight
            FROM team_evaluations te
            JOIN peer_evaluations pe ON te.team_evaluation_id = pe.team_evaluation_id
            WHERE te.period_id = :period_id
              AND pe.target_emp_no = :target_emp_no
        """)
        results = connection.execute(query, {"period_id": period_id, "target_emp_no": target_emp_no}).fetchall()
        return [row_to_dict(row) for row in results]


def fetch_keywords_for_peer_evaluations(peer_evaluation_ids: List[int]) -> Dict[int, List[str]]:
    """
    동료 평가 ID 리스트별 키워드(시스템/커스텀) 모음 조회
    """
    if not peer_evaluation_ids:
        return {}
    
    with engine.connect() as connection:
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
        
        results = connection.execute(query, params).fetchall()
        keyword_map = defaultdict(list)
        for row in results:
            row_dict = row_to_dict(row)
            keyword_map[row_dict["peer_evaluation_id"]].append(row_dict["keyword"])
        return dict(keyword_map)


def fetch_tasks_for_peer_evaluations(peer_evaluation_ids: List[int]) -> Dict[int, List[int]]:
    """
    동료 평가별 task_id 리스트 조회
    peer_evaluations의 target_emp_no(피평가자)의 tasks를 조회
    """
    if not peer_evaluation_ids:
        return {}
    
    with engine.connect() as connection:
        placeholders = ','.join([f':id_{i}' for i in range(len(peer_evaluation_ids))])
        params = {f'id_{i}': peer_id for i, peer_id in enumerate(peer_evaluation_ids)}
        
        query = text(f"""
            SELECT DISTINCT
                pe.peer_evaluation_id,
                t.task_id
            FROM peer_evaluations pe
            LEFT JOIN tasks t ON t.emp_no = pe.target_emp_no
            WHERE pe.peer_evaluation_id IN ({placeholders})
              AND t.task_id IS NOT NULL
        """)
        
        results = connection.execute(query, params).fetchall()
        task_map = defaultdict(list)
        for row in results:
            row_dict = row_to_dict(row)
            task_map[row_dict["peer_evaluation_id"]].append(row_dict["task_id"])
        return dict(task_map)


def fetch_task_summaries(period_id: int, task_ids: List[int]) -> Dict[int, str]:
    """
    task_summaries에서 업무 요약 조회
    task_performance 컬럼 사용
    """
    if not task_ids:
        return {}
    
    with engine.connect() as connection:
        placeholders = ','.join([f':task_{i}' for i in range(len(task_ids))])
        params = {f'task_{i}': task_id for i, task_id in enumerate(task_ids)}
        
        # task_summaries 테이블에서 task_performance 조회
        query = text(f"""
            SELECT ts.task_id, ts.task_performance as summary
            FROM task_summaries ts
            WHERE ts.task_id IN ({placeholders})
              AND ts.period_id = :period_id
        """)
        
        # period_id 파라미터 추가
        params['period_id'] = period_id
        
        results = connection.execute(query, params).fetchall()
        return {row_to_dict(row)["task_id"]: row_to_dict(row)["summary"] for row in results}


def get_all_employees_in_period(period_id: int) -> List[str]:
    """특정 분기에 동료평가를 받은 모든 직원 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT pe.target_emp_no
            FROM team_evaluations te
            JOIN peer_evaluations pe ON te.team_evaluation_id = pe.team_evaluation_id
            WHERE te.period_id = :period_id
            ORDER BY pe.target_emp_no
        """)
        
        results = connection.execute(query, {"period_id": period_id}).fetchall()
        return [row_to_dict(row)["target_emp_no"] for row in results]


def get_team_evaluation_id_by_emp_and_period(emp_no: str, period_id: int) -> Optional[int]:
    """해당 분기와 직원에 해당하는 team_evaluation_id 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT te.team_evaluation_id
            FROM team_evaluations te
            JOIN teams t ON te.team_id = t.team_id
            JOIN employees e ON e.team_id = t.team_id
            WHERE te.period_id = :period_id
              AND e.emp_no = :emp_no
            LIMIT 1
        """)
        
        result = connection.execute(query, {
            "period_id": period_id,
            "emp_no": emp_no
        }).fetchone()
        
        return row_to_dict(result)["team_evaluation_id"] if result else None


def fetch_feedback_report_id(team_id: int, period_id: int, emp_no: str) -> Optional[int]:
    """
    team_id, period_id, emp_no로 feedback_report_id를 조회
    """
    from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_evaluation_id
    team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
    if not team_evaluation_id:
        return None
    with engine.connect() as connection:
        query = text("""
            SELECT feedback_report_id FROM feedback_reports
            WHERE team_evaluation_id = :team_evaluation_id AND emp_no = :emp_no
        """)
        result = connection.execute(query, {
            "team_evaluation_id": team_evaluation_id,
            "emp_no": emp_no
        })
        return result.scalar_one_or_none()


# --- 모듈 3 전용 DB 저장 함수들 ---

def format_peer_evaluation_result(strengths: List[str], concerns: List[str], collaboration_observations: List[str]) -> str:
    """강점, 우려, 협업관찰을 JSON 문자열로 포맷팅"""
    strength = strengths[0] if strengths else "동료들로부터 긍정적인 평가를 받고 있습니다."
    concern = concerns[0] if concerns else "지속적인 성장을 위한 개선 영역이 있습니다."
    collaboration = collaboration_observations[0] if collaboration_observations else "팀 내에서 협업에 참여하고 있습니다."

    result_dict = {
        "strengths": strength,
        "concerns": concern,
        "collaboration_observations": collaboration
    }
    
    return json.dumps(result_dict, ensure_ascii=False, indent=2)


def save_feedback_peer_summary(emp_no: str, period_id: int, ai_peer_talk_summary: str) -> Optional[int]:
    """
    feedback_reports 테이블에 동료평가 요약 저장
    """
    try:
        team_evaluation_id = get_team_evaluation_id_by_emp_and_period(emp_no, period_id)
        if not team_evaluation_id:
            print(f"Warning: team_evaluation_id를 찾을 수 없습니다. emp_no={emp_no}, period_id={period_id}")
            return None
        
        with engine.connect() as connection:
            # 기존 레코드 확인
            check_query = text("""
                SELECT feedback_report_id 
                FROM feedback_reports 
                WHERE team_evaluation_id = :team_eval_id AND emp_no = :emp_no
            """)
            
            existing = connection.execute(check_query, {
                "team_eval_id": team_evaluation_id,
                "emp_no": emp_no
            }).fetchone()
            
            if existing:
                # 업데이트
                update_query = text("""
                    UPDATE feedback_reports 
                    SET ai_peer_talk_summary = :ai_peer_talk_summary,
                        updated_at = NOW()
                    WHERE feedback_report_id = :feedback_id
                """)
                
                connection.execute(update_query, {
                    "ai_peer_talk_summary": ai_peer_talk_summary,
                    "feedback_id": row_to_dict(existing)["feedback_report_id"]
                })
                connection.commit()
                
                print(f"DB: feedback_reports 업데이트 완료 - emp_no={emp_no}")
                return row_to_dict(existing)["feedback_report_id"]
                
            else:
                # 새 레코드 삽입
                insert_query = text("""
                    INSERT INTO feedback_reports (team_evaluation_id, emp_no, ai_peer_talk_summary, created_at, updated_at)
                    VALUES (:team_eval_id, :emp_no, :ai_peer_talk_summary, NOW(), NOW())
                """)
                
                result = connection.execute(insert_query, {
                    "team_eval_id": team_evaluation_id,
                    "emp_no": emp_no,
                    "ai_peer_talk_summary": ai_peer_talk_summary
                })
                connection.commit()
                
                # 삽입된 ID 조회
                new_id = result.lastrowid
                print(f"DB: feedback_reports 새 레코드 삽입 완료 - emp_no={emp_no}, ID={new_id}")
                return new_id
        
    except Exception as e:
        print(f"DB 저장 실패 (feedback_reports): {str(e)}")
        return None


def save_final_evaluation_peer_summary(emp_no: str, period_id: int, ai_peer_talk_summary: str) -> Optional[int]:
    """
    final_evaluation_reports 테이블에 동료평가 요약 저장
    """
    try:
        team_evaluation_id = get_team_evaluation_id_by_emp_and_period(emp_no, period_id)
        if not team_evaluation_id:
            print(f"Warning: team_evaluation_id를 찾을 수 없습니다. emp_no={emp_no}, period_id={period_id}")
            return None
        
        with engine.connect() as connection:
            # 기존 레코드 확인
            check_query = text("""
                SELECT final_evaluation_report_id 
                FROM final_evaluation_reports 
                WHERE team_evaluation_id = :team_eval_id AND emp_no = :emp_no
            """)
            
            existing = connection.execute(check_query, {
                "team_eval_id": team_evaluation_id,
                "emp_no": emp_no
            }).fetchone()
            
            if existing:
                # 업데이트
                update_query = text("""
                    UPDATE final_evaluation_reports 
                    SET ai_peer_talk_summary = :ai_peer_talk_summary,
                        updated_at = NOW()
                    WHERE final_evaluation_report_id = :final_evaluation_id
                """)
                
                connection.execute(update_query, {
                    "ai_peer_talk_summary": ai_peer_talk_summary,
                    "final_evaluation_id": row_to_dict(existing)["final_evaluation_report_id"]
                })
                connection.commit()
                
                print(f"DB: final_evaluation_reports 업데이트 완료 - emp_no={emp_no}")
                return row_to_dict(existing)["final_evaluation_report_id"]
                
            else:
                # 새 레코드 삽입
                insert_query = text("""
                    INSERT INTO final_evaluation_reports (team_evaluation_id, emp_no, ai_peer_talk_summary, created_at, updated_at)
                    VALUES (:team_eval_id, :emp_no, :ai_peer_talk_summary, NOW(), NOW())
                """)
                
                result = connection.execute(insert_query, {
                    "team_eval_id": team_evaluation_id,
                    "emp_no": emp_no,
                    "ai_peer_talk_summary": ai_peer_talk_summary
                })
                connection.commit()
                
                # 삽입된 ID 조회
                new_id = result.lastrowid
                print(f"DB: final_evaluation_reports 새 레코드 삽입 완료 - emp_no={emp_no}, ID={new_id}")
                return new_id
        
    except Exception as e:
        print(f"DB 저장 실패 (final_evaluation_reports): {str(e)}")
        return None


def check_existing_peer_evaluation_by_period(period_id: int, emp_no: str) -> bool:
    """분기별로 해당 직원의 동료평가 결과가 이미 존재하는지 확인"""
    try:
        with engine.connect() as connection:
            if period_id == 4:
                query = text("""
                    SELECT fer.final_evaluation_report_id
                    FROM final_evaluation_reports fer
                    JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                    WHERE te.period_id = :period_id
                      AND fer.emp_no = :emp_no
                      AND fer.ai_peer_talk_summary IS NOT NULL
                      AND fer.ai_peer_talk_summary != ''
                    LIMIT 1
                """)
            else:
                query = text("""
                    SELECT fr.feedback_report_id
                    FROM feedback_reports fr
                    JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                    WHERE te.period_id = :period_id
                      AND fr.emp_no = :emp_no
                      AND fr.ai_peer_talk_summary IS NOT NULL
                      AND fr.ai_peer_talk_summary != ''
                    LIMIT 1
                """)
            
            result = connection.execute(query, {
                "period_id": period_id,
                "emp_no": emp_no
            }).fetchone()
            
            return result is not None
            
    except Exception as e:
        print(f"기존 동료평가 확인 실패: {str(e)}")
        return False