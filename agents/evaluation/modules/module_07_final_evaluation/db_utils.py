# ================================================================
# db_utils_module7.py - 모듈 7 데이터베이스 관련 유틸리티
# ================================================================

import json
import statistics
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from dotenv import load_dotenv

load_dotenv()

from config.settings import *

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

# ================================================================
# 팀 단위 DB 조회 함수들
# ================================================================

def fetch_team_members(team_id: str) -> List[Dict]:
    """팀원 기본 정보 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT emp_no, emp_name, cl, position, team_id
            FROM employees 
            WHERE team_id = :team_id
            ORDER BY cl DESC, position DESC
        """)
        results = connection.execute(query, {"team_id": team_id}).fetchall()
        return [row_to_dict(row) for row in results]

def fetch_team_achievement_data(team_id: str, period_id: int) -> List[Dict]:
    """팀 전체 달성률 데이터 한 번에 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT fer.emp_no, fer.contribution_rate, 
                   fer.ai_annual_achievement_rate,
                   fer.ai_annual_performance_summary_comment, 
                   fer.ai_peer_talk_summary,
                   e.emp_name, e.cl, e.position
            FROM final_evaluation_reports fer
            JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
            JOIN employees e ON fer.emp_no = e.emp_no
            WHERE e.team_id = :team_id AND te.period_id = :period_id
            ORDER BY fer.ai_annual_achievement_rate DESC
        """)
        results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]

def fetch_team_fourp_data(team_id: str, period_id: int) -> List[Dict]:
    """팀 전체 4P 데이터 한 번에 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT fer.emp_no, fer.ai_4p_evaluation, e.emp_name
            FROM final_evaluation_reports fer
            JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
            JOIN employees e ON fer.emp_no = e.emp_no
            WHERE e.team_id = :team_id AND te.period_id = :period_id
        """)
        results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        
        fourp_data = []
        for row in results:
            try:
                fourp_evaluation = json.loads(row.ai_4p_evaluation) if row.ai_4p_evaluation else {}
                fourp_data.append({
                    "emp_no": row.emp_no,
                    "emp_name": row.emp_name,
                    "fourp_results": fourp_evaluation
                })
            except json.JSONDecodeError:
                print(f"4P JSON 파싱 실패: {row.emp_no}")
                fourp_data.append({
                    "emp_no": row.emp_no,
                    "emp_name": row.emp_name,
                    "fourp_results": {}
                })
        
        return fourp_data

def fetch_team_quarterly_data(team_id: str, period_id: int) -> Dict:
    """팀 전체 분기별 Task 데이터 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT t.emp_no, ts.period_id, ts.task_id, t.task_name,
                   ts.ai_contribution_score, ts.ai_analysis_comment_task, ts.task_performance
            FROM task_summaries ts
            JOIN tasks t ON ts.task_id = t.task_id
            JOIN employees e ON t.emp_no = e.emp_no
            WHERE e.team_id = :team_id AND ts.period_id <= :period_id
            ORDER BY t.emp_no, ts.period_id, ts.task_id
        """)
        results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        
        # emp_no별로 그룹화
        quarterly_data = {}
        for row in results:
            emp_no = row.emp_no
            if emp_no not in quarterly_data:
                quarterly_data[emp_no] = []
            quarterly_data[emp_no].append(row_to_dict(row))
        
        return quarterly_data

def batch_update_temp_evaluations(score_data: List[Dict], period_id: int = 4) -> Dict:
    """팀 전체 temp_evaluations 배치 업데이트 (raw_score, score 모두 저장)"""
    success_count = 0
    failed_members = []
    
    with engine.connect() as connection:
        try:
            for data in score_data:
                try:
                    # team_evaluation_id 조회
                    team_eval_query = text("""
                        SELECT te.team_evaluation_id
                        FROM team_evaluations te
                        JOIN employees e ON e.team_id = te.team_id
                        WHERE e.emp_no = :emp_no AND te.period_id = :period_id
                    """)
                    
                    team_eval_result = connection.execute(team_eval_query, {
                        "emp_no": data["emp_no"],
                        "period_id": period_id
                    }).scalar_one_or_none()
                    
                    if not team_eval_result:
                        failed_members.append(data["emp_no"])
                        print(f"DB 업데이트 실패: {data['emp_no']} (team_evaluation_id 없음)")
                        continue

                    raw_score_val = data["raw_score"]
                    try:
                        # raw_score가 json 문자열일 경우 dict로 파싱, 아니면 그대로 사용
                        raw_score_dict = json.loads(raw_score_val)
                        display_raw_score = raw_score_dict.get('raw_hybrid_score', raw_score_val)
                    except (json.JSONDecodeError, TypeError):
                        display_raw_score = raw_score_val

                    query = text("""
                        UPDATE temp_evaluations 
                        SET ai_reason = :ai_reason,
                            raw_score = :raw_score,
                            score = :score,
                            comment = :comment
                        WHERE emp_no = :emp_no AND team_evaluation_id = :team_evaluation_id
                    """)
                    
                    result = connection.execute(query, {
                        "emp_no": data["emp_no"],
                        "team_evaluation_id": team_eval_result,
                        "ai_reason": data["ai_reason"],
                        "raw_score": raw_score_val,  # JSON 문자열 저장
                        "score": data["score"],  # 정규화된 점수 저장
                        "comment": data["comment"]
                    })
                    
                    if result.rowcount > 0:
                        success_count += 1
                        print(f"DB 업데이트 성공: {data['emp_no']} (원시: {display_raw_score}, 정규화: {data['score']})")
                    else:
                        failed_members.append(data["emp_no"])
                        print(f"DB 업데이트 실패: {data['emp_no']} (행 없음)")
                        
                except Exception as e:
                    failed_members.append(data["emp_no"])
                    print(f"DB 업데이트 실패: {data['emp_no']} - {e}")
            
            connection.commit()
            print(f"배치 업데이트 완료: 성공 {success_count}건, 실패 {len(failed_members)}건")
            
            return {
                "success_count": success_count,
                "failed_members": failed_members
            }
            
        except Exception as e:
            print(f"배치 업데이트 실패: {e}")
            connection.rollback()
            return {
                "success_count": 0,
                "failed_members": [data["emp_no"] for data in score_data]
            }

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def get_all_teams_with_data(period_id: int = 4) -> List[str]:
    """평가 데이터가 있는 모든 팀 ID 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT e.team_id
            FROM employees e
            JOIN final_evaluation_reports fer ON e.emp_no = fer.emp_no
            JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
            WHERE te.period_id = :period_id
            ORDER BY e.team_id
        """)
        results = connection.execute(query, {"period_id": period_id}).fetchall()
        return [row.team_id for row in results]