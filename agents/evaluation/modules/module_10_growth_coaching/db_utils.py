# ================================================================
# db_utils_module10.py - 모듈 10 데이터베이스 관련 유틸리티
# ================================================================

import sys
import os
import json
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from dotenv import load_dotenv

load_dotenv()

# DB 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)
from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

# ================================================================
# 데이터 수집 함수들
# ================================================================

def fetch_basic_info(emp_no: str) -> Dict:
    """기본 정보 조회"""
    with engine.connect() as connection:
        try:
            query = text("""
                SELECT emp_no, emp_name, cl, position, team_id
                FROM employees WHERE emp_no = :emp_no
            """)
            result = connection.execute(query, {"emp_no": emp_no}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"기본 정보 조회 실패: {e}")
            return {}

def fetch_performance_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """성과 데이터 수집 (모듈 2 결과)"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                query = text("""
                    SELECT fr.contribution_rate, fr.ai_overall_contribution_summary_comment,
                           AVG(ts.ai_achievement_rate) as ai_achievement_rate,
                           AVG(ts.ai_contribution_score) as avg_contribution_score
                    FROM feedback_reports fr
                    JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                    LEFT JOIN (
                        SELECT ts.*, t.emp_no 
                        FROM task_summaries ts 
                        JOIN tasks t ON ts.task_id = t.task_id
                        WHERE ts.period_id = :period_id
                    ) ts ON ts.emp_no = fr.emp_no
                    WHERE fr.emp_no = :emp_no AND te.period_id = :period_id
                    GROUP BY fr.emp_no, fr.contribution_rate, fr.ai_overall_contribution_summary_comment
                """)
            else:  # annual
                query = text("""
                    SELECT fer.contribution_rate, fer.ai_annual_achievement_rate as ai_achievement_rate,
                           fer.ai_annual_performance_summary_comment, fer.score
                    FROM final_evaluation_reports fer
                    JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                    WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
                """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"성과 데이터 조회 실패: {e}")
            return {}

def fetch_peer_talk_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """Peer Talk 데이터 수집 (모듈 4 결과)"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                table = "feedback_reports"
            else:
                table = "final_evaluation_reports"
                
            query = text(f"""
                SELECT fer.ai_peer_talk_summary
                FROM {table} fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            
            if result and result.ai_peer_talk_summary:
                try:
                    return json.loads(result.ai_peer_talk_summary)
                except json.JSONDecodeError:
                    print(f"Peer Talk JSON 파싱 실패: {emp_no}")
                    return {}
            return {}
        except Exception as e:
            print(f"Peer Talk 데이터 조회 실패: {e}")
            return {}

def fetch_fourp_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """4P 데이터 수집 (모듈 6 결과)"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                table = "feedback_reports"
            else:
                table = "final_evaluation_reports"
                
            query = text(f"""
                SELECT fer.ai_4p_evaluation
                FROM {table} fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            
            if result and result.ai_4p_evaluation:
                try:
                    return json.loads(result.ai_4p_evaluation)
                except json.JSONDecodeError:
                    print(f"4P JSON 파싱 실패: {emp_no}")
                    return {}
            return {}
        except Exception as e:
            print(f"4P 데이터 조회 실패: {e}")
            return {}

def fetch_collaboration_data(emp_no: str, period_id: int) -> Dict:
    """협업 데이터 수집 (모듈 3 결과에서 개인 부분 추출)"""
    with engine.connect() as connection:
        try:
            # 직원의 team_id 조회
            team_query = text("SELECT team_id FROM employees WHERE emp_no = :emp_no")
            team_result = connection.execute(team_query, {"emp_no": emp_no}).fetchone()
            
            if not team_result:
                return {}
                
            team_id = team_result.team_id
            
            # team_evaluations에서 협업 매트릭스 조회
            collab_query = text("""
                SELECT ai_collaboration_matrix
                FROM team_evaluations
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            collab_result = connection.execute(collab_query, {
                "team_id": team_id, 
                "period_id": period_id
            }).fetchone()
            
            if collab_result and collab_result.ai_collaboration_matrix:
                try:
                    collaboration_matrix = json.loads(collab_result.ai_collaboration_matrix)
                    
                    # collaboration_matrix에서 해당 emp_no 찾기
                    for member in collaboration_matrix.get("collaboration_matrix", []):
                        if member.get("emp_no") == emp_no:
                            return {
                                "collaboration_rate": member.get("collaboration_rate", 0),
                                "team_role": member.get("team_role", ""),
                                "key_collaborators": member.get("key_collaborators", []),
                                "collaboration_bias": member.get("collaboration_bias", "보통"),
                                "overall_evaluation": member.get("overall_evaluation", "")
                            }
                except json.JSONDecodeError:
                    print(f"협업 매트릭스 JSON 파싱 실패: {emp_no}")
                    return {}
            
            return {}
        except Exception as e:
            print(f"협업 데이터 조회 실패: {e}")
            return {}

def fetch_module7_score_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """모듈 7 팀 내 정규화 점수 데이터 수집 (연말만)"""
    if report_type != "annual":
        return {}
        
    with engine.connect() as connection:
        try:
            # temp_evaluations에서 팀 내 정규화 점수 조회
            query = text("""
                SELECT raw_score, score, ai_reason
                FROM temp_evaluations
                WHERE TempEvaluation_empNo = :emp_no
            """)
            
            result = connection.execute(query, {"emp_no": emp_no}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"모듈 7 점수 데이터 조회 실패: {e}")
            return {}

def fetch_module9_final_score_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """모듈 9 부문 정규화 최종 점수 데이터 수집 (연말만)"""
    if report_type != "annual":
        return {}
        
    with engine.connect() as connection:
        try:
            # final_evaluation_reports에서 최종 점수
            query = text("""
                SELECT fer.score, fer.ranking, fer.cl_reason
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"모듈 9 최종 점수 데이터 조회 실패: {e}")
            return {}

def calculate_ranking_by_achievement(emp_no: str, team_id: str, period_id: int, report_type: str) -> int:
    """팀 내 달성률 기반 순위를 동적으로 계산"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                # 분기 랭킹: task_summaries의 ai_achievement_rate 평균 기준
                query = text("""
                    WITH team_achievements AS (
                        SELECT
                            e.emp_no,
                            COALESCE(AVG(ts.ai_achievement_rate), 0) as achievement_rate
                        FROM employees e
                        LEFT JOIN (
                            SELECT t.emp_no, ts.ai_achievement_rate
                            FROM tasks t
                            JOIN task_summaries ts ON t.task_id = ts.task_id
                            WHERE ts.period_id = :period_id
                        ) ts ON e.emp_no = ts.emp_no
                        WHERE e.team_id = :team_id
                        GROUP BY e.emp_no
                    )
                    SELECT
                        emp_no,
                        RANK() OVER (ORDER BY achievement_rate DESC) as ranking
                    FROM team_achievements
                """)
            else:  # annual
                # 연간 랭킹: final_evaluation_reports의 ai_annual_achievement_rate 기준
                query = text("""
                    WITH team_achievements AS (
                        SELECT
                            e.emp_no,
                            COALESCE(fer.ai_annual_achievement_rate, 0) as achievement_rate
                        FROM employees e
                        LEFT JOIN (
                            SELECT fer_inner.emp_no, fer_inner.ai_annual_achievement_rate
                            FROM final_evaluation_reports fer_inner
                            JOIN team_evaluations te ON fer_inner.team_evaluation_id = te.team_evaluation_id
                            WHERE te.period_id = :period_id
                        ) fer ON e.emp_no = fer.emp_no
                        WHERE e.team_id = :team_id
                    )
                    SELECT
                        emp_no,
                        RANK() OVER (ORDER BY achievement_rate DESC) as ranking
                    FROM team_achievements
                """)

            params = {"team_id": team_id, "period_id": period_id}
            rank_list = connection.execute(query, params).fetchall()

            for row in rank_list:
                if row[0] == emp_no:
                    return int(row[1]) if row[1] is not None else 0

            return 0

        except Exception as e:
            print(f"달성률 기반 순위 계산 실패: {e}")
            return 0

# ================================================================
# DB 저장 함수들
# ================================================================

def save_individual_result(emp_no: str, period_id: int, report_type: str, 
                         individual_result: Dict, overall_comment: str) -> bool:
    """개인용 결과 + 종합 총평 저장"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                # feedback_reports 테이블에 저장
                query = text("""
                    UPDATE feedback_reports 
                    SET ai_growth_coaching = :result,
                        overall_comment = :overall_comment
                    WHERE emp_no = :emp_no 
                    AND team_evaluation_id = (
                        SELECT team_evaluation_id 
                        FROM team_evaluations 
                        WHERE period_id = :period_id 
                        AND team_id = (SELECT team_id FROM employees WHERE emp_no = :emp_no)
                    )
                """)
            else:
                # final_evaluation_reports 테이블에 저장
                query = text("""
                    UPDATE final_evaluation_reports 
                    SET ai_growth_coaching = :result,
                        overall_comment = :overall_comment
                    WHERE emp_no = :emp_no 
                    AND team_evaluation_id = (
                        SELECT team_evaluation_id 
                        FROM team_evaluations 
                        WHERE period_id = :period_id 
                        AND team_id = (SELECT team_id FROM employees WHERE emp_no = :emp_no)
                    )
                """)
            
            result = connection.execute(query, {
                "emp_no": emp_no,
                "period_id": period_id,
                "result": json.dumps(individual_result, ensure_ascii=False),
                "overall_comment": overall_comment
            })
            
            connection.commit()
            return result.rowcount > 0
            
        except Exception as e:
            print(f"개인용 결과 저장 실패: {e}")
            connection.rollback()
            return False

def save_manager_result(emp_no: str, period_id: int, manager_result: Dict) -> bool:
    """팀장용 결과 저장 (team_evaluations.ai_team_coaching에 누적)"""
    with engine.connect() as connection:
        try:
            # 기존 team_coaching 데이터 조회
            team_id_query = text("SELECT team_id FROM employees WHERE emp_no = :emp_no")
            team_result = connection.execute(team_id_query, {"emp_no": emp_no}).fetchone()
            
            if not team_result:
                return False
                
            team_id = team_result.team_id
            
            # 기존 ai_team_coaching 데이터 조회
            existing_query = text("""
                SELECT ai_team_coaching 
                FROM team_evaluations 
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            existing_result = connection.execute(existing_query, {
                "team_id": team_id,
                "period_id": period_id
            }).fetchone()
            
            if existing_result and existing_result.ai_team_coaching:
                # 기존 데이터가 있으면 누적
                try:
                    existing_data = json.loads(existing_result.ai_team_coaching)
                except json.JSONDecodeError:
                    existing_data = {"general_coaching": [], "focused_coaching": []}
            else:
                # 기존 데이터가 없으면 새로 생성
                existing_data = {"general_coaching": [], "focused_coaching": []}
            
            # 현재 직원 데이터 추가/업데이트
            # general_coaching에서 기존 직원 데이터 제거
            existing_data["general_coaching"] = [
                gc for gc in existing_data["general_coaching"] 
                if gc.get("emp_no") != emp_no
            ]
            # focused_coaching에서도 기존 직원 데이터 제거
            existing_data["focused_coaching"] = [
                fc for fc in existing_data["focused_coaching"] 
                if fc.get("emp_no") != emp_no
            ]
            
            # 새 데이터 추가
            existing_data["general_coaching"].extend(manager_result["general_coaching"])
            existing_data["focused_coaching"].extend(manager_result["focused_coaching"])
            
            # DB 업데이트
            update_query = text("""
                UPDATE team_evaluations 
                SET ai_team_coaching = :result
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            result = connection.execute(update_query, {
                "team_id": team_id,
                "period_id": period_id,
                "result": json.dumps(existing_data, ensure_ascii=False)
            })
            
            connection.commit()
            return result.rowcount > 0
            
        except Exception as e:
            print(f"팀장용 결과 저장 실패: {e}")
            connection.rollback()
            return False

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def get_teams_with_data(period_id: int = 4) -> List[str]:
    """데이터가 있는 팀 목록 조회"""
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

# ================================================================
# 데이터 정리 함수들
# ================================================================

def clean_ai_team_coaching_data(team_id: str, period_id: int):
    """기존 ai_team_coaching 데이터에서 빈 emp_no 항목들을 제거"""
    with engine.connect() as connection:
        try:
            # 기존 데이터 조회
            query = text("""
                SELECT ai_team_coaching 
                FROM team_evaluations 
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            result = connection.execute(query, {
                "team_id": team_id,
                "period_id": period_id
            }).fetchone()
            
            if not result or not result.ai_team_coaching:
                print(f"정리할 데이터가 없습니다: {team_id}")
                return
            
            try:
                data = json.loads(result.ai_team_coaching)
            except json.JSONDecodeError:
                print(f"JSON 파싱 실패: {team_id}")
                return
            
            # 빈 emp_no 항목들 제거
            original_general_count = len(data.get("general_coaching", []))
            original_focused_count = len(data.get("focused_coaching", []))
            
            data["general_coaching"] = [
                item for item in data.get("general_coaching", [])
                if item.get("emp_no") and item.get("emp_no").strip()
            ]
            
            data["focused_coaching"] = [
                item for item in data.get("focused_coaching", [])
                if item.get("emp_no") and item.get("emp_no").strip()
            ]
            
            cleaned_general_count = len(data.get("general_coaching", []))
            cleaned_focused_count = len(data.get("focused_coaching", []))
            
            # 업데이트
            update_query = text("""
                UPDATE team_evaluations 
                SET ai_team_coaching = :result
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            connection.execute(update_query, {
                "team_id": team_id,
                "period_id": period_id,
                "result": json.dumps(data, ensure_ascii=False)
            })
            
            connection.commit()
            
            print(f"✅ 데이터 정리 완료: {team_id}")
            print(f"   general_coaching: {original_general_count} → {cleaned_general_count}")
            print(f"   focused_coaching: {original_focused_count} → {cleaned_focused_count}")
            
        except Exception as e:
            print(f"❌ 데이터 정리 실패: {e}")
            connection.rollback()

def clean_all_team_coaching_data(period_id: int = 4):
    """모든 팀의 ai_team_coaching 데이터 정리"""
    teams = get_teams_with_data(period_id)
    
    print(f"🧹 {len(teams)}개 팀의 데이터 정리 시작...")
    
    for team_id in teams:
        clean_ai_team_coaching_data(team_id, period_id)
    
    print("✅ 모든 팀 데이터 정리 완료!")