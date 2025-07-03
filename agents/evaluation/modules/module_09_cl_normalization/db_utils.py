# ai-performance-management-system/shared/tools/py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from typing import Dict, List, Optional, Any
from decimal import Decimal
import json

from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

def safe_decimal_to_float(value) -> float:
    """Decimal이나 다양한 타입을 안전하게 float로 변환"""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except:
        return 0.0

# ================================================================
# 데이터 조회 함수들
# ================================================================

def fetch_member_task_data(emp_no: str, period_id: int) -> List[Dict]:
    """직원의 업무 데이터 조회 (tasks + task_summaries)"""
    
    with engine.connect() as connection:
        query = text("""
            SELECT 
                t.task_id,
                t.task_name,
                t.task_detail,
                t.target_level,
                t.weight as task_weight,
                ts.ai_contribution_score,
                ts.ai_achievement_rate,
                ts.ai_assessed_grade,
                ts.ai_analysis_comment_task,
                ts.task_performance,
                tk.kpi_name,
                tk.kpi_description,
                tk.weight as kpi_weight
            FROM tasks t
            JOIN task_summaries ts ON t.task_id = ts.task_id
            JOIN team_kpis tk ON t.team_kpi_id = tk.team_kpi_id
            WHERE t.emp_no = :emp_no 
              AND ts.period_id = :period_id
            ORDER BY t.weight DESC, ts.ai_contribution_score DESC
        """)
        
        results = connection.execute(query, {
            "emp_no": emp_no,
            "period_id": period_id
        }).fetchall()
        
        tasks = []
        for row in results:
            task = row_to_dict(row)
            # 숫자 타입 변환
            task['task_weight'] = safe_decimal_to_float(task['task_weight'])
            task['ai_contribution_score'] = safe_decimal_to_float(task['ai_contribution_score'])
            task['ai_achievement_rate'] = safe_decimal_to_float(task['ai_achievement_rate'])
            task['kpi_weight'] = safe_decimal_to_float(task['kpi_weight'])
            tasks.append(task)
        
        return tasks

def fetch_member_peer_evaluation_data(emp_no: str, team_evaluation_id: int, period_id: int) -> Dict:
    """직원의 동료평가 AI 요약 데이터 조회"""
    
    with engine.connect() as connection:
        # 연말 평가 (period_id = 4)인 경우 final_evaluation_reports에서 조회
        if period_id == 4:
            query = text("""
                SELECT 
                    fer.ai_peer_talk_summary as peer_summary
                FROM final_evaluation_reports fer
                WHERE fer.emp_no = :emp_no 
                  AND fer.team_evaluation_id = :team_evaluation_id
            """)
        else:
            # 분기 평가 (period_id = 1,2,3)인 경우 feedback_reports에서 조회
            query = text("""
                SELECT 
                    fr.ai_peer_talk_summary as peer_summary
                FROM feedback_reports fr
                WHERE fr.emp_no = :emp_no 
                  AND fr.team_evaluation_id = :team_evaluation_id
            """)
        
        result = connection.execute(query, {
            "emp_no": emp_no,
            "team_evaluation_id": team_evaluation_id
        }).fetchone()
        
        # AI 요약 데이터 파싱
        peer_summary = {}
        
        if result and result.peer_summary:
            try:
                peer_summary = json.loads(result.peer_summary)
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 실패: {emp_no} 동료평가 요약")
                peer_summary = {}
        
        return {
            "peer_summary": peer_summary,
            "strengths": peer_summary.get("strengths", ""),
            "concerns": peer_summary.get("concerns", ""),
            "collaboration_observations": peer_summary.get("collaboration_observations", ""),
            "period_type": "annual" if period_id == 4 else "quarterly"
        }

def fetch_headquarter_cl_members_enhanced(headquarter_id: int, cl_group: str, period_id: int) -> List[Dict]:
    """본부 내 특정 CL 그룹의 모든 직원 데이터 조회 (업무+동료평가 포함)"""
    
    # CL 숫자 추출 (CL3 -> 3)
    cl_number = int(cl_group.replace('CL', ''))
    
    with engine.connect() as connection:
        query = text("""
            SELECT 
                e.emp_no, e.emp_name, e.cl, e.position, e.team_id,
                fer.score as module7_score,           -- 모듈7 정규화 점수
                te.score as baseline_score,           -- 모듈7 원본 점수
                te.manager_score,                     -- 팀장 수정 점수  
                COALESCE(te.reason, '') as captain_reason,          -- 팀장 수정 사유
                fer.ai_annual_achievement_rate as kpi_achievement,  -- KPI 달성률
                fer.final_evaluation_report_id,
                tea.team_evaluation_id                -- 동료평가 조회용
            FROM employees e
            JOIN teams t ON e.team_id = t.team_id
            JOIN headquarters h ON t.headquarter_id = h.headquarter_id
            JOIN temp_evaluations te ON e.emp_no = te.emp_no
            JOIN team_evaluations tea ON t.team_id = tea.team_id
            JOIN final_evaluation_reports fer ON (e.emp_no = fer.emp_no AND tea.team_evaluation_id = fer.team_evaluation_id)
            WHERE h.headquarter_id = :headquarter_id 
              AND e.cl = :cl_number
              AND tea.period_id = :period_id
              AND te.manager_score IS NOT NULL
              AND fer.score IS NOT NULL
            ORDER BY e.emp_no
        """)
        
        results = connection.execute(query, {
            "headquarter_id": headquarter_id,
            "cl_number": cl_number, 
            "period_id": period_id
        }).fetchall()
        
        members = []
        for row in results:
            member = row_to_dict(row)
            
            # 점수들을 float로 변환
            member['module7_score'] = safe_decimal_to_float(member['module7_score'])
            member['baseline_score'] = safe_decimal_to_float(member['baseline_score'])
            member['manager_score'] = safe_decimal_to_float(member['manager_score'])
            member['kpi_achievement'] = safe_decimal_to_float(member['kpi_achievement'])
            
            # 차이 계산
            member['score_diff'] = round(member['manager_score'] - member['baseline_score'], 2)
            
            # 업무 데이터 조회
            member['task_data'] = fetch_member_task_data(member['emp_no'], period_id)
            
            # 동료평가 데이터 조회
            member['peer_evaluation_data'] = fetch_member_peer_evaluation_data(
                member['emp_no'], member['team_evaluation_id'], period_id
            )
            
            members.append(member)
        
        return members

def get_all_cl_groups_in_headquarter(headquarter_id: int, period_id: int) -> List[str]:
    """본부 내 존재하는 모든 CL 그룹 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT e.cl
            FROM employees e
            JOIN teams t ON e.team_id = t.team_id
            JOIN headquarters h ON t.headquarter_id = h.headquarter_id
            JOIN temp_evaluations te ON e.emp_no = te.emp_no
            JOIN team_evaluations tea ON t.team_id = tea.team_id
            WHERE h.headquarter_id = :headquarter_id 
              AND tea.period_id = :period_id
              AND te.manager_score IS NOT NULL
            ORDER BY e.cl DESC
        """)
        
        results = connection.execute(query, {
            "headquarter_id": headquarter_id,
            "period_id": period_id
        }).fetchall()
        
        return [f"CL{row.cl}" for row in results]

# ================================================================
# DB 업데이트 함수
# ================================================================

def update_team_rankings(period_id: int) -> Dict:
    """팀별 독립 순위 업데이트 (같은 점수는 같은 순위: 1, 2, 2, 4 형태)"""
    
    success_count = 0
    failed_teams = []
    total_teams = 0
    
    with engine.connect() as connection:
        try:
            # 1. 해당 기간의 모든 팀 조회
            teams_query = text("""
                SELECT DISTINCT t.team_id, t.team_name
                FROM teams t
                JOIN team_evaluations tea ON t.team_id = tea.team_id
                JOIN final_evaluation_reports fer ON tea.team_evaluation_id = fer.team_evaluation_id
                WHERE tea.period_id = :period_id
                  AND fer.score IS NOT NULL
                ORDER BY t.team_id
            """)
            
            teams = connection.execute(teams_query, {"period_id": period_id}).fetchall()
            total_teams = len(teams)
            
            print(f"📊 팀별 순위 업데이트 시작: {total_teams}개 팀")
            
            for team in teams:
                team_id = team.team_id
                team_name = team.team_name
                
                try:
                    # 2. 각 팀별로 순위 계산 및 업데이트
                    ranking_query = text("""
                        UPDATE final_evaluation_reports fer
                        SET ranking = (
                            SELECT rank_num
                            FROM (
                                SELECT 
                                    emp_no,
                                    RANK() OVER (
                                        PARTITION BY t2.team_id 
                                        ORDER BY fer2.score DESC, fer2.emp_no ASC
                                    ) as rank_num
                                FROM final_evaluation_reports fer2
                                JOIN team_evaluations tea2 ON fer2.team_evaluation_id = tea2.team_evaluation_id
                                JOIN teams t2 ON tea2.team_id = t2.team_id
                                WHERE t2.team_id = :team_id 
                                  AND tea2.period_id = :period_id
                                  AND fer2.score IS NOT NULL
                            ) ranked
                            WHERE ranked.emp_no = fer.emp_no
                        )
                        WHERE fer.team_evaluation_id IN (
                            SELECT tea.team_evaluation_id 
                            FROM team_evaluations tea 
                            WHERE tea.team_id = :team_id 
                              AND tea.period_id = :period_id
                        )
                    """)
                    
                    result = connection.execute(ranking_query, {
                        "team_id": team_id,
                        "period_id": period_id
                    })
                    
                    # 3. 업데이트된 행 수 확인
                    updated_rows = result.rowcount
                    
                    if updated_rows > 0:
                        success_count += 1
                        print(f"   ✅ {team_name} (팀ID: {team_id}): {updated_rows}명 순위 업데이트")
                        
                        # 4. 순위 결과 확인 (디버깅용)
                        check_query = text("""
                            SELECT 
                                fer.emp_no,
                                fer.score,
                                fer.ranking,
                                e.emp_name
                            FROM final_evaluation_reports fer
                            JOIN team_evaluations tea ON fer.team_evaluation_id = tea.team_evaluation_id
                            JOIN employees e ON fer.emp_no = e.emp_no
                            WHERE tea.team_id = :team_id 
                              AND tea.period_id = :period_id
                              AND fer.score IS NOT NULL
                            ORDER BY fer.ranking ASC, fer.emp_no ASC
                        """)
                        
                        rankings = connection.execute(check_query, {
                            "team_id": team_id,
                            "period_id": period_id
                        }).fetchall()
                        
                        # 상위 3명만 출력
                        for i, rank in enumerate(rankings[:3]):
                            print(f"     {rank.ranking}위: {rank.emp_no} ({rank.emp_name}) - {rank.score:.2f}점")
                        
                    else:
                        failed_teams.append(team_name)
                        print(f"   ⚠️ {team_name} (팀ID: {team_id}): 순위 업데이트 실패 (업데이트된 행 없음)")
                        
                except Exception as e:
                    failed_teams.append(team_name)
                    print(f"   ❌ {team_name} (팀ID: {team_id}): 순위 업데이트 실패 - {str(e)}")
            
            connection.commit()
            
            success_rate = (success_count / total_teams * 100) if total_teams > 0 else 0
            print(f"💾 팀별 순위 업데이트 완료: {success_count}/{total_teams}개 팀 성공 ({success_rate:.1f}%)")
            
            if failed_teams:
                print(f"   ❌ 실패한 팀: {', '.join(failed_teams)}")
            
            return {
                "success_count": success_count,
                "failed_count": len(failed_teams),
                "total_teams": total_teams,
                "success_rate": round(success_rate, 1),
                "failed_teams": failed_teams
            }
            
        except Exception as e:
            print(f"❌ 팀별 순위 업데이트 실패: {str(e)}")
            connection.rollback()
            return {
                "success_count": 0,
                "failed_count": total_teams,
                "total_teams": total_teams,
                "success_rate": 0.0,
                "failed_teams": [team.team_name for team in teams] if 'teams' in locals() else [],
                "error": str(e)
            }

def batch_update_final_evaluation_reports(adjustments: List[Dict], period_id: int) -> Dict:
    """CL 그룹의 조정 결과를 final_evaluation_reports에 실제 업데이트"""
    
    success_count = 0
    failed_members = []
    
    with engine.connect() as connection:
        try:
            for adj in adjustments:
                try:
                    query = text("""
                        UPDATE final_evaluation_reports fer
                        SET score = :score,
                            cl_reason = :cl_reason
                        WHERE fer.final_evaluation_report_id = :report_id
                    """)
                    
                    # 향상된 사유 생성
                    validity_analysis = adj.get("validity_analysis", {})
                    detailed_reason = f"{adj['reason']} | 타당성분석: 업무증거 {validity_analysis.get('task_evidence', 0):.2f}, 동료평가 {validity_analysis.get('peer_consistency', 0):.2f}"
                    
                    result = connection.execute(query, {
                        "report_id": adj.get("final_evaluation_report_id"),
                        "score": adj["final_score"],
                        "cl_reason": detailed_reason
                    })
                    
                    if result.rowcount > 0:
                        success_count += 1
                        print(f"   ✅ {adj['emp_no']}: {adj['original_score']:.2f} → {adj['final_score']:.2f} ({adj['change_amount']:+.2f}) | {validity_analysis.get('validity_grade', 'N/A')}")
                    else:
                        failed_members.append(adj["emp_no"])
                        print(f"   ❌ {adj['emp_no']}: 업데이트 실패 (행 없음)")
                        
                except Exception as e:
                    failed_members.append(adj["emp_no"])
                    print(f"   ❌ {adj['emp_no']}: 업데이트 실패 - {str(e)}")
            
            connection.commit()
            print(f"💾 DB 업데이트 완료: 성공 {success_count}건, 실패 {len(failed_members)}건")
            
            return {
                "success_count": success_count,
                "failed_count": len(failed_members),
                "failed_members": failed_members
            }
            
        except Exception as e:
            print(f"❌ 배치 업데이트 실패: {str(e)}")
            connection.rollback()
            return {
                "success_count": 0,
                "failed_count": len(adjustments),
                "failed_members": [adj["emp_no"] for adj in adjustments]
            }
        

def get_all_headquarters_info() -> List[Dict]:
    """모든 본부 정보 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT h.headquarter_id, h.headquarter_name
            FROM headquarters h
            JOIN teams t ON h.headquarter_id = t.headquarter_id
            JOIN employees e ON t.team_id = e.team_id
            JOIN temp_evaluations te ON e.emp_no = te.emp_no
            WHERE te.manager_score IS NOT NULL
            ORDER BY h.headquarter_id
        """)
        results = connection.execute(query).fetchall()
        return [row_to_dict(row) for row in results]