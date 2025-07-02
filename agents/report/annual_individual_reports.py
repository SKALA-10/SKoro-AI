import os
import json
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import ProgrammingError

from config.settings import DatabaseConfig

print("✅ 연말 개인 최종 평가 리포트 생성기 - 기본 라이브러리 임포트 완료")

# --- 1. 데이터베이스 연동 함수 ---
def get_db_engine() -> Engine:
    db_config = DatabaseConfig()
    engine = create_engine(db_config.DATABASE_URL, pool_pre_ping=True)
    print("✅ 데이터베이스 엔진 생성 완료")
    return engine

# --- 2. 데이터 조회 함수 ---
def fetch_final_evaluation_emp_nos(engine: Engine, period_id: Optional[int] = None, teams: Optional[list] = None) -> List[str]:
    """
    final_evaluation_reports 테이블에서 직원 번호를 조회합니다. (팀장 제외)
    period_id와 teams가 주어지면 해당 조건에 맞는 직원들만 조회합니다.
    """
    try:
        if teams and period_id:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                SELECT DISTINCT fer.emp_no 
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                JOIN employees e ON fer.emp_no = e.emp_no
                WHERE te.period_id = :period_id 
                AND e.team_id IN ({placeholders})
                AND e.role != 'MANAGER'  # 팀장 제외
                ORDER BY fer.emp_no;
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"✅ 팀 {teams}, 분기 {period_id}의 연말 평가 대상자 조회")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                SELECT DISTINCT fer.emp_no 
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                JOIN employees e ON fer.emp_no = e.emp_no
                WHERE e.team_id IN ({placeholders})
                AND e.role != 'MANAGER'  # 팀장 제외
                ORDER BY fer.emp_no;
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"✅ 팀 {teams}의 연말 평가 대상자 조회")
        elif period_id:
            query = text("""
                SELECT DISTINCT fer.emp_no 
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                JOIN employees e ON fer.emp_no = e.emp_no
                WHERE te.period_id = :period_id
                AND e.role != 'MANAGER'  # 팀장 제외
                ORDER BY fer.emp_no;
            """)
            params = {'period_id': period_id}
            print(f"✅ 분기 {period_id}의 연말 평가 대상자 조회")
        else:
            query = text("""
                SELECT DISTINCT fer.emp_no 
                FROM final_evaluation_reports fer
                JOIN employees e ON fer.emp_no = e.emp_no
                WHERE e.role != 'MANAGER'  # 팀장 제외
                ORDER BY fer.emp_no;
            """)
            params = {}
            print(f"✅ 모든 연말 평가 대상자 조회")
        
        with engine.connect() as connection:
            results = connection.execute(query, params).fetchall()
        emp_nos = [row[0] for row in results]
        print(f"✅ 총 {len(emp_nos)}명의 연말 평가 리포트를 생성합니다. 대상: {emp_nos}")
        return emp_nos
    except Exception as e:
        print(f"❌ 연말 평가 대상자 조회 중 오류 발생: {e}")
        return []

def fetch_final_evaluation_data(engine: Engine, emp_no: str) -> Optional[Row]:
    query = text("""
        SELECT fer.*, e.emp_name, e.cl, t.team_name, p.period_name
        FROM final_evaluation_reports fer
        JOIN employees e ON fer.emp_no = e.emp_no
        JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
        JOIN teams t ON te.team_id = t.team_id
        JOIN periods p ON te.period_id = p.period_id
        WHERE fer.emp_no = :emp_no
        LIMIT 1;
    """)
    with engine.connect() as connection:
        result = connection.execute(query, {"emp_no": emp_no}).first()
    if result:
        print(f"   - {emp_no}님 연말 평가 데이터 조회 완료: {result.emp_name}")
    return result

def fetch_temp_evaluation_data(engine: Engine, emp_no: str) -> Optional[Row]:
    query = text("""
        SELECT comment, raw_score
        FROM temp_evaluations
        WHERE emp_no = :emp_no
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"emp_no": emp_no}).first()
    print(f"   - {emp_no}님의 팀장 평가 데이터 조회 완료")
    return result

def fetch_quarterly_performance(engine: Engine, emp_no: str) -> List[Dict]:
    base_quarters = [
        {"분기": "1분기", "순위": 0, "달성률": 0, "실적_요약": ""},
        {"분기": "2분기", "순위": 0, "달성률": 0, "실적_요약": ""},
        {"분기": "3분기", "순위": 0, "달성률": 0, "실적_요약": ""},
        {"분기": "4분기", "순위": 0, "달성률": 0, "실적_요약": ""}
    ]
    
    # 1-3분기 데이터는 feedback_reports에서 가져오기
    query = text("""
        SELECT fr.ai_achievement_rate as achievement_rate,
               fr.ai_overall_contribution_summary_comment as performance_summary,
               fr.ranking,
               p.period_name, p.order_in_year
        FROM feedback_reports fr
        JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
        JOIN periods p ON te.period_id = p.period_id
        WHERE fr.emp_no = :emp_no 
        ORDER BY p.order_in_year
    """)
    with engine.connect() as conn:
        feedback_results = conn.execute(query, {"emp_no": emp_no}).fetchall()
    for result in feedback_results:
        period_name = result.period_name or ""
        quarter_index = None
        if "1분기" in period_name:
            quarter_index = 0
        elif "2분기" in period_name:
            quarter_index = 1
        elif "3분기" in period_name:
            quarter_index = 2
        if quarter_index is not None:
            base_quarters[quarter_index] = {
                "분기": f"{quarter_index + 1}분기",
                "순위": result.ranking or 0,
                "달성률": result.achievement_rate or 0,
                "실적_요약": result.performance_summary or ""
            }
    
    # 4분기 데이터는 final_evaluation_reports에서 가져오기
    final_query = text("""
        SELECT ai_annual_achievement_rate as achievement_rate,
               ai_annual_performance_summary_comment as performance_summary,
               ranking
        FROM final_evaluation_reports
        WHERE emp_no = :emp_no
        LIMIT 1
    """)
    with engine.connect() as conn:
        final_result = conn.execute(final_query, {"emp_no": emp_no}).first()
        if final_result:
            base_quarters[3] = {
                "분기": "4분기",
                "순위": final_result.ranking or 0,
                "달성률": final_result.achievement_rate or 0,
                "실적_요약": final_result.performance_summary or ""
            }
    
    print(f"   - {emp_no}님의 분기별 성과 데이터 처리 완료")
    return base_quarters

def fetch_tasks_for_final_report(engine: Engine, emp_no: str, team_evaluation_id: int) -> List[Row]:
    # 연말 평가용 Task 데이터만 가져오기 (최종 상태)
    query = text("""
        SELECT DISTINCT tk.task_name, ts.task_performance, ts.ai_achievement_rate, ts.ai_analysis_comment_task
        FROM tasks tk
        JOIN task_summaries ts ON tk.task_id = ts.task_id
        WHERE tk.emp_no = :emp_no 
        AND tk.team_kpi_id IS NOT NULL
        AND ts.ai_achievement_rate IS NOT NULL
        ORDER BY tk.task_name, ts.ai_achievement_rate DESC
    """)
    with engine.connect() as connection:
        results = connection.execute(query, {"emp_no": emp_no}).fetchall()
    
    # Task별로 최고 달성률 데이터만 선택 (연말 최종 상태)
    task_dict = {}
    for result in results:
        task_name = result.task_name
        if task_name not in task_dict or (result.ai_achievement_rate or 0) > (task_dict[task_name].ai_achievement_rate or 0):
            task_dict[task_name] = result
    
    final_results = list(task_dict.values())
    print(f"   - {emp_no}님의 연말 Task 데이터 {len(final_results)}건 조회 완료")
    return final_results

# --- 3. JSON 리포트 생성 함수 ---
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal): return float(obj)
        return super(DecimalEncoder, self).default(obj)

def safe_json_parse(json_str: str) -> Dict[str, Any]:
    try:
        if json_str is None: return {}
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}

def safe_convert_to_serializable(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict): return {k: safe_convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple): return tuple(safe_convert_to_serializable(item) for item in obj)
    if obj is None: return ""
    return obj

def generate_final_individual_report(
    final_data: Row,
    temp_eval: Optional[Row],
    분기별_업무: List[Dict],
    업무표: List[Row]
) -> Dict[str, Any]:
    # 기본 정보
    cl_레벨 = str(final_data.cl).strip() if final_data.cl else ""
    if cl_레벨 and cl_레벨.isdigit():
        cl_레벨 = f"CL{cl_레벨}"
    elif cl_레벨 and not cl_레벨.startswith("CL"):
        cl_레벨 = f"CL{cl_레벨}"

    # 최종 평가 점수 및 raw_score
    raw_score = safe_json_parse(temp_eval.raw_score) if temp_eval and getattr(temp_eval, 'raw_score', None) else {}
    fourp = safe_json_parse(final_data.ai_4p_evaluation)
    peer_talk = safe_json_parse(final_data.ai_peer_talk_summary)
    growth = safe_json_parse(final_data.ai_growth_coaching)

    # 업무표
    업무표_json = [
        {
            "Task명": t.task_name or "",
            "핵심_Task": t.task_performance or "",
            "누적_달성률(%)": safe_convert_to_serializable(t.ai_achievement_rate),
            "분석_코멘트": t.ai_analysis_comment_task or ""
        }
        for t in 업무표
    ]

    report = {
        "기본_정보": {
            "성명": final_data.emp_name or "",
            "직위": cl_레벨,
            "소속": final_data.team_name or "",
            "업무_수행_기간": final_data.period_name or ""
        },
        "최종_평가": {
            "최종_점수": safe_convert_to_serializable(final_data.score),
            "업적": safe_convert_to_serializable(raw_score.get("achievement_score")),
            "SK_Values": {
                "Passionate": safe_convert_to_serializable(raw_score.get("passionate_score")),
                "Proactive": safe_convert_to_serializable(raw_score.get("proactive_score")),
                "Professional": safe_convert_to_serializable(raw_score.get("professional_score")),
                "People": safe_convert_to_serializable(raw_score.get("people_score"))
            },
            "성과_요약": final_data.ai_annual_performance_summary_comment or ""
        },
        "분기별_업무_기여도": 분기별_업무,
        "팀_업무_목표_및_개인_달성률": {
            "업무표": 업무표_json,
            "개인_종합_달성률": safe_convert_to_serializable(final_data.ai_annual_achievement_rate),
            "종합_기여_코멘트": final_data.ai_annual_performance_summary_comment or ""
        },
        "Peer_Talk": {
            "강점": peer_talk.get('strengths', ""),
            "우려": peer_talk.get('concerns', ""),
            "협업_관찰": peer_talk.get('collaboration_observations', "")
        },
        "성장_제안_및_개선_피드백": {
            "성장_포인트": growth.get('growth_points', []),
            "보완_영역": growth.get('improvement_areas', []),
            "추천_활동": growth.get('recommended_activities', [])
        },
        "팀장_Comment": temp_eval.comment if temp_eval else "",
        "종합_Comment": final_data.overall_comment or ""
    }
    print(f"   - 🔍 연말 개인 최종 평가 리포트 생성 완료")
    return report

# --- 4. DB 저장 및 메인 실행 함수 ---
def save_final_json_report_to_db(engine: Engine, emp_no: str, json_report: Dict[str, Any]):
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)

    query = text("UPDATE final_evaluation_reports SET report = :report_content WHERE emp_no = :emp_no;")
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                result = connection.execute(query, {"report_content": json_content, "emp_no": emp_no})
                if result.rowcount > 0:
                    transaction.commit()
                    print(f"   - ✅ {emp_no}님의 연말 최종 평가 리포트가 final_evaluation_reports.report에 성공적으로 저장되었습니다.")
                else:
                    transaction.rollback()
                    print(f"   - ⚠️ {emp_no}님에 해당하는 레코드를 찾을 수 없어 DB에 저장하지 못했습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_final_individual_report(report: dict) -> bool:
    required_keys = ["기본_정보", "최종_평가", "분기별_업무_기여도", "팀_업무_목표_및_개인_달성률", "Peer_Talk", "성장_제안_및_개선_피드백", "팀장_Comment", "종합_Comment"]
    print(f"   - 🔍 연말 개인 최종 평가 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    print(f"   - ✅ 연말 개인 최종 평가 리포트 검증 성공!")
    return True

def clear_existing_final_reports(engine: Engine, period_id: Optional[int] = None, teams: Optional[list] = None):
    """
    기존 final_evaluation_reports.report 데이터를 NULL로 업데이트하여 삭제합니다.
    period_id와 teams가 주어지면 해당 조건에 맞는 데이터만 삭제합니다.
    """
    try:
        if teams and period_id:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE final_evaluation_reports 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND emp_no IN (
                    SELECT e.emp_no FROM employees e WHERE e.team_id IN ({placeholders})
                )
                AND team_evaluation_id IN (
                    SELECT te.team_evaluation_id FROM team_evaluations te WHERE te.period_id = :period_id
                )
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"🗑️ 팀 {teams}, 분기 {period_id}의 기존 final_evaluation_reports.report 데이터를 삭제합니다...")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE final_evaluation_reports 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND emp_no IN (
                    SELECT e.emp_no FROM employees e WHERE e.team_id IN ({placeholders})
                )
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"🗑️ 팀 {teams}의 기존 final_evaluation_reports.report 데이터를 삭제합니다...")
        elif period_id:
            query = text("""
                UPDATE final_evaluation_reports 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND team_evaluation_id IN (
                    SELECT te.team_evaluation_id FROM team_evaluations te WHERE te.period_id = :period_id
                )
            """)
            params = {'period_id': period_id}
            print(f"🗑️ 분기 {period_id}의 기존 final_evaluation_reports.report 데이터를 삭제합니다...")
        else:
            query = text("UPDATE final_evaluation_reports SET report = NULL WHERE report IS NOT NULL")
            params = {}
            print(f"🗑️ 모든 기존 final_evaluation_reports.report 데이터를 삭제합니다...")
        
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query, params)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 final_evaluation_reports.report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def main(emp_no: Optional[str] = None, period_id: Optional[int] = None, teams: Optional[list] = None):
    """
    메인 실행 함수: 연말 개인 최종 평가 리포트를 생성하고 DB에 저장합니다.
    
    Args:
        emp_no: 특정 직원 번호. None이면 모든 직원을 처리합니다.
        period_id: 특정 분기 ID. None이면 모든 분기를 처리합니다.
        teams: 특정 팀 ID 리스트. None이면 모든 팀을 처리합니다.
    """
    try:
        engine = get_db_engine()
        
        if emp_no is None:
            # 모든 직원 처리
            print(f"\n🗑️ 기존 final_evaluation_reports.report 데이터를 삭제합니다...")
            clear_existing_final_reports(engine, period_id, teams)
            
            all_emp_nos = fetch_final_evaluation_emp_nos(engine, period_id, teams)
            if not all_emp_nos:
                print("처리할 연말 평가 대상자가 없습니다.")
                return
                
            success_count, error_count = 0, 0
            for current_emp_no in all_emp_nos:
                print(f"\n{'='*50}\n🚀 연말 개인 최종 평가 리포트 생성 시작 ({current_emp_no})\n{'='*50}")
                try:
                    final_data = fetch_final_evaluation_data(engine, current_emp_no)
                    if not final_data:
                        print(f"⚠️ {current_emp_no}님의 연말 평가 데이터를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                        error_count += 1
                        continue
                    temp_eval = fetch_temp_evaluation_data(engine, current_emp_no)
                    분기별_업무 = fetch_quarterly_performance(engine, current_emp_no)
                    업무표 = fetch_tasks_for_final_report(engine, current_emp_no, final_data.team_evaluation_id)
                    report = generate_final_individual_report(final_data, temp_eval, 분기별_업무, 업무표)
                    if not validate_final_individual_report(report):
                        print(f"   - ❌ 연말 개인 최종 평가 리포트 데이터 검증 실패")
                        error_count += 1
                        continue
                    save_final_json_report_to_db(engine, current_emp_no, report)
                    success_count += 1
                except Exception as e:
                    print(f"⚠️ {current_emp_no}님 처리 중 심각한 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                time.sleep(0.5)
            print(f"\n🎉 연말 개인 최종 평가 리포트 생성 완료!")
            print(f"✅ 성공: {success_count}개")
            print(f"❌ 실패: {error_count}개")
            print(f"📊 총 처리: {len(all_emp_nos)}개")
        else:
            # 특정 직원만 처리
            print(f"\n🎯 특정 직원 {emp_no}님 처리 시작")
            if period_id:
                print(f"📅 대상 분기: {period_id}")
            print(f"{'='*50}")
            try:
                final_data = fetch_final_evaluation_data(engine, emp_no)
                if not final_data:
                    print(f"❌ {emp_no}님의 연말 평가 데이터를 조회할 수 없습니다.")
                    return
                temp_eval = fetch_temp_evaluation_data(engine, emp_no)
                분기별_업무 = fetch_quarterly_performance(engine, emp_no)
                업무표 = fetch_tasks_for_final_report(engine, emp_no, final_data.team_evaluation_id)
                report = generate_final_individual_report(final_data, temp_eval, 분기별_업무, 업무표)
                if not validate_final_individual_report(report):
                    print(f"❌ 연말 개인 최종 평가 리포트 데이터 검증 실패")
                    return
                save_final_json_report_to_db(engine, emp_no, report)
                print(f"\n✅ {emp_no}님 처리 완료!")
            except Exception as e:
                print(f"❌ {emp_no}님 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 