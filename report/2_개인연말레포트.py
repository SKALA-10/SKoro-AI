import os
import json
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import ProgrammingError
from dotenv import load_dotenv

print("✅ 기본 라이브러리 임포트 완료")

# --- 1. 데이터베이스 연동 함수 ---

def get_db_engine() -> Engine:
    load_dotenv()
    db_user = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError("데이터베이스 연결 정보가 .env 파일에 올바르게 설정되지 않았습니다.")
    dsn = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
    engine = create_engine(dsn)
    print("✅ 데이터베이스 엔진 생성 완료")
    return engine

def clear_existing_final_reports(engine: Engine):
    """기존 final_evaluation_reports.report 데이터 삭제"""
    try:
        query = text("UPDATE final_evaluation_reports SET report = NULL WHERE report IS NOT NULL")
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 final evaluation report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def fetch_all_final_evaluation_ids(engine: Engine) -> List[int]:
    query = text("SELECT final_evaluation_report_id FROM final_evaluation_reports ORDER BY final_evaluation_report_id;")
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()
    ids = [row[0] for row in results]
    print(f"✅ 총 {len(ids)}개의 최종 평가 리포트를 생성합니다. 대상 ID: {ids}")
    return ids

# --- ★★★ 수정된 함수 ★★★ ---
def fetch_final_evaluation_data(engine: Engine, final_evaluation_report_id: int) -> Optional[Row]:
    """최종 평가 리포트의 기본 데이터를 조회합니다."""
    try:
        # 문제가 된 CASE 구문을 제거하고, 확정된 'overall_comment' 컬럼만 직접 조회합니다.
        query = text("""
            SELECT
                fer.final_evaluation_report_id, e.emp_no, e.emp_name, e.cl, e.position,
                t.team_name, p.period_id, p.period_name,
                fer.score, fer.ranking, fer.ai_annual_achievement_rate,
                fer.ai_annual_performance_summary_comment,
                fer.ai_peer_talk_summary, fer.ai_growth_coaching,
                fer.overall_comment
            FROM final_evaluation_reports fer
            JOIN employees e ON fer.emp_no = e.emp_no
            JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
            JOIN teams t ON te.team_id = t.team_id
            JOIN periods p ON te.period_id = p.period_id
            WHERE fer.final_evaluation_report_id = :report_id;
        """)
        with engine.connect() as connection:
            result = connection.execute(query, {"report_id": final_evaluation_report_id}).first()
        if result:
            print(f"   - Final Report ID {final_evaluation_report_id} 데이터 조회 완료: {result.emp_name}님")
        return result
    except Exception as e:
        print(f"   - ❌ 최종 평가 데이터 조회 중 오류 발생: {e}")
        return None

def fetch_quarterly_performance(engine: Engine, final_eval_data: Row) -> List[Dict]:
    """분기별 성과(feedback_reports)와 최종 성과(final_evaluation_reports)를 통합하여 연간 성과 리스트를 생성합니다."""
    quarterly_data = []
    emp_no = final_eval_data.emp_no

    query_with_ranking = text("""
        SELECT fr.ranking, fr.ai_annual_achievement_rate as achievement_rate,
               fr.ai_overall_contribution_summary_comment as performance_summary, p.period_name
        FROM feedback_reports fr
        JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
        JOIN periods p ON te.period_id = p.period_id
        WHERE fr.emp_no = :emp_no ORDER BY p.order_in_year
    """)
    query_without_ranking = text("""
        SELECT fr.ai_annual_achievement_rate as achievement_rate,
               fr.ai_overall_contribution_summary_comment as performance_summary, p.period_name
        FROM feedback_reports fr
        JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
        JOIN periods p ON te.period_id = p.period_id
        WHERE fr.emp_no = :emp_no ORDER BY p.order_in_year
    """)

    try:
        with engine.connect() as conn:
            try:
                feedback_results = conn.execute(query_with_ranking, {"emp_no": emp_no}).fetchall()
                print(f"   - {emp_no}님의 분기별 성과(feedback_reports) 조회 성공 ('ranking' 컬럼 사용)")
            except ProgrammingError:
                print(f"   - ⚠️ 'feedback_reports.ranking' 컬럼이 없어 대체 쿼리를 실행합니다.")
                feedback_results = conn.execute(query_without_ranking, {"emp_no": emp_no}).fetchall()

        for idx, result in enumerate(feedback_results, 1):
            rank = result.ranking if hasattr(result, 'ranking') and result.ranking is not None else idx
            quarterly_data.append({
                "분기": result.period_name or f"{idx}분기",
                "순위": rank,
                "달성률": result.achievement_rate or 0,
                "실적 요약": result.performance_summary or ""
            })
    except Exception as e:
        print(f"   - ❌ 분기별 성과(feedback_reports) 조회 중 오류 발생: {e}. 분기 데이터는 비어있게 됩니다.")

    quarterly_data.append({
        "분기": "연간 최종",
        "순위": final_eval_data.ranking or "N/A",
        "달성률": final_eval_data.ai_annual_achievement_rate or 0,
        "실적 요약": final_eval_data.ai_annual_performance_summary_comment or ""
    })

    print(f"   - {emp_no}님의 분기+최종 성과 데이터 {len(quarterly_data)}건 처리 완료")
    return quarterly_data

def fetch_tasks_for_final_report(engine: Engine, emp_no: str, period_id: int) -> List[Row]:
    query = text("""
        SELECT
            tk.task_name, ts.task_performance, ts.ai_achievement_rate,
            ts.ai_analysis_comment_task
        FROM tasks tk
        JOIN task_summaries ts ON tk.task_id = ts.task_id
        WHERE tk.emp_no = :emp_no AND ts.period_id = :period_id;
    """)
    with engine.connect() as connection:
        results = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchall()
    print(f"   - {emp_no}님의 Task 데이터 {len(results)}건 조회 완료")
    return results

def fetch_temp_evaluation_data(engine: Engine, emp_no: str) -> Optional[Row]:
    try:
        query = text("""
            SELECT raw_score, reason
            FROM temp_evaluations
            WHERE emp_no = :emp_no
            LIMIT 1
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"emp_no": emp_no}).first()
        print(f"   - {emp_no}님의 임시 평가 데이터(temp_evaluations) 조회 완료")
        return result
    except Exception as e:
        print(f"   - 임시 평가 데이터 조회 실패: {e}")
        return None

# --- 2. JSON 리포트 생성 함수 ---
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal): return float(obj)
        return super(DecimalEncoder, self).default(obj)

def safe_json_parse(json_str: str) -> Dict[str, Any]:
    try: return json.loads(json_str) if json_str else {}
    except (json.JSONDecodeError, TypeError): return {}

def safe_convert_to_serializable(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict): return {k: safe_convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple): return tuple(safe_convert_to_serializable(item) for item in obj)
    if obj is None: return ""
    return obj

def generate_korean_final_evaluation_report(최종평가데이터: Row, 분기별성과데이터: List[Dict], 업무데이터: List[Row], 임시평가데이터: Optional[Row]) -> Dict[str, Any]:
    peer_talk_데이터 = safe_json_parse(최종평가데이터.ai_peer_talk_summary)
    growth_데이터 = safe_json_parse(최종평가데이터.ai_growth_coaching)

    cl_레벨 = str(최종평가데이터.cl).strip() if 최종평가데이터.cl else ""
    if cl_레벨 and cl_레벨.isdigit(): cl_레벨 = f"CL{cl_레벨}"
    elif cl_레벨 and not cl_레벨.startswith("CL"): cl_레벨 = f"CL{cl_레벨}"

    raw_score_데이터 = {}
    if 임시평가데이터 and 임시평가데이터.raw_score:
        try: raw_score_데이터 = json.loads(임시평가데이터.raw_score) if isinstance(임시평가데이터.raw_score, str) else {}
        except: raw_score_데이터 = {}

    업무표 = [{"Task명": t.task_name or "", "핵심 Task": t.task_performance or "",
              "누적 달성률 (%)": safe_convert_to_serializable(t.ai_achievement_rate),
              "분석 코멘트": t.ai_analysis_comment_task or ""} for t in 업무데이터]

    한국어리포트 = {
        "기본 정보": {
            "성명": 최종평가데이터.emp_name or "", "직위": cl_레벨,
            "소속": 최종평가데이터.team_name or "", "업무 수행 기간": 최종평가데이터.period_name or ""
        },
        "최종 평가": {
            "최종 평가 점수": safe_convert_to_serializable(최종평가데이터.score),
            "점수 구성표": {
                "업적 (팀 목표 기여도)": safe_convert_to_serializable(raw_score_데이터.get("achievement_score", 0)),
                "SK Values (4P)": {"Passionate": safe_convert_to_serializable(raw_score_데이터.get("passionate_score", 0)),
                                   "Proactive": safe_convert_to_serializable(raw_score_데이터.get("proactive_score", 0)),
                                   "Professional": safe_convert_to_serializable(raw_score_데이터.get("professional_score", 0)),
                                   "People": safe_convert_to_serializable(raw_score_데이터.get("people_score", 0))}
            },
            "성과 요약": 최종평가데이터.ai_annual_performance_summary_comment or ""
        },
        "분기별 업무 목표 기여도": {"분기별표": 분기별성과데이터},
        "팀 업무 목표 및 개인 달성률": {
            "업무표": 업무표,
            "개인 종합 달성률": safe_convert_to_serializable(최종평가데이터.ai_annual_achievement_rate),
            "종합 기여 코멘트": 최종평가데이터.ai_annual_performance_summary_comment or ""
        },
        "Peer Talk": {
            "강점": peer_talk_데이터.get('strengths', []), "우려": peer_talk_데이터.get('concerns', []),
            "협업 관찰": peer_talk_데이터.get('collaboration_observations', "")
        },
        "성장 제안 및 개선 피드백": {
            "성장 포인트": growth_데이터.get('growth_points', []), "보완 영역": growth_데이터.get('improvement_areas', []),
            "추천 활동": growth_데이터.get('recommended_activities', [])
        },
        "팀장 Comment": 임시평가데이터.reason if 임시평가데이터 and 임시평가데이터.reason else "",
        "종합 Comment": 최종평가데이터.overall_comment or ""
    }

    print(f"   - 🔍 한국어 최종 평가 리포트 생성 완료")
    return 한국어리포트

# --- 3. DB 저장 및 메인 실행 함수 ---
def save_final_json_report_to_db(engine: Engine, final_evaluation_report_id: int, json_report: Dict[str, Any]):
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)

    query = text("UPDATE final_evaluation_reports SET report = :report_content WHERE final_evaluation_report_id = :report_id;")
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(query, {"report_content": json_content, "report_id": final_evaluation_report_id})
                transaction.commit()
                print(f"   - ✅ Final Report ID {final_evaluation_report_id}의 최종 평가 리포트가 DB에 성공적으로 저장되었습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()

def validate_korean_final_report(report: dict) -> bool:
    required_keys = ["기본 정보", "최종 평가", "분기별 업무 목표 기여도",
                    "팀 업무 목표 및 개인 달성률", "Peer Talk", "성장 제안 및 개선 피드백",
                    "팀장 Comment", "종합 Comment"]

    print(f"   - 🔍 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    print(f"   - ✅ 리포트 검증 성공!")
    return True

def main():
    try:
        engine = get_db_engine()

        print(f"\n🗑️ 모든 기존 final evaluation report 데이터를 삭제합니다...")
        clear_existing_final_reports(engine)

        all_report_ids = fetch_all_final_evaluation_ids(engine)
        if not all_report_ids:
            print("처리할 최종 평가 리포트가 데이터베이스에 없습니다.")
            return

        success_count, error_count = 0, 0
        for report_id in all_report_ids:
            print(f"\n{'='*50}\n🚀 최종 평가 리포트 생성 시작 (ID: {report_id})\n{'='*50}")
            try:
                최종평가데이터 = fetch_final_evaluation_data(engine, report_id)
                if not 최종평가데이터:
                    print(f"⚠️ Final Report ID {report_id}를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                    error_count += 1
                    continue

                분기별성과데이터 = fetch_quarterly_performance(engine, 최종평가데이터)

                업무데이터 = fetch_tasks_for_final_report(engine, 최종평가데이터.emp_no, 최종평가데이터.period_id)
                임시평가데이터 = fetch_temp_evaluation_data(engine, 최종평가데이터.emp_no)

                한국어리포트 = generate_korean_final_evaluation_report(최종평가데이터, 분기별성과데이터, 업무데이터, 임시평가데이터)

                if not validate_korean_final_report(한국어리포트):
                    print(f"   - ❌ 리포트 데이터 검증 실패")
                    error_count += 1
                    continue

                save_final_json_report_to_db(engine, report_id, 한국어리포트)
                success_count += 1

            except Exception as e:
                print(f"⚠️ ID {report_id} 처리 중 심각한 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue

            time.sleep(0.5)

        print(f"\n🎉 최종 평가 리포트 생성 완료!")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        print(f"📊 총 처리: {len(all_report_ids)}개")

    except ValueError as e:
        print(f"설정 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()