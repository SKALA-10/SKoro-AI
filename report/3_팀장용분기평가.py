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

print("✅ 팀 평가 리포트 생성기 - 기본 라이브러리 임포트 완료")

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

def clear_existing_team_reports(engine: Engine):
    """기존 team_evaluations.report 데이터 삭제"""
    try:
        query = text("UPDATE team_evaluations SET report = NULL WHERE report IS NOT NULL")
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 team evaluation report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def fetch_all_team_evaluation_ids(engine: Engine) -> List[int]:
    query = text("SELECT team_evaluation_id FROM team_evaluations ORDER BY team_evaluation_id;")
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()
    ids = [row[0] for row in results]
    print(f"✅ 총 {len(ids)}개의 팀 평가 리포트를 생성합니다. 대상 ID: {ids}")
    return ids

# --- ★★★ 수정된 함수 1 ★★★ ---
def fetch_team_evaluation_basic_data(engine: Engine, team_evaluation_id: int) -> Optional[Row]:
    """팀 평가의 기본 데이터를 조회합니다. (year_over_year_growth 포함)"""
    try:
        query = text("""
            SELECT 
                te.team_evaluation_id, te.team_id, t.team_name,
                te.period_id, p.period_name,
                te.average_achievement_rate, te.year_over_year_growth, -- 전분기 대비 성장률 추가
                te.ai_team_comparison, te.ai_team_overall_analysis_comment, 
                te.ai_collaboration_matrix, te.ai_team_coaching, 
                te.ai_risk, te.overall_comment,
                m.emp_name as manager_name
            FROM team_evaluations te
            JOIN teams t ON te.team_id = t.team_id
            JOIN periods p ON te.period_id = p.period_id
            LEFT JOIN employees m ON t.team_id = m.team_id AND m.role = 'MANAGER'
            WHERE te.team_evaluation_id = :team_evaluation_id;
        """)
        with engine.connect() as connection:
            result = connection.execute(query, {"team_evaluation_id": team_evaluation_id}).first()
        if result:
            print(f"   - Team Evaluation ID {team_evaluation_id} 데이터 조회 완료: {result.team_name} 팀")
        return result
    except Exception as e:
        print(f"   - ❌ 팀 평가 데이터 조회 중 오류 발생: {e}")
        return None

def fetch_team_kpis(engine: Engine, team_id: int) -> List[Row]:
    """팀 KPI 데이터를 조회합니다."""
    try:
        query = text("""
            SELECT kpi_name, ai_kpi_analysis_comment, ai_kpi_progress_rate
            FROM team_kpis WHERE team_id = :team_id ORDER BY ai_kpi_progress_rate DESC;
        """)
        with engine.connect() as connection:
            results = connection.execute(query, {"team_id": team_id}).fetchall()
        print(f"   - 팀 KPI 데이터 {len(results)}건 조회 완료")
        return results
    except Exception as e:
        print(f"   - ❌ 팀 KPI 데이터 조회 중 오류 발생: {e}")
        return []

# --- ★★★ 수정된 함수 2 ★★★ ---
def fetch_team_members_feedback(engine: Engine, team_id: int, period_id: int) -> List[Row]:
    """팀원들의 피드백 데이터를 조회합니다 (DB의 ranking 순서)."""
    try:
        # DB에 새로 추가된 ranking 컬럼을 조회
        query = text("""
            SELECT 
                fr.ranking, fr.ai_achievement_rate, e.emp_name,
                fr.ai_overall_contribution_summary_comment
            FROM feedback_reports fr
            JOIN employees e ON fr.emp_no = e.emp_no
            JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
            WHERE te.team_id = :team_id AND te.period_id = :period_id
            ORDER BY fr.ranking ASC, fr.ai_achievement_rate DESC;
        """)
        with engine.connect() as connection:
            results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        print(f"   - 팀원 피드백 데이터 {len(results)}건 조회 완료 (ranking 컬럼 기준 정렬)")
        return results
    except Exception as e:
        print(f"   - ❌ 팀원 피드백 데이터 조회 중 오류 발생: {e}")
        return []

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

# --- ★★★ 수정된 함수 3 ★★★ ---
def generate_korean_team_evaluation_report(
    팀평가기본데이터: Row, 
    팀kpi데이터: List[Row], 
    팀원피드백데이터: List[Row]
) -> Dict[str, Any]:
    """한국어 팀 평가 리포트 생성"""
    
    팀비교데이터 = safe_json_parse(팀평가기본데이터.ai_team_comparison)
    협업매트릭스데이터 = safe_json_parse(팀평가기본데이터.ai_collaboration_matrix)
    팀코칭데이터 = safe_json_parse(팀평가기본데이터.ai_team_coaching)
    리스크데이터 = safe_json_parse(팀평가기본데이터.ai_risk)
    
    한국어리포트 = {
        "기본_정보": {
            "팀명": 팀평가기본데이터.team_name or "",
            "팀장명": 팀평가기본데이터.manager_name or "미지정",
            "업무_수행_기간": 팀평가기본데이터.period_name or ""
        },
        "팀_종합_평가": {
            "평균_달성률": safe_convert_to_serializable(팀평가기본데이터.average_achievement_rate),
            "유사팀_평균": safe_convert_to_serializable(팀비교데이터.get("overall", {}).get("similar_avg_rate", 0)),
            "비교_분석": 팀비교데이터.get("overall", {}).get("comparison_result", ""),
            "팀_성과_분석_코멘트": 팀평가기본데이터.ai_team_overall_analysis_comment or "",
            # 전 분기 대비 성과 추이 추가
            "전_분기_대비_성과_추이": f"{팀평가기본데이터.year_over_year_growth}%" if 팀평가기본데이터.year_over_year_growth is not None else "N/A",
        },
        "팀_업무_목표_및_달성률": {
            "kpi_목록": [
                {
                    "팀_업무_목표": kpi.kpi_name or "", "kpi_분석_코멘트": kpi.ai_kpi_analysis_comment or "",
                    "달성률": safe_convert_to_serializable(kpi.ai_kpi_progress_rate),
                    "달성률_평균_전사유사팀": safe_convert_to_serializable(팀비교데이터.get("kpis", {}).get("similar_avg_rate", 0)),
                    "비교_분석": 팀비교데이터.get("kpis", {}).get("comparison_result", "")
                } for kpi in 팀kpi데이터
            ],
            "전사_유사팀_비교분석_코멘트": 팀비교데이터.get("overall", {}).get("comment", "")
        },
        "팀원_성과_분석": {
            # DB에서 가져온 ranking 값을 직접 사용
            "팀원별_기여도": [
                {
                    "순위": member.ranking or 'N/A', # DB ranking 사용
                    "달성률": safe_convert_to_serializable(member.ai_achievement_rate),
                    "이름": member.emp_name or "",
                    "기여_내용": member.ai_overall_contribution_summary_comment or ""
                } for member in 팀원피드백데이터 # enumerate 제거
            ],
            "종합_평가": "팀원들의 전반적인 성과 기여도와 달성률을 기준으로 한 종합 분석입니다.",
            "기여도_기준": "개인별 업무 달성률과 팀 목표 기여도를 종합하여 평가하였습니다."
        },
        "협업_네트워크": {
            "협업_매트릭스": 협업매트릭스데이터.get("collaboration_matrix", []),
            "팀_협업_요약": 협업매트릭스데이터.get("team_summary", ""),
            "협업률_설명": "개인이 수행한 전체 업무 중, 다른 팀원과 실제로 협업한 업무가 차지하는 비율입니다.",
            "협업_편중도_설명": "특정 동료에게만 협업이 쏠려있는지, 혹은 여러 동료와 고르게 협업하는지를 나타내는 지표입니다."
        },
        "팀원별_코칭_제안": {
            "일반_코칭": 팀코칭데이터.get("general_coaching", []),
            "집중_코칭": 팀코칭데이터.get("focused_coaching", [])
        },
        "리스크_및_향후_운영_제안": {
            "주요_리스크": 리스크데이터.get("risk_analysis", {}).get("major_risks", [])
        },
        "총평": {
            "주요_인사이트": 팀평가기본데이터.overall_comment or "",
        }
    }
    
    print(f"   - 🔍 한국어 팀 평가 리포트 생성 완료")
    return 한국어리포트

# --- 3. DB 저장 및 메인 실행 함수 ---
def save_team_json_report_to_db(engine: Engine, team_evaluation_id: int, json_report: Dict[str, Any]):
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)

    query = text("UPDATE team_evaluations SET report = :report_content WHERE team_evaluation_id = :team_evaluation_id;")
    
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                result = connection.execute(query, {"report_content": json_content, "team_evaluation_id": team_evaluation_id})
                if result.rowcount > 0:
                    transaction.commit()
                    print(f"   - ✅ Team Evaluation ID {team_evaluation_id}의 팀 평가 리포트가 team_evaluations.report에 성공적으로 저장되었습니다.")
                else:
                    transaction.rollback()
                    print(f"   - ⚠️ Team Evaluation ID {team_evaluation_id}에 해당하는 레코드를 찾을 수 없어 DB에 저장하지 못했습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_korean_team_report(report: dict) -> bool:
    required_keys = ["기본_정보", "팀_종합_평가", "팀_업무_목표_및_달성률", "팀원_성과_분석", "협업_네트워크", "팀원별_코칭_제안", "리스크_및_향후_운영_제안", "총평"]
    print(f"   - 🔍 팀 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    print(f"   - ✅ 팀 리포트 검증 성공!")
    return True

def main():
    """메인 실행 함수"""
    try:
        engine = get_db_engine()

        print(f"\n🗑️ 모든 기존 team evaluation report 데이터를 삭제합니다...")
        clear_existing_team_reports(engine)

        all_team_evaluation_ids = fetch_all_team_evaluation_ids(engine)
        if not all_team_evaluation_ids:
            print("처리할 팀 평가가 데이터베이스에 없습니다.")
            return

        success_count, error_count = 0, 0
        for team_evaluation_id in all_team_evaluation_ids:
            print(f"\n{'='*50}\n🚀 팀 평가 리포트 생성 시작 (ID: {team_evaluation_id})\n{'='*50}")
            try:
                팀평가기본데이터 = fetch_team_evaluation_basic_data(engine, team_evaluation_id)
                if not 팀평가기본데이터:
                    print(f"⚠️ Team Evaluation ID {team_evaluation_id}를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                    error_count += 1
                    continue

                팀kpi데이터 = fetch_team_kpis(engine, 팀평가기본데이터.team_id)
                팀원피드백데이터 = fetch_team_members_feedback(engine, 팀평가기본데이터.team_id, 팀평가기본데이터.period_id)
                한국어리포트 = generate_korean_team_evaluation_report(팀평가기본데이터, 팀kpi데이터, 팀원피드백데이터)

                if not validate_korean_team_report(한국어리포트):
                    print(f"   - ❌ 팀 리포트 데이터 검증 실패")
                    error_count += 1
                    continue

                save_team_json_report_to_db(engine, team_evaluation_id, 한국어리포트)
                success_count += 1

            except Exception as e:
                print(f"⚠️ Team Evaluation ID {team_evaluation_id} 처리 중 심각한 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
            time.sleep(0.5)

        print(f"\n🎉 팀 평가 리포트 생성 완료!")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        print(f"📊 총 처리: {len(all_team_evaluation_ids)}개")
    except ValueError as e:
        print(f"설정 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()