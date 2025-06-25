import os
import json
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.row import Row
from dotenv import load_dotenv

print("✅ 최종 팀 평가 리포트 생성기 - 기본 라이브러리 임포트 완료")

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
                print(f"✅ 기존 team evaluation final report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def fetch_all_team_evaluation_ids(engine: Engine) -> List[int]:
    """모든 팀 평가 ID 조회"""
    query = text("SELECT team_evaluation_id FROM team_evaluations ORDER BY team_evaluation_id;")
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()
    ids = [row[0] for row in results]
    print(f"✅ 총 {len(ids)}개의 최종 팀 평가 리포트를 생성합니다. 대상 ID: {ids}")
    return ids

# --- 2. 데이터 조회 함수 ---

def fetch_team_evaluation_data(engine: Engine, team_evaluation_id: int) -> dict:
    """리포트 생성에 필요한 모든 데이터를 DB에서 직접 조회"""
    
    team_info_query = text("""
        SELECT 
            te.team_evaluation_id, t.team_id, t.team_name, p.period_id, p.period_name,
            te.average_achievement_rate, te.year_over_year_growth,
            te.ai_team_comparison, te.ai_team_overall_analysis_comment,
            te.ai_risk, te.ai_plan, te.overall_comment,
            m.emp_name as manager_name
        FROM team_evaluations te
        JOIN teams t ON te.team_id = t.team_id
        JOIN periods p ON te.period_id = p.period_id
        LEFT JOIN employees m ON t.team_id = m.team_id AND m.role = 'MANAGER'
        WHERE te.team_evaluation_id = :id
    """)
    with engine.connect() as conn:
        team_info = conn.execute(team_info_query, {"id": team_evaluation_id}).first()
    if not team_info:
        raise ValueError(f"Team Evaluation ID {team_evaluation_id}를 찾을 수 없습니다.")

    kpi_query = text("SELECT kpi_name, ai_kpi_analysis_comment, ai_kpi_progress_rate FROM team_kpis WHERE team_id = :team_id")
    with engine.connect() as conn:
        kpis = conn.execute(kpi_query, {"team_id": team_info.team_id}).fetchall()

    summary_query = text("""
        SELECT te.raw_score, fer.ai_annual_performance_summary_comment
        FROM temp_evaluations te
        LEFT JOIN final_evaluation_reports fer ON te.emp_no = fer.emp_no AND fer.team_evaluation_id = te.team_evaluation_id
        WHERE te.team_evaluation_id = :team_eval_id
    """)
    with engine.connect() as conn:
        summaries = conn.execute(summary_query, {"team_eval_id": team_evaluation_id}).fetchall()

    print(f"   - 팀 ID {team_info.team_id}의 모든 데이터 조회 완료")
    return {"team_info": team_info, "kpis": kpis, "summaries": summaries}


# --- 3. JSON 처리 및 리포트 생성 함수 ---

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal): return float(obj)
        return super(DecimalEncoder, self).default(obj)
def safe_convert_to_serializable(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict): return {key: safe_convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list): return [safe_convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple): return tuple(safe_convert_to_serializable(item) for item in obj)
    if obj is None: return ""
    return obj
def safe_json_parse(json_str: str, default_value: Any = None) -> Any:
    if default_value is None: default_value = {}
    try:
        return json.loads(json_str) if isinstance(json_str, str) and json_str else default_value
    except (json.JSONDecodeError, TypeError): return default_value

def generate_team_evaluation_report(data: dict) -> dict:
    """DB 데이터를 요구사항에 맞는 개조식 구조로 변환하여 리포트 생성"""
    
    team_info = data["team_info"]
    kpis = data["kpis"]
    summaries = data["summaries"]
    
    team_comparison = safe_json_parse(team_info.ai_team_comparison, {})
    risk_data = safe_json_parse(team_info.ai_risk, {})
    plan_data = safe_json_parse(team_info.ai_plan, {})
    
    # --- 각 섹션별 데이터 구조화 ---
    기본정보 = {"팀명": team_info.team_name or "","팀장명": team_info.manager_name or "미지정","업무 수행 기간": team_info.period_name or ""}
    
    overall_comp = team_comparison.get("overall", {})
    팀종합평가 = {
        "평균 달성률": safe_convert_to_serializable(team_info.average_achievement_rate),
        "유사팀 평균": safe_convert_to_serializable(overall_comp.get("similar_avg_rate", 0)),
        "비교 분석": overall_comp.get("comparison_result", ""),
        "팀 성과 분석 코멘트": team_info.ai_team_overall_analysis_comment or "",
        "전 분기 대비 성과 추이": f"{team_info.year_over_year_growth}%" if team_info.year_over_year_growth is not None else "N/A"
    }
    
    kpi_comp = team_comparison.get("kpis", {})
    팀업무목표및달성률 = {"업무목표표": [{"팀 업무 목표": kpi.kpi_name or "", "kpi 분석 코멘트": kpi.ai_kpi_analysis_comment or "","달성률": safe_convert_to_serializable(kpi.ai_kpi_progress_rate),
                     "달성률 평균 (전사 유사팀)": safe_convert_to_serializable(kpi_comp.get("similar_avg_rate", 0)),"비교 분석": str(kpi_comp.get("comparison_result", ""))} for kpi in kpis],
                      "전사 유사팀 비교분석 코멘트": overall_comp.get("comment", "")}

    summary_comments = " | ".join([s.ai_annual_performance_summary_comment for s in summaries if s.ai_annual_performance_summary_comment])
    raw_score_sample = safe_json_parse(summaries[0].raw_score, {}) if summaries else {}
    팀성과요약 = {"업적 (팀 목표 기여도)": raw_score_sample.get("achievement_score", "N/A"),
                "SK Values (4P)": {"Passionate": raw_score_sample.get("passionate_score", "N/A"),"Proactive": raw_score_sample.get("proactive_score", "N/A"),
                                 "Professional": raw_score_sample.get("professional_score", "N/A"),"People": raw_score_sample.get("people_score", "N/A")},
                "성과 요약": summary_comments}

    # --- ★★★ 수정된 부분 1: 리스크 요인 개조식 구조화 ★★★ ---
    risk_analysis = risk_data.get("risk_analysis", {})
    major_risks_formatted = []
    for risk in risk_analysis.get("major_risks", []):
        if isinstance(risk, dict):
            major_risks_formatted.append({
                "주요리스크": risk.get("risk_name", ""),
                "리스크 심각도": risk.get("severity", ""),
                "리스크 설명": risk.get("description", ""),
                "발생 원인": risk.get("causes", []),
                "영향 예측": risk.get("impacts", []),
                "운영 개선 전략 제안": risk.get("strategies", [])
            })
    팀조직력및리스크요인 = {"주요 리스크 목록": major_risks_formatted}

    # --- ★★★ 수정된 부분 2: 다음 연도 운영 제안 개조식 구조화 ★★★ ---
    annual_plans = plan_data.get("annual_plans", [{}])[0]
    다음연도운영제안 = {
        "핵심 인력 운용 전략": annual_plans.get("personnel_strategies", []),
        "협업 구조 개선 방향": annual_plans.get("collaboration_improvements", [])
    }
    
    # --- ★★★ 수정된 부분 3: 총평 구조화 ★★★ ---
    총평 = {"종합 의견": team_info.overall_comment or "작성된 총평이 없습니다."}

    # 최종 리포트
    final_report = {
        "기본 정보": 기본정보, "팀 종합 평가": 팀종합평가, "팀 업무 목표 및 달성률": 팀업무목표및달성률,
        "팀 성과 요약": 팀성과요약, "팀 조직력 및 리스크 요인": 팀조직력및리스크요인,
        "다음 연도 운영 제안": 다음연도운영제안, "총평": 총평
    }
    
    print(f"   - 최종 팀 평가 리포트 생성 완료")
    return final_report

# --- 4. DB 저장 및 메인 실행 함수 ---
def save_team_json_report_to_db(engine: Engine, team_evaluation_id: int, json_report: dict):
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)
    
    query = text("UPDATE team_evaluations SET report = :report WHERE team_evaluation_id = :id")
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(query, {"report": json_content, "id": team_evaluation_id})
                transaction.commit()
                print(f"   - ✅ Team Eval ID {team_evaluation_id}의 최종 팀 평가 리포트가 DB에 저장되었습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_team_report(report: dict) -> bool:
    required_keys = ["기본 정보", "팀 종합 평가", "팀 업무 목표 및 달성률", 
                    "팀 성과 요약", "팀 조직력 및 리스크 요인", 
                    "다음 연도 운영 제안", "총평"]
    
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
        print("\n🗑️ 기존 최종 팀 평가 리포트 데이터 삭제 중...")
        clear_existing_team_reports(engine)
        
        all_ids = fetch_all_team_evaluation_ids(engine)
        if not all_ids:
            print("처리할 팀 평가 데이터가 없습니다.")
            return

        success_count, error_count = 0, 0
        for team_eval_id in all_ids:
            print(f"\n{'='*60}\n🚀 최종 팀 평가 리포트 생성 시작 (ID: {team_eval_id})\n{'='*60}")
            try:
                team_data = fetch_team_evaluation_data(engine, team_eval_id)
                final_report = generate_team_evaluation_report(team_data)
                
                if not validate_team_report(final_report):
                    error_count += 1
                    continue
                
                save_team_json_report_to_db(engine, team_eval_id, final_report)
                success_count += 1
            except Exception as e:
                print(f"⚠️ ID {team_eval_id} 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
            time.sleep(0.5)
            
        print(f"\n🎉 최종 팀 평가 리포트 생성 완료!")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        print(f"📊 총 처리: {len(all_ids)}개")
    except Exception as e:
        print(f"메인 함수 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()