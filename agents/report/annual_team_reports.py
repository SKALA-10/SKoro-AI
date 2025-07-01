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

print("✅ 최종 팀 평가 리포트 생성기 (Period 4 전용) - 기본 라이브러리 임포트 완료")

# --- 1. 데이터베이스 연동 함수 ---

def get_db_engine() -> Engine:
    """
    config.settings의 DatabaseConfig를 사용하여 SQLAlchemy 엔진을 생성합니다.
    """
    db_config = DatabaseConfig()
    engine = create_engine(db_config.DATABASE_URL, pool_pre_ping=True)
    print("✅ 데이터베이스 엔진 생성 완료")
    return engine

def clear_existing_period4_team_reports(engine: Engine):
    """기존 Period 4의 team_evaluations.report 데이터만 삭제"""
    try:
        query = text("""
            UPDATE team_evaluations 
            SET report = NULL 
            WHERE report IS NOT NULL 
            AND period_id = 4
        """)
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 Period 4 team evaluation final report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def fetch_period4_team_evaluation_ids(engine: Engine) -> List[int]:
    """Period 4의 팀 평가 ID만 조회"""
    query = text("""
        SELECT team_evaluation_id 
        FROM team_evaluations 
        WHERE period_id = 4
        ORDER BY team_evaluation_id;
    """)
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()
    ids = [row[0] for row in results]
    print(f"✅ Period 4 전용: 총 {len(ids)}개의 최종 팀 평가 리포트를 생성합니다. 대상 ID: {ids}")
    return ids

# --- 2. 데이터 조회 함수 ---

def fetch_team_evaluation_data(engine: Engine, team_evaluation_id: int) -> dict:
    """리포트 생성에 필요한 모든 데이터를 DB에서 직접 조회 (Period 4 검증 포함)"""
    
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
        WHERE te.team_evaluation_id = :id AND te.period_id = 4
    """)
    with engine.connect() as conn:
        team_info = conn.execute(team_info_query, {"id": team_evaluation_id}).first()
    
    if not team_info:
        raise ValueError(f"Team Evaluation ID {team_evaluation_id}를 찾을 수 없거나 Period 4가 아닙니다.")

    # Period 4 검증
    if team_info.period_id != 4:
        raise ValueError(f"Team Evaluation ID {team_evaluation_id}는 Period {team_info.period_id}입니다. Period 4만 처리 가능합니다.")

    # KPI 데이터 조회
    kpi_query = text("SELECT kpi_name, ai_kpi_analysis_comment, ai_kpi_progress_rate FROM team_kpis WHERE team_id = :team_id")
    with engine.connect() as conn:
        kpis = conn.execute(kpi_query, {"team_id": team_info.team_id}).fetchall()

    # 팀원 성과 요약 데이터 조회
    summary_query = text("""
        SELECT 
            e.emp_name,
            fer.ranking,
            te.raw_score,
            fer.score as final_score,
            fer.contribution_rate,
            fer.ai_annual_performance_summary_comment
        FROM temp_evaluations te
        JOIN employees e ON te.emp_no = e.emp_no
        LEFT JOIN final_evaluation_reports fer ON te.emp_no = fer.emp_no AND fer.team_evaluation_id = te.team_evaluation_id
        WHERE te.team_evaluation_id = :team_eval_id
        ORDER BY fer.ranking ASC, e.emp_name ASC
    """)
    with engine.connect() as conn:
        summaries = conn.execute(summary_query, {"team_eval_id": team_evaluation_id}).fetchall()

    print(f"   - Period 4 팀 ID {team_info.team_id}의 모든 데이터 조회 완료")
    return {"team_info": team_info, "kpis": kpis, "summaries": summaries}


# --- 3. JSON 처리 및 리포트 생성 함수 ---

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal): 
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def safe_convert_to_serializable(obj):
    if isinstance(obj, Decimal): 
        return float(obj)
    if isinstance(obj, dict): 
        return {key: safe_convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list): 
        return [safe_convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple): 
        return tuple(safe_convert_to_serializable(item) for item in obj)
    if obj is None: 
        return ""
    return obj

def safe_json_parse(json_str: str, default_value: Any = None) -> Any:
    if default_value is None: 
        default_value = {}
    try:
        return json.loads(json_str) if isinstance(json_str, str) and json_str else default_value
    except (json.JSONDecodeError, TypeError): 
        return default_value

def generate_team_evaluation_report(data: dict) -> dict:
    """DB 데이터를 요구사항에 맞는 개조식 구조로 변환하여 리포트 생성"""
    
    team_info = data["team_info"]
    kpis = data["kpis"]
    summaries = data["summaries"]
    
    team_comparison = safe_json_parse(team_info.ai_team_comparison, {})
    risk_data = safe_json_parse(team_info.ai_risk, {})
    plan_data = safe_json_parse(team_info.ai_plan, {})
    
    # --- 1. 기본 정보 ---
    기본정보 = {
        "팀명": team_info.team_name or "",
        "팀장명": team_info.manager_name or "미지정",
        "업무 수행 기간": team_info.period_name or "",
        "평가 구분": "연간 최종 평가 (Period 4)"
    }
    
    # --- 2. 팀 종합 평가 ---
    overall_comp = team_comparison.get("overall", {})
    팀종합평가 = {
        "평균 달성률": safe_convert_to_serializable(team_info.average_achievement_rate),
        "유사팀 평균": safe_convert_to_serializable(overall_comp.get("similar_avg_rate", 0)),
        "비교 분석": overall_comp.get("comparison_result", ""),
        "팀 성과 분석 코멘트": team_info.ai_team_overall_analysis_comment or ""
    }
    
    # --- 3. 팀 업무 목표 및 달성률 ---
    kpis_comparison_data_list = team_comparison.get("kpis", []) 
    업무목표표 = []
    for kpi in kpis:
        matched_kpi_comp_item = next(
            (item for item in kpis_comparison_data_list if item.get("kpi_name") == kpi.kpi_name),
            {}
        )
        
        업무목표표.append({
            "팀 업무 목표": kpi.kpi_name or "",
            "kpi 분석 코멘트": kpi.ai_kpi_analysis_comment or "",
            "달성률": safe_convert_to_serializable(kpi.ai_kpi_progress_rate),
            "달성률 평균 (전사 유사팀)": safe_convert_to_serializable(matched_kpi_comp_item.get("similar_avg_rate", None)),
            "비교 분석": str(matched_kpi_comp_item.get("comparison_result", ""))
        })
    
    팀업무목표및달성률 = {
        "업무목표표": 업무목표표,
        "전사 유사팀 비교분석 코멘트": overall_comp.get("comment", "")
    }

    # --- 4. 팀 성과 요약 ---
    # 팀원별 성과 표 생성
    팀원별성과표 = []
    
    for s in summaries:
        raw_score_data = safe_json_parse(s.raw_score, {})
        
        # 팀원별 성과 표 데이터
        팀원별성과표.append({
            "순위": s.ranking or "N/A",
            "이름": s.emp_name or "",
            "업적": safe_convert_to_serializable(raw_score_data.get("achievement_score", "N/A")),
            "SK Values (4P)": {
                "Passionate": safe_convert_to_serializable(raw_score_data.get("passionate_score", "N/A")),
                "Proactive": safe_convert_to_serializable(raw_score_data.get("proactive_score", "N/A")),
                "Professional": safe_convert_to_serializable(raw_score_data.get("professional_score", "N/A")),
                "People": safe_convert_to_serializable(raw_score_data.get("people_score", "N/A"))
            },
            "최종 점수": safe_convert_to_serializable(s.final_score),
            "기여도": safe_convert_to_serializable(s.contribution_rate),
            "성과요약": s.ai_annual_performance_summary_comment or "성과 요약이 없습니다."
        })
    
    팀성과요약 = {
        "팀원별_성과_표": 팀원별성과표,
        "평가_기준_해석_및_유의사항": "업적 점수는 팀 목표 대비 개인 기여도를 반영하며, SK Values (4P)는 Passionate(열정), Proactive(주도성), Professional(전문성), People(협업성)을 평가합니다. 최종 점수는 업적과 4P 점수에 CL 정규화가 적용된 값이며, 기여도는 팀 목표 달성을 위한 상대적인 기여 정도를 기준으로 합니다. 순위는 최종 점수 기준으로 정렬됩니다."
    }

    # --- 5. 팀 조직력 및 리스크 요인 ---
    risk_analysis = risk_data.get("risk_analysis", {})
    major_risks_formatted = []
    
    for risk in risk_analysis.get("major_risks", []):
        if isinstance(risk, dict):
            impacts = risk.get("impacts", [])
            impact_details = []
            for impact in impacts:
                if isinstance(impact, dict):
                    impact_detail = {
                        "영향 설명": impact.get("description", "")
                    }
                    impact_details.append(impact_detail)
                elif isinstance(impact, str):
                    impact_details.append({"영향 설명": impact})
            
            strategies = risk.get("strategies", [])
            strategy_details = []
            for strategy in strategies:
                if isinstance(strategy, dict):
                    strategy_details.append(strategy.get("description", ""))
                elif isinstance(strategy, str):
                    strategy_details.append(strategy)
            
            major_risks_formatted.append({
                "주요리스크": risk.get("risk_name", ""),
                "리스크 심각도": risk.get("severity", ""),
                "리스크 설명": risk.get("description", ""),
                "발생 원인": risk.get("causes", []),
                "영향 예측": impact_details,
                "운영 개선 전략 제안": strategy_details
            })
    
    팀조직력및리스크요인 = {
        "주요 리스크 목록": major_risks_formatted
    }

    # --- 6. 다음 연도 운영 제안 ---
    annual_plans = plan_data.get("annual_plans", [])
    if annual_plans and len(annual_plans) > 0:
        plan = annual_plans[0]
        
        personnel_strategies = []
        for strategy in plan.get("personnel_strategies", []):
            if isinstance(strategy, dict):
                personnel_strategies.append({
                    "대상": strategy.get("target", ""),
                    "실행 방안": strategy.get("action", "")
                })
            elif isinstance(strategy, str):
                personnel_strategies.append({"실행 방안": strategy})
        
        collaboration_improvements = []
        for improvement in plan.get("collaboration_improvements", []):
            if isinstance(improvement, dict):
                collaboration_improvements.append({
                    "현재 문제점": improvement.get("current_issue", ""),
                    "개선 방안": improvement.get("improvement", ""),
                    "기대효과": improvement.get("expected_benefit", ""),
                    "목표 지표": improvement.get("target", "")
                })
            elif isinstance(improvement, str):
                collaboration_improvements.append({"개선 방안": improvement})
        
        다음연도운영제안 = {
            "핵심 인력 운용 전략": personnel_strategies,
            "협업 구조 개선 방향": collaboration_improvements
        }
    else:
        다음연도운영제안 = {
            "핵심 인력 운용 전략": [],
            "협업 구조 개선 방향": []
        }
    
    # --- 7. 총평 ---
    총평 = {
        "종합 의견": team_info.overall_comment or "작성된 총평이 없습니다."
    }

    # --- 최종 리포트 구성 ---
    final_report = {
        "기본 정보": 기본정보,
        "팀 종합 평가": 팀종합평가,
        "팀 업무 목표 및 달성률": 팀업무목표및달성률,
        "팀 성과 요약": 팀성과요약,
        "팀 조직력 및 리스크 요인": 팀조직력및리스크요인,
        "다음 연도 운영 제안": 다음연도운영제안,
        "총평": 총평
    }
    
    print(f"   - Period 4 최종 팀 평가 리포트 생성 완료")
    return final_report

# --- 4. DB 저장 및 메인 실행 함수 ---
def save_team_json_report_to_db(engine: Engine, team_evaluation_id: int, json_report: dict):
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)
    
    # Period 4 재검증을 위한 쿼리
    query = text("""
        UPDATE team_evaluations 
        SET report = :report 
        WHERE team_evaluation_id = :id AND period_id = 4
    """)
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                result = connection.execute(query, {"report": json_content, "id": team_evaluation_id})
                if result.rowcount > 0:
                    transaction.commit()
                    print(f"   - ✅ Team Eval ID {team_evaluation_id} (Period 4)의 최종 팀 평가 리포트가 DB에 저장되었습니다.")
                else:
                    transaction.rollback()
                    print(f"   - ⚠️ Team Eval ID {team_evaluation_id}가 Period 4가 아니거나 존재하지 않습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_team_report(report: dict) -> bool:
    required_keys = ["기본 정보", "팀 종합 평가", "팀 업무 목표 및 달성률", 
                    "팀 성과 요약", "팀 조직력 및 리스크 요인", 
                    "다음 연도 운영 제안", "총평"]
    
    print(f"   - 🔍 Period 4 최종 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    print(f"   - ✅ Period 4 최종 리포트 검증 성공!")
    return True

def main(team_evaluation_id: Optional[int] = None):
    """
    메인 실행 함수: Period 4에만 최종 팀 평가 리포트를 생성하고 report에 저장합니다.
    
    Args:
        team_evaluation_id: 특정 팀 평가 ID. None이면 모든 Period 4 팀 평가를 처리합니다.
    """
    try:
        engine = get_db_engine()

        if team_evaluation_id is None:
            # 모든 Period 4 팀 평가 처리
            print(f"\n🗑️ 기존 Period 4 최종 팀 평가 리포트 데이터 삭제 중...")
            clear_existing_period4_team_reports(engine)
            
            period4_ids = fetch_period4_team_evaluation_ids(engine)
            if not period4_ids:
                print("처리할 Period 4 팀 평가 데이터가 없습니다.")
                return

            success_count, error_count = 0, 0
            for current_team_eval_id in period4_ids:
                print(f"\n{'='*60}\n🚀 Period 4 최종 팀 평가 리포트 생성 시작 (ID: {current_team_eval_id})\n{'='*60}")
                try:
                    team_data = fetch_team_evaluation_data(engine, current_team_eval_id)
                    final_report = generate_team_evaluation_report(team_data)
                    
                    if not validate_team_report(final_report):
                        error_count += 1
                        continue
                    
                    save_team_json_report_to_db(engine, current_team_eval_id, final_report)
                    success_count += 1
                except Exception as e:
                    print(f"⚠️ ID {current_team_eval_id} 처리 중 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                time.sleep(0.5)
                
            print(f"\n🎉 Period 4 최종 팀 평가 리포트 생성 완료!")
            print(f"✅ 성공: {success_count}개")
            print(f"❌ 실패: {error_count}개")
            print(f"📊 총 처리: {len(period4_ids)}개")
            
        else:
            # 특정 팀 평가만 처리
            print(f"\n🎯 특정 팀 평가 ID {team_evaluation_id} 처리 시작")
            print(f"{'='*50}")
            
            try:
                team_data = fetch_team_evaluation_data(engine, team_evaluation_id)
                final_report = generate_team_evaluation_report(team_data)
                
                if not validate_team_report(final_report):
                    print(f"❌ 팀 평가 리포트 데이터 검증 실패")
                    return
                
                save_team_json_report_to_db(engine, team_evaluation_id, final_report)
                print(f"\n✅ Team Evaluation ID {team_evaluation_id} 처리 완료!")
                
            except Exception as e:
                print(f"❌ Team Evaluation ID {team_evaluation_id} 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return

    except Exception as e:
        print(f"메인 함수 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()