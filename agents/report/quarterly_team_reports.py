import os
import json
import time
from typing import Dict, Any, List, Optional, Sequence
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import ProgrammingError

from config.settings import DatabaseConfig

print("✅ 팀 평가 리포트 생성기 - 기본 라이브러리 임포트 완료")

# --- 1. 데이터베이스 연동 함수 ---

def get_db_engine() -> Engine:
    """
    config.settings의 DatabaseConfig를 사용하여 SQLAlchemy 엔진을 생성합니다.
    """
    db_config = DatabaseConfig()
    engine = create_engine(db_config.DATABASE_URL, pool_pre_ping=True)
    print("✅ 데이터베이스 엔진 생성 완료")
    return engine

def clear_existing_team_reports(engine: Engine, teams: Optional[list] = None, period_id: Optional[int] = None):
    """
    기존 team_evaluations.report 데이터를 NULL로 업데이트하여 삭제합니다.
    teams와 period_id가 주어지면 해당 팀, 해당 분기의 데이터만 삭제합니다.
    """
    try:
        if teams and period_id:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE team_evaluations 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND team_id IN ({placeholders})
                AND period_id = :period_id
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"🗑️ 팀 {teams}, 분기 {period_id}의 기존 team_evaluations.report 데이터를 삭제합니다...")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE team_evaluations 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND team_id IN ({placeholders})
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"🗑️ 팀 {teams}의 기존 team_evaluations.report 데이터를 삭제합니다...")
        elif period_id:
            query = text("""
                UPDATE team_evaluations 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND period_id = :period_id
            """)
            params = {'period_id': period_id}
            print(f"🗑️ 분기 {period_id}의 기존 team_evaluations.report 데이터를 삭제합니다...")
        else:
            query = text("UPDATE team_evaluations SET report = NULL WHERE report IS NOT NULL")
            params = {}
            print(f"🗑️ 모든 기존 team_evaluations.report 데이터를 삭제합니다...")
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query, params)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 team_evaluations.report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def fetch_team_evaluation_ids(engine: Engine, period_id: Optional[int] = None, teams: Optional[list] = None) -> List[int]:
    """
    조건에 맞는 team_evaluation_id들을 조회합니다.
    """
    try:
        if teams and period_id:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                SELECT team_evaluation_id FROM team_evaluations 
                WHERE team_id IN ({placeholders}) AND period_id = :period_id
                ORDER BY team_evaluation_id
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"✅ 팀 {teams}, 분기 {period_id}의 team_evaluation_id 조회")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                SELECT team_evaluation_id FROM team_evaluations 
                WHERE team_id IN ({placeholders})
                ORDER BY team_evaluation_id
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"✅ 팀 {teams}의 모든 team_evaluation_id 조회")
        elif period_id:
            query = text("""
                SELECT team_evaluation_id FROM team_evaluations 
                WHERE period_id = :period_id
                ORDER BY team_evaluation_id
            """)
            params = {'period_id': period_id}
            print(f"✅ 분기 {period_id}의 모든 team_evaluation_id 조회")
        else:
            query = text("SELECT team_evaluation_id FROM team_evaluations ORDER BY team_evaluation_id")
            params = {}
            print(f"✅ 모든 team_evaluation_id 조회")
        
        with engine.connect() as connection:
            results = connection.execute(query, params).fetchall()
        ids = [row[0] for row in results]
        print(f"✅ 총 {len(ids)}개의 팀 평가 리포트를 생성합니다. 대상 ID: {ids}")
        return ids
    except Exception as e:
        print(f"❌ team_evaluation_id 조회 중 오류 발생: {e}")
        return []

def fetch_team_evaluation_basic_data(engine: Engine, team_evaluation_id: int) -> Optional[Row]:
    """
    팀 평가의 기본 데이터를 조회합니다. (year_over_year_growth 포함)
    """
    try:
        query = text("""
            SELECT 
                te.team_evaluation_id, te.team_id, t.team_name,
                te.period_id, p.period_name,
                te.average_achievement_rate, te.year_over_year_growth,
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
    """
    팀 KPI 데이터를 조회합니다.
    """
    try:
        query = text("""
            SELECT kpi_name, ai_kpi_analysis_comment, ai_kpi_progress_rate
            FROM team_kpis WHERE team_id = :team_id ORDER BY ai_kpi_progress_rate DESC;
        """)
        with engine.connect() as connection:
            results = connection.execute(query, {"team_id": team_id}).fetchall()
        print(f"   - 팀 KPI 데이터 {len(results)}건 조회 완료")
        return list(results)
    except Exception as e:
        print(f"   - ❌ 팀 KPI 데이터 조회 중 오류 발생: {e}")
        return []

def fetch_team_members_feedback(engine: Engine, team_id: int, period_id: int) -> List[Row]:
    """
    팀원들의 피드백 데이터를 조회합니다 (DB의 ranking 순서).
    """
    try:
        query = text("""
            SELECT 
                fr.ranking, fr.ai_achievement_rate, e.emp_name,
                fr.ai_overall_contribution_summary_comment, fr.contribution_rate
            FROM feedback_reports fr
            JOIN employees e ON fr.emp_no = e.emp_no
            JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
            WHERE te.team_id = :team_id AND te.period_id = :period_id
            ORDER BY fr.ranking ASC, fr.ai_achievement_rate DESC;
        """)
        with engine.connect() as connection:
            results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        print(f"   - 팀원 피드백 데이터 {len(results)}건 조회 완료 (ranking 컬럼 기준 정렬)")
        return list(results)
    except Exception as e:
        print(f"   - ❌ 팀원 피드백 데이터 조회 중 오류 발생: {e}")
        return []

# --- 2. JSON 리포트 생성 함수 ---
class DecimalEncoder(json.JSONEncoder):
    """Decimal 타입을 JSON으로 직렬화하기 위한 커스텀 인코더"""
    def default(self, obj):
        if isinstance(obj, Decimal): return float(obj)
        return super(DecimalEncoder, self).default(obj)

def safe_json_parse(json_str: str) -> Dict[str, Any]:
    """JSON 문자열을 안전하게 파싱하는 헬퍼 함수"""
    try: 
        if json_str is None: return {} # None인 경우 빈 딕셔너리 반환
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError): 
        print(f"   - ⚠️ JSON 파싱 실패: {json_str[:100]}...") # 디버깅을 위해 일부 출력
        return {}

def safe_convert_to_serializable(obj):
    """모든 타입을 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict): return {k: safe_convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple): return tuple(safe_convert_to_serializable(item) for item in obj)
    if obj is None: return "" # None을 빈 문자열로 변환
    return obj

def generate_korean_team_evaluation_report(
    팀평가기본데이터: Row, 
    팀kpi데이터: List[Row], 
    팀원피드백데이터: List[Row]
) -> Dict[str, Any]:
    """
    한국어 팀 평가 리포트를 생성합니다.
    요구사항에 맞춰 중첩된 JSON 필드들을 상세히 파싱하여 포함합니다.
    """
    
    # JSON 컬럼 데이터 안전하게 파싱
    팀비교데이터 = safe_json_parse(팀평가기본데이터.ai_team_comparison)
    협업매트릭스데이터_raw = safe_json_parse(팀평가기본데이터.ai_collaboration_matrix)
    팀코칭데이터 = safe_json_parse(팀평가기본데이터.ai_team_coaching)
    리스크데이터 = safe_json_parse(팀평가기본데이터.ai_risk)
    
    # 협업 매트릭스 데이터 처리
    # ai_collaboration_matrix가 직접 리스트 형태의 JSON으로 저장될 경우를 대비
    협업매트릭스_리스트: List[Dict[str, Any]] = []
    if isinstance(협업매트릭스데이터_raw, list): # ai_collaboration_matrix가 직접 리스트인 경우
        협업매트릭스_리스트 = 협업매트릭스데이터_raw
        협업팀_요약 = "" # team_summary는 보통 리스트를 감싸는 객체에 있음.
    elif isinstance(협업매트릭스데이터_raw, dict): # ai_collaboration_matrix가 객체이고 그 안에 list가 있는 경우
        협업매트릭스_리스트 = 협업매트릭스데이터_raw.get("collaboration_matrix", [])
        협업팀_요약 = 협업매트릭스데이터_raw.get("team_summary", "")
    
    협업_네트워크_상세 = []
    for member_data in 협업매트릭스_리스트:
        if isinstance(member_data, dict):
            협업_네트워크_상세.append({
                "이름": member_data.get("name", ""),
                "총_Task_수": safe_convert_to_serializable(member_data.get("total_tasks", 0)),
                "협업률": f"{safe_convert_to_serializable(member_data.get('collaboration_rate', 0))}%",
                "핵심_협업자": member_data.get("key_collaborators", []),
                "팀_내_역할": member_data.get("team_role", ""),
                "Peer_Talk_평가": member_data.get("peer_talk_summary", ""),
                "협업_편중도": f"{safe_convert_to_serializable(member_data.get('collaboration_bias', 0))}",
                "종합_평가": member_data.get("overall_evaluation", "")
            })

    # 팀원별 코칭 제안 데이터 처리
    일반_코칭_리스트 = []
    for item in 팀코칭데이터.get("general_coaching", []):
        if isinstance(item, dict):
            name = item.get("name", "")
            emp_no = item.get("emp_no", "")
            팀원명_표시 = f"{name}({emp_no})" if name and emp_no else name or emp_no or ""
            
            일반_코칭_리스트.append({
                "팀원명(사번)": 팀원명_표시,
                "핵심_강점": item.get("strengths", []),
                "성장_보완점": item.get("improvement_points", []),
                "협업_특성": item.get("collaboration_style", ""),
                "성과_기여_요약": item.get("performance_summary", ""),
                "다음_분기_코칭_제안": item.get("next_quarter_coaching", "")
            })

    집중_코칭_리스트 = []
    for item in 팀코칭데이터.get("focused_coaching", []):
        if isinstance(item, dict):
            name = item.get("name", "")
            emp_no = item.get("emp_no", "")
            팀원명_표시 = f"{name}({emp_no})" if name and emp_no else name or emp_no or ""
            
            집중_코칭_리스트.append({
                "팀원명(사번)": 팀원명_표시,
                "핵심_이슈": item.get("issue_summary", ""),
                "상세_분석": item.get("root_cause_analysis", ""),
                "리스크_요소": item.get("risk_factors", []),
                "코칭_제안": item.get("coaching_plan", "")
            })
    
    # 리스크 및 향후 운영 제안 데이터 처리
    주요_리스크_리스트 = []
    for risk_item in 리스크데이터.get("risk_analysis", {}).get("major_risks", []):
        if isinstance(risk_item, dict):
            # 영향 예측 상세 처리
            영향_예측_리스트 = []
            for impact in risk_item.get("impacts", []):
                if isinstance(impact, dict):
                    영향_예측_리스트.append({
                        "영향_범위": impact.get("impact_scope", ""),
                        "발생_시점": impact.get("timeline", ""),
                        "영향_설명": impact.get("description", "")
                    })
            
            # 운영 개선 전략 상세 처리
            운영_개선_전략_리스트 = []
            for strategy in risk_item.get("strategies", []):
                if isinstance(strategy, dict):
                    운영_개선_전략_리스트.append({
                        "전략_설명": strategy.get("description", "")
                    })

            주요_리스크_리스트.append({
                "주요리스크": risk_item.get("risk_name", ""),
                "리스크_심각도": risk_item.get("severity", ""),
                "리스크_설명": risk_item.get("description", ""),
                "발생_원인": risk_item.get("causes", []),
                "영향_예측": 영향_예측_리스트,
                "운영_개선_전략_제안": 운영_개선_전략_리스트
            })


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
        },
        "팀_업무_목표_및_달성률": {
            "kpi_목록": [
                {
                    "팀_업무_목표": kpi.kpi_name or "", 
                    "kpi_분석_코멘트": kpi.ai_kpi_analysis_comment or "",
                    "달성률": safe_convert_to_serializable(kpi.ai_kpi_progress_rate),
                    # KPI별 비교 분석 데이터 매핑 (팀비교데이터의 'kpis'가 리스트를 가정)
                    # 실제 JSON 구조에 따라 이 부분은 수정이 필요할 수 있습니다.
                    # 여기서는 kpi_name을 기준으로 매핑을 시도합니다.
                    "달성률_평균_전사유사팀": safe_convert_to_serializable(
                        next((item.get("similar_avg_rate") for item in 팀비교데이터.get("kpis", []) if item.get("kpi_name") == kpi.kpi_name), 0)
                    ),
                    "비교_분석": next((item.get("comparison_result") for item in 팀비교데이터.get("kpis", []) if item.get("kpi_name") == kpi.kpi_name), "")
                } for kpi in 팀kpi데이터
            ],
            "전사_유사팀_비교분석_코멘트": 팀비교데이터.get("overall", {}).get("comment", "")
        },
        "팀원_성과_분석": {
            "팀원별_기여도": [
                {
                    "순위": member.ranking or 'N/A',
                    "이름": member.emp_name or "",
                    "달성률": safe_convert_to_serializable(member.ai_achievement_rate),
                    "누적_기여도": safe_convert_to_serializable(member.contribution_rate),
                    "기여_내용": member.ai_overall_contribution_summary_comment or ""
                } for member in 팀원피드백데이터
            ],
            # "종합_평가": "팀원들의 전반적인 성과 기여도와 달성률을 기준으로 한 종합 분석입니다.",
            "기여도_기준": "개인별 업무 달성률과 누적 기여도를 종합하여 평가하였습니다."
        },
        "협업_네트워크": {
            "협업_매트릭스": 협업_네트워크_상세, # 상세 매핑된 리스트 사용
            "팀_협업_요약": 협업팀_요약, # team_summary 사용
            "협업률_설명": "개인이 수행한 전체 업무 중, 다른 팀원과 실제로 협업한 업무가 차지하는 비율입니다.",
            "협업_편중도_설명": "특정 동료에게만 협업이 쏠려있는지, 혹은 여러 동료와 고르게 협업하는지를 나타내는 지표입니다."
        },
        "팀원별_코칭_제안": {
            "일반_코칭": 일반_코칭_리스트, # 상세 매핑된 리스트 사용
            "집중_코칭": 집중_코칭_리스트  # 상세 매핑된 리스트 사용
        },
        "리스크_및_향후_운영_제안": {
            "주요_리스크": 주요_리스크_리스트 # 상세 매핑된 리스트 사용
        },
        "총평": {
            "주요_인사이트": 팀평가기본데이터.overall_comment or "",
        }
    }
    
    print(f"   - 🔍 한국어 팀 평가 리포트 생성 완료")
    return 한국어리포트

# --- 3. DB 저장 및 메인 실행 함수 ---
def save_team_json_report_to_db(engine: Engine, team_evaluation_id: int, json_report: Dict[str, Any]):
    """
    생성된 JSON 리포트를 team_evaluations.report 컬럼에 저장합니다.
    """
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
    """
    한국어 팀 평가 리포트 JSON 데이터의 필수 키 유효성을 검증합니다.
    """
    required_keys = ["기본_정보", "팀_종합_평가", "팀_업무_목표_및_달성률", "팀원_성과_분석", "협업_네트워크", "팀원별_코칭_제안", "리스크_및_향후_운영_제안", "총평"]
    print(f"   - 🔍 팀 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    print(f"   - ✅ 팀 리포트 검증 성공!")
    return True

def main(period_id: Optional[int] = None, teams: Optional[list] = None):
    """
    메인 실행 함수: 팀 평가 리포트를 한국어 JSON으로 생성하고 DB에 저장합니다.
    
    Args:
        period_id: 특정 분기 ID. None이면 모든 분기를 처리합니다.
        teams: 특정 팀 ID 리스트. None이면 모든 팀을 처리합니다.
    """
    try:
        engine = get_db_engine()

        if period_id is None and teams is None:
            # 모든 팀 평가 처리
            print(f"\n🗑️ 모든 기존 team_evaluations.report 데이터를 삭제합니다...")
            clear_existing_team_reports(engine)

            all_team_evaluation_ids = fetch_team_evaluation_ids(engine)
            if not all_team_evaluation_ids:
                print("처리할 팀 평가가 데이터베이스에 없습니다.")
                return

            success_count, error_count = 0, 0
            for current_team_evaluation_id in all_team_evaluation_ids:
                print(f"\n{'='*50}\n🚀 팀 평가 리포트 생성 시작 (ID: {current_team_evaluation_id})\n{'='*50}")
                try:
                    팀평가기본데이터 = fetch_team_evaluation_basic_data(engine, current_team_evaluation_id)
                    if not 팀평가기본데이터:
                        print(f"⚠️ Team Evaluation ID {current_team_evaluation_id}를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                        error_count += 1
                        continue

                    팀kpi데이터 = fetch_team_kpis(engine, 팀평가기본데이터.team_id)
                    팀원피드백데이터 = fetch_team_members_feedback(engine, 팀평가기본데이터.team_id, 팀평가기본데이터.period_id)
                    
                    한국어리포트 = generate_korean_team_evaluation_report(팀평가기본데이터, 팀kpi데이터, 팀원피드백데이터)

                    if not validate_korean_team_report(한국어리포트):
                        print(f"   - ❌ 팀 리포트 데이터 검증 실패")
                        error_count += 1
                        continue

                    save_team_json_report_to_db(engine, current_team_evaluation_id, 한국어리포트)
                    success_count += 1

                except Exception as e:
                    print(f"⚠️ Team Evaluation ID {current_team_evaluation_id} 처리 중 심각한 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                time.sleep(0.5)

            print(f"\n🎉 팀 평가 리포트 생성 완료!")
            print(f"✅ 성공: {success_count}개")
            print(f"❌ 실패: {error_count}개")
            print(f"📊 총 처리: {len(all_team_evaluation_ids)}개")
            
        else:
            # 특정 조건으로 처리
            if teams and period_id:
                print(f"\n🎯 특정 팀 {teams}, 분기 {period_id} 처리 시작")
            elif teams:
                print(f"\n🎯 특정 팀 {teams} 처리 시작 (모든 분기)")
            elif period_id:
                print(f"\n🎯 특정 분기 {period_id} 처리 시작 (모든 팀)")
            print(f"{'='*50}")
            
            # 조건에 맞는 team_evaluation_id 조회
            target_team_evaluation_ids = fetch_team_evaluation_ids(engine, period_id, teams)
            if not target_team_evaluation_ids:
                print("조건에 맞는 팀 평가가 데이터베이스에 없습니다.")
                return

            success_count, error_count = 0, 0
            for current_team_evaluation_id in target_team_evaluation_ids:
                print(f"\n{'='*50}\n🚀 팀 평가 리포트 생성 시작 (ID: {current_team_evaluation_id})\n{'='*50}")
                try:
                    팀평가기본데이터 = fetch_team_evaluation_basic_data(engine, current_team_evaluation_id)
                    if not 팀평가기본데이터:
                        print(f"⚠️ Team Evaluation ID {current_team_evaluation_id}를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                        error_count += 1
                        continue

                    팀kpi데이터 = fetch_team_kpis(engine, 팀평가기본데이터.team_id)
                    팀원피드백데이터 = fetch_team_members_feedback(engine, 팀평가기본데이터.team_id, 팀평가기본데이터.period_id)
                    
                    한국어리포트 = generate_korean_team_evaluation_report(팀평가기본데이터, 팀kpi데이터, 팀원피드백데이터)

                    if not validate_korean_team_report(한국어리포트):
                        print(f"   - ❌ 팀 리포트 데이터 검증 실패")
                        error_count += 1
                        continue

                    save_team_json_report_to_db(engine, current_team_evaluation_id, 한국어리포트)
                    success_count += 1

                except Exception as e:
                    print(f"⚠️ Team Evaluation ID {current_team_evaluation_id} 처리 중 심각한 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                time.sleep(0.5)

            print(f"\n🎉 팀 평가 리포트 생성 완료!")
            print(f"✅ 성공: {success_count}개")
            print(f"❌ 실패: {error_count}개")
            print(f"📊 총 처리: {len(target_team_evaluation_ids)}개")
                
    except ValueError as e:
        print(f"설정 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 