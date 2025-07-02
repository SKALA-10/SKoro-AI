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

print("✅ 개인 평가 리포트 생성기 - 기본 라이브러리 임포트 완료")

# --- 1. 데이터베이스 연동 함수 ---

def get_db_engine() -> Engine:
    """
    config.settings의 DatabaseConfig를 사용하여 SQLAlchemy 엔진을 생성합니다.
    """
    db_config = DatabaseConfig()
    engine = create_engine(db_config.DATABASE_URL, pool_pre_ping=True)
    print("✅ 데이터베이스 엔진 생성 완료")
    return engine

def clear_existing_feedback_reports(engine: Engine, teams: Optional[list] = None, period_id: Optional[int] = None):
    """
    기존 feedback_reports.report 데이터를 NULL로 업데이트하여 삭제합니다.
    teams와 period_id가 주어지면 해당 팀, 해당 분기의 직원들만 삭제합니다.
    """
    try:
        if teams and period_id:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE feedback_reports 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND emp_no IN (
                    SELECT emp_no FROM employees WHERE team_id IN ({placeholders})
                )
                AND team_evaluation_id IN (
                    SELECT team_evaluation_id FROM team_evaluations WHERE period_id = :period_id
                )
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"🗑️ 팀 {teams}, 분기 {period_id}의 기존 feedback_reports.report 데이터를 삭제합니다...")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE feedback_reports 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND emp_no IN (
                    SELECT emp_no FROM employees WHERE team_id IN ({placeholders})
                )
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"🗑️ 팀 {teams}의 기존 feedback_reports.report 데이터를 삭제합니다...")
        elif period_id:
            query = text("""
                UPDATE feedback_reports 
                SET report = NULL 
                WHERE report IS NOT NULL 
                AND team_evaluation_id IN (
                    SELECT team_evaluation_id FROM team_evaluations WHERE period_id = :period_id
                )
            """)
            params = {'period_id': period_id}
            print(f"🗑️ 분기 {period_id}의 기존 feedback_reports.report 데이터를 삭제합니다...")
        else:
            query = text("UPDATE feedback_reports SET report = NULL WHERE report IS NOT NULL")
            params = {}
            print(f"🗑️ 모든 기존 feedback_reports.report 데이터를 삭제합니다...")
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query, params)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 feedback_reports.report 데이터 {affected_rows}개 삭제 완료")
    except Exception as e:
        print(f"❌ 기존 데이터 삭제 중 오류 발생: {e}")
        raise

def fetch_team_emp_nos(engine: Engine, teams: list) -> list:
    """
    특정 팀들의 직원 번호를 조회합니다.
    """
    placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
    query = text(f"SELECT DISTINCT emp_no FROM employees WHERE team_id IN ({placeholders}) ORDER BY emp_no;")
    params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
    with engine.connect() as connection:
        results = connection.execute(query, params).fetchall()
    emp_nos = [row[0] for row in results]
    print(f"✅ 팀 {teams}의 직원 {len(emp_nos)}명을 처리합니다. 대상 직원: {emp_nos}")
    return emp_nos

def fetch_all_feedback_report_ids(engine: Engine) -> List[str]:
    """
    feedback_reports 테이블에서 모든 직원 번호를 조회합니다.
    """
    query = text("SELECT DISTINCT emp_no FROM feedback_reports ORDER BY emp_no;")
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()
    emp_nos = [row[0] for row in results]
    print(f"✅ 총 {len(emp_nos)}개의 개인 평가 리포트를 생성합니다. 대상 직원: {emp_nos}")
    return emp_nos

def fetch_feedback_basic_data(engine: Engine, emp_no: str, period_id: Optional[int] = None) -> Optional[Row]:
    """
    개인 평가의 기본 데이터를 조회합니다.
    """
    try:
        if period_id:
            query = text("""
                SELECT
                    e.emp_no, e.emp_name, e.cl, e.position,
                    t.team_name, p.period_id, p.period_name,
                    fr.ai_achievement_rate,
                    fr.ai_overall_contribution_summary_comment,
                    fr.ai_peer_talk_summary, fr.ai_4p_evaluation,
                    fr.ai_growth_coaching, fr.overall_comment
                FROM feedback_reports fr
                JOIN employees e ON fr.emp_no = e.emp_no
                JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                JOIN teams t ON te.team_id = t.team_id
                JOIN periods p ON te.period_id = p.period_id
                WHERE fr.emp_no = :emp_no AND p.period_id = :period_id
                LIMIT 1;
            """)
            params = {"emp_no": emp_no, "period_id": period_id}
        else:
            query = text("""
                SELECT
                    e.emp_no, e.emp_name, e.cl, e.position,
                    t.team_name, p.period_id, p.period_name,
                    fr.ai_achievement_rate,
                    fr.ai_overall_contribution_summary_comment,
                    fr.ai_peer_talk_summary, fr.ai_4p_evaluation,
                    fr.ai_growth_coaching, fr.overall_comment
                FROM feedback_reports fr
                JOIN employees e ON fr.emp_no = e.emp_no
                JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                JOIN teams t ON te.team_id = t.team_id
                JOIN periods p ON te.period_id = p.period_id
                WHERE fr.emp_no = :emp_no
                LIMIT 1;
            """)
            params = {"emp_no": emp_no}
            
        with engine.connect() as connection:
            result = connection.execute(query, params).first()
        if result:
            print(f"   - {emp_no}님 개인 평가 데이터 조회 완료: {result.emp_name}")
        return result
    except Exception as e:
        print(f"   - ❌ 개인 평가 데이터 조회 중 오류 발생: {e}")
        return None

def fetch_quarterly_performance(engine: Engine, emp_no: str) -> List[Dict]:
    """
    분기별 성과 데이터를 조회합니다.
    """
    # 1~4분기 기본 구조 먼저 생성
    base_quarters = [
        {"분기": "1분기", "달성률": 0, "실적 요약": ""},
        {"분기": "2분기", "달성률": 0, "실적 요약": ""},
        {"분기": "3분기", "달성률": 0, "실적 요약": ""},
        {"분기": "4분기", "달성률": 0, "실적 요약": ""}
    ]
    
    quarterly_data = base_quarters.copy()
    
    try:
        query = text("""
            SELECT fr.ai_achievement_rate as achievement_rate,
                   fr.ai_overall_contribution_summary_comment as performance_summary, 
                   p.period_name, p.order_in_year
            FROM feedback_reports fr
            JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
            JOIN periods p ON te.period_id = p.period_id
            WHERE fr.emp_no = :emp_no 
            ORDER BY p.order_in_year
        """)
        
        with engine.connect() as conn:
            feedback_results = conn.execute(query, {"emp_no": emp_no}).fetchall()
            print(f"   - {emp_no}님의 분기별 성과 조회: {len(feedback_results)}건")

        # 조회된 데이터를 기본 구조에 매핑
        for result in feedback_results:
            period_name = result.period_name or ""
            
            # 분기 매핑
            quarter_index = None
            if "1분기" in period_name:
                quarter_index = 0
            elif "2분기" in period_name:
                quarter_index = 1
            elif "3분기" in period_name:
                quarter_index = 2
            elif "4분기" in period_name:
                quarter_index = 3
            
            # 해당 분기에 데이터 업데이트
            if quarter_index is not None:
                quarterly_data[quarter_index] = {
                    "분기": f"{quarter_index + 1}분기",
                    "달성률": result.achievement_rate or 0,
                    "실적 요약": result.performance_summary or ""
                }
        
    except Exception as e:
        print(f"   - ❌ 분기별 성과 조회 중 오류: {e}")

    print(f"   - {emp_no}님의 분기별 성과 데이터 처리 완료")
    return quarterly_data

def fetch_tasks_for_final_report(engine: Engine, emp_no: str, period_id: int) -> List[Row]:
    """
    직원의 Task 데이터를 조회합니다.
    """
    try:
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
        return list(results)
    except Exception as e:
        print(f"   - ❌ Task 데이터 조회 중 오류 발생: {e}")
        return []

def fetch_temp_evaluation_data(engine: Engine, emp_no: str) -> Optional[Row]:
    """
    임시 평가 데이터를 조회합니다.
    """
    try:
        query = text("""
            SELECT raw_score, comment
            FROM temp_evaluations
            WHERE emp_no = :emp_no
            LIMIT 1
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"emp_no": emp_no}).first()
        print(f"   - {emp_no}님의 임시 평가 데이터 조회 완료")
        return result
    except Exception as e:
        print(f"   - 임시 평가 데이터 조회 실패: {e}")
        return None

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

def parse_4p_evaluation_text(evaluation_text: str) -> Dict[str, str]:
    """4P 평가 텍스트를 파싱해서 각 항목별로 분리"""
    result = {
        "Passionate": "",
        "Proactive": "", 
        "Professional": "",
        "People": "",
        "종합 평가": ""
    }
    
    if not evaluation_text:
        return result
    
    lines = evaluation_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('* Passionate'):
            result["Passionate"] = line.replace('* Passionate 성과 하이라이트: ', '').replace('* Passionate', '').strip()
        elif line.startswith('* Proactive'):
            result["Proactive"] = line.replace('* Proactive 주도적 성과: ', '').replace('* Proactive', '').strip()
        elif line.startswith('* Professional'):
            result["Professional"] = line.replace('* Professional 전문성 발휘: ', '').replace('* Professional', '').strip()
        elif line.startswith('* People'):
            result["People"] = line.replace('* People 협업 기여: ', '').replace('* People', '').strip()
        elif line.startswith('* 종합 평가'):
            result["종합 평가"] = line.replace('* 종합 평가: ', '').strip()
    
    return result

def generate_korean_feedback_report(
    피드백기본데이터: Row,
    업무데이터: List[Row],
    임시평가데이터: Optional[Row]
) -> Dict[str, Any]:
    """
    한국어 개인 평가 리포트를 생성합니다.
    """
    
    # JSON 컬럼 데이터 안전하게 파싱
    peer_talk_데이터 = safe_json_parse(피드백기본데이터.ai_peer_talk_summary)
    growth_데이터 = safe_json_parse(피드백기본데이터.ai_growth_coaching)
    fourp_데이터 = safe_json_parse(피드백기본데이터.ai_4p_evaluation)
    
    # 4P 평가 텍스트 파싱
    fourp_파싱데이터 = parse_4p_evaluation_text(fourp_데이터.get('evaluation_text', ''))

    # CL 레벨 처리
    cl_레벨 = str(피드백기본데이터.cl).strip() if 피드백기본데이터.cl else ""
    if cl_레벨 and cl_레벨.isdigit():
        cl_레벨 = f"CL{cl_레벨}"
    elif cl_레벨 and not cl_레벨.startswith("CL"):
        cl_레벨 = f"CL{cl_레벨}"

    # 업무표 데이터 처리
    업무표 = [
        {
            "Task명": t.task_name or "",
            "핵심 Task": t.task_performance or "",
            "누적 달성률 (%)": safe_convert_to_serializable(t.ai_achievement_rate),
            "분석 코멘트": t.ai_analysis_comment_task or ""
        }
        for t in 업무데이터
    ]

    한국어리포트 = {
        "기본_정보": {
            "성명": 피드백기본데이터.emp_name or "",
            "직위": cl_레벨,
            "소속": 피드백기본데이터.team_name or "",
            "업무_수행_기간": 피드백기본데이터.period_name or ""
        },
        "팀_업무_목표_및_개인_달성률": {
            "업무표": 업무표,
            "개인_종합_달성률": safe_convert_to_serializable(피드백기본데이터.ai_achievement_rate),
            "종합_기여_코멘트": 피드백기본데이터.ai_overall_contribution_summary_comment or "",
            "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
        },
        "Peer_Talk": {
            "강점": peer_talk_데이터.get('strengths', []),
            "우려": peer_talk_데이터.get('concerns', []),
            "협업_관찰": peer_talk_데이터.get('collaboration_observations', "")
        },
        "업무_실행_및_태도": {
            "Passionate": fourp_파싱데이터.get('Passionate', ''),
            "Proactive": fourp_파싱데이터.get('Proactive', ''), 
            "Professional": fourp_파싱데이터.get('Professional', ''),
            "People": fourp_파싱데이터.get('People', ''),
            "종합_평가": fourp_파싱데이터.get('종합 평가', '')
        },
        "성장_제안_및_개선_피드백": {
            "성장_포인트": growth_데이터.get('growth_points', []),
            "보완_영역": growth_데이터.get('improvement_areas', []),
            "추천_활동": growth_데이터.get('recommended_activities', [])
        },
        "총평": 피드백기본데이터.overall_comment or ""
    }

    print(f"   - 🔍 한국어 개인 평가 리포트 생성 완료")
    return 한국어리포트

# --- 3. DB 저장 및 메인 실행 함수 ---
def save_feedback_json_report_to_db(engine: Engine, emp_no: str, json_report: Dict[str, Any], period_id: Optional[int] = None):
    """
    생성된 JSON 리포트를 feedback_reports.report 컬럼에 저장합니다.
    """
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)

    if period_id:
        query = text("""
            UPDATE feedback_reports 
            SET report = :report_content 
            WHERE emp_no = :emp_no 
            AND team_evaluation_id IN (
                SELECT te.team_evaluation_id 
                FROM team_evaluations te 
                WHERE te.period_id = :period_id
            );
        """)
        params = {"report_content": json_content, "emp_no": emp_no, "period_id": period_id}
    else:
        query = text("UPDATE feedback_reports SET report = :report_content WHERE emp_no = :emp_no;")
        params = {"report_content": json_content, "emp_no": emp_no}
        
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                result = connection.execute(query, params)
                if result.rowcount > 0:
                    transaction.commit()
                    print(f"   - ✅ {emp_no}님의 개인 평가 리포트가 feedback_reports.report에 성공적으로 저장되었습니다.")
                else:
                    transaction.rollback()
                    print(f"   - ⚠️ {emp_no}님에 해당하는 레코드를 찾을 수 없어 DB에 저장하지 못했습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_korean_feedback_report(report: dict) -> bool:
    """
    한국어 개인 평가 리포트 JSON 데이터의 필수 키 유효성을 검증합니다.
    """
    required_keys = ["기본_정보", "팀_업무_목표_및_개인_달성률", "Peer_Talk", 
                    "업무_실행_및_태도", "성장_제안_및_개선_피드백", "총평"]
    print(f"   - 🔍 개인 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    print(f"   - ✅ 개인 리포트 검증 성공!")
    return True

def main(emp_no: Optional[str] = None, period_id: Optional[int] = None, teams: Optional[list] = None):
    """
    메인 실행 함수: 개인 평가 리포트를 한국어 JSON으로 생성하고 DB에 저장합니다.
    
    Args:
        emp_no: 특정 직원 번호. None이면 모든 직원을 처리합니다.
        period_id: 특정 분기 ID. None이면 모든 분기를 처리합니다.
        teams: 특정 팀들의 직원 번호 리스트. None이면 모든 직원을 처리합니다.
    """
    try:
        engine = get_db_engine()

        if emp_no is None:
            # 모든 개인 평가 처리
            print(f"\n🗑️ 모든 기존 feedback_reports.report 데이터를 삭제합니다...")
            clear_existing_feedback_reports(engine, teams, period_id)

            if teams:
                all_emp_nos = fetch_team_emp_nos(engine, teams)
            else:
                all_emp_nos = fetch_all_feedback_report_ids(engine)
            if not all_emp_nos:
                print("처리할 개인 평가가 데이터베이스에 없습니다.")
                return

            success_count, error_count = 0, 0
            for current_emp_no in all_emp_nos:
                print(f"\n{'='*50}\n🚀 개인 평가 리포트 생성 시작 ({current_emp_no})\n{'='*50}")
                try:
                    피드백기본데이터 = fetch_feedback_basic_data(engine, current_emp_no, period_id)
                    if not 피드백기본데이터:
                        print(f"⚠️ {current_emp_no}님의 개인 평가 데이터를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                        error_count += 1
                        continue

                    업무데이터 = fetch_tasks_for_final_report(engine, current_emp_no, 피드백기본데이터.period_id)
                    임시평가데이터 = fetch_temp_evaluation_data(engine, current_emp_no)

                    한국어리포트 = generate_korean_feedback_report(
                        피드백기본데이터, 업무데이터, 임시평가데이터
                    )

                    if not validate_korean_feedback_report(한국어리포트):
                        print(f"   - ❌ 개인 리포트 데이터 검증 실패")
                        error_count += 1
                        continue

                    save_feedback_json_report_to_db(engine, current_emp_no, 한국어리포트, period_id)
                    success_count += 1

                except Exception as e:
                    print(f"⚠️ {current_emp_no}님 처리 중 심각한 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                time.sleep(0.5)

            print(f"\n🎉 개인 평가 리포트 생성 완료!")
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
                피드백기본데이터 = fetch_feedback_basic_data(engine, emp_no, period_id)
                if not 피드백기본데이터:
                    print(f"❌ {emp_no}님의 개인 평가 데이터를 조회할 수 없습니다.")
                    return

                업무데이터 = fetch_tasks_for_final_report(engine, emp_no, 피드백기본데이터.period_id)
                임시평가데이터 = fetch_temp_evaluation_data(engine, emp_no)

                한국어리포트 = generate_korean_feedback_report(
                    피드백기본데이터, 업무데이터, 임시평가데이터
                )

                if not validate_korean_feedback_report(한국어리포트):
                    print(f"❌ 개인 리포트 데이터 검증 실패")
                    return

                save_feedback_json_report_to_db(engine, emp_no, 한국어리포트, period_id)
                print(f"\n✅ {emp_no}님 처리 완료!")
                
            except Exception as e:
                print(f"❌ {emp_no}님 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return
                
    except ValueError as e:
        print(f"설정 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()