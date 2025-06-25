import os
import json
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.row import Row
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

def clear_all_middle_reports(engine: Engine) -> int:
    """모든 middle_report 데이터를 삭제"""
    query = text("UPDATE team_evaluations SET middle_report = NULL WHERE middle_report IS NOT NULL")
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                result = connection.execute(query)
                cleared_count = result.rowcount
                transaction.commit()
                print(f"✅ {cleared_count}개의 기존 middle_report 데이터를 삭제했습니다.")
                return cleared_count
            except Exception as e:
                print(f"❌ middle_report 삭제 중 오류: {e}")
                transaction.rollback()
                raise

def fetch_all_team_evaluation_ids(engine: Engine) -> List[int]:
    """모든 팀 평가 ID 조회"""
    query = text("SELECT team_evaluation_id FROM team_evaluations ORDER BY team_evaluation_id;")
    with engine.connect() as connection:
        results = connection.execute(query).fetchall()
    ids = [row[0] for row in results]
    print(f"✅ 총 {len(ids)}개의 팀 평가 리포트를 생성합니다. 대상 ID: {ids}")
    return ids

# --- 2. JSON 구조 파싱 및 안전 처리 함수 ---

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

def safe_json_parse(json_str: str, default_value: dict = None) -> dict:
    if default_value is None: default_value = {}
    try:
        if json_str:
            parsed = json.loads(json_str)
            return safe_convert_to_serializable(parsed)
        return default_value
    except (json.JSONDecodeError, TypeError): return default_value

# --- ★★★ 성능 개선된 데이터 조회 함수 ★★★ ---
def fetch_team_report_data(engine: Engine, team_evaluation_id: int) -> dict:
    """팀 리포트 생성에 필요한 모든 데이터를 최소한의 쿼리로 조회하고 구조화하는 함수"""
    
    # 1. 팀 기본 정보 조회
    team_info_query = text("""
        SELECT te.team_evaluation_id, t.team_id, t.team_name, p.period_id, p.period_name, te.ai_collaboration_matrix
        FROM team_evaluations te
        JOIN teams t ON te.team_id = t.team_id
        JOIN periods p ON te.period_id = p.period_id
        WHERE te.team_evaluation_id = :id
    """)
    with engine.connect() as conn:
        team_info = conn.execute(team_info_query, {"id": team_evaluation_id}).first()
    if not team_info:
        raise ValueError(f"Team Evaluation ID {team_evaluation_id}를 찾을 수 없습니다.")

    # 2. 팀장 정보 조회
    manager_query = text("SELECT emp_name FROM employees WHERE team_id = :team_id AND role = 'MANAGER' LIMIT 1")
    with engine.connect() as conn:
        manager = conn.execute(manager_query, {"team_id": team_info.team_id}).first()

    # 3. 팀원 목록 및 임시 평가 데이터 한번에 조회 (raw_score 추가)
    members_query = text("""
        SELECT e.emp_no, e.emp_name, e.position, e.cl, te.score, te.raw_score, te.ai_reason, te.comment
        FROM temp_evaluations te
        JOIN employees e ON te.emp_no = e.emp_no
        WHERE te.team_evaluation_id = :id
    """)
    with engine.connect() as conn:
        members = conn.execute(members_query, {"id": team_evaluation_id}).fetchall()
    
    emp_nos = [member.emp_no for member in members]
    all_tasks, all_feedback, all_final_eval = {}, {}, {}

    if emp_nos:
        # 4. 모든 팀원의 Tasks, Feedback, Final Eval 데이터를 한번의 쿼리로 조회
        tasks_query = text("""
            SELECT tk.emp_no, tk.task_name, ts.task_performance, ts.ai_achievement_rate, ts.ai_analysis_comment_task
            FROM tasks tk JOIN task_summaries ts ON tk.task_id = ts.task_id
            WHERE tk.emp_no IN :emp_nos AND ts.period_id = :period_id
        """)
        feedback_query = text("""
            SELECT emp_no, ai_achievement_rate, ai_overall_contribution_summary_comment, ai_peer_talk_summary, 
                   ai_4p_evaluation, ai_growth_coaching, overall_comment
            FROM feedback_reports WHERE emp_no IN :emp_nos AND team_evaluation_id = :id
        """)
        final_eval_query = text("""
            SELECT emp_no, ai_annual_performance_summary_comment, ai_4p_evaluation as final_4p_evaluation
            FROM final_evaluation_reports WHERE emp_no IN :emp_nos AND team_evaluation_id = :id
        """)

        with engine.connect() as conn:
            tasks_results = conn.execute(tasks_query, {"emp_nos": emp_nos, "period_id": team_info.period_id}).fetchall()
            feedback_results = conn.execute(feedback_query, {"emp_nos": emp_nos, "id": team_evaluation_id}).fetchall()
            final_eval_results = conn.execute(final_eval_query, {"emp_nos": emp_nos, "id": team_evaluation_id}).fetchall()
        
        # 조회된 데이터를 emp_no를 키로 하는 딕셔너리로 변환
        for task in tasks_results:
            all_tasks.setdefault(task.emp_no, []).append(task)
        for feedback in feedback_results:
            all_feedback[feedback.emp_no] = feedback
        for final_eval in final_eval_results:
            all_final_eval[final_eval.emp_no] = final_eval
    
    # 5. 각 팀원별로 조회된 데이터를 조합
    member_details_list = []
    for member in members:
        member_details_list.append({
            "member_info": member,
            "tasks": all_tasks.get(member.emp_no, []),
            "feedback": all_feedback.get(member.emp_no),
            "final_eval": all_final_eval.get(member.emp_no)
        })
    
    print(f"   - 팀 ID {team_info.team_id}의 모든 데이터 조회 완료")
    return {"team_info": team_info, "manager": manager, "member_details": member_details_list}

# --- 3. 구조화된 리포트 생성 함수 ---

def generate_structured_team_report(team_data: dict) -> dict:
    """요구사항에 따라 완전히 구조화된 팀 리포트 생성"""
    
    team_info = team_data["team_info"]
    manager = team_data["manager"]
    member_details_list = team_data["member_details"]
    
    # 기본 정보
    basic_info = {
        "팀명": str(team_info.team_name or ""),
        "팀장명": str(manager.emp_name) if manager else "미지정",
        "업무 수행 기간": str(team_info.period_name or "")
    }
    
    # 팀원 평가 요약표
    member_summary_table = [
        {
            "팀원명": str(detail["member_info"].emp_name or ""),
            "AI 추천 점수 (CL 정규화)": safe_convert_to_serializable(detail["member_info"].score),
            "핵심 기여 요약": str(detail["member_info"].ai_reason or ""),
            "종합 코멘트": str(detail["member_info"].comment or "")
        } for detail in member_details_list
    ]
    
    # 협업 네트워크 (요구사항에 맞게 개선)
    collaboration_matrix_data = safe_json_parse(team_info.ai_collaboration_matrix, {})
    
    # 협업 네트워크 표 구조 개선
    collaboration_table = []
    members_data = collaboration_matrix_data.get("members", [])
    
    for member_data in members_data:
        collaboration_table.append({
            "이름": str(member_data.get("name", "")),
            "총 Task 수": safe_convert_to_serializable(member_data.get("total_tasks", 0)),
            "협업률": f"{safe_convert_to_serializable(member_data.get('collaboration_rate', 0))}%",
            "핵심 협업자": ", ".join(member_data.get("key_collaborators", [])),
            "팀 내 역할": str(member_data.get("team_role", "")),
            "협업 편중도": str(member_data.get("collaboration_bias", "N/A")),
            "종합 평가": str(member_data.get("overall_evaluation", ""))
        })
    
    collaboration_network = {
        "협업 네트워크 표": collaboration_table,
        "팀 요약": str(collaboration_matrix_data.get("team_summary", "")),
        "설명": {
            "협업률": "개인이 수행한 전체 업무 중, 다른 팀원과 실제로 협업한 업무가 차지하는 비율입니다.",
            "협업 편중도": "특정 동료에게만 협업이 쏠려있는지, 혹은 여러 동료와 고르게 협업하는지를 나타내는 지표입니다."
        }
    }
    
    # 팀원별 상세 평가
    member_detailed_evaluations = []
    for detail in member_details_list:
        member, tasks, feedback, final_eval = (
            detail["member_info"], detail.get("tasks", []), 
            detail.get("feedback"), detail.get("final_eval")
        )
        
        cl_level = f"CL{member.cl}" if member.cl else ""
        basic_content = {
            "이름": str(member.emp_name or ""), 
            "직무": str(member.position or ""), 
            "CL레벨": cl_level
        }
        
        # AI 점수 산출 기준 데이터 처리
        raw_score_data = safe_json_parse(getattr(member, 'raw_score', None), {})
        final_4p_data = safe_json_parse(getattr(final_eval, 'final_4p_evaluation', None), {})

        ai_score_criteria = {
            "업적 점수": safe_convert_to_serializable(raw_score_data.get("achievement_score", "N/A")),
            "실적 요약": str(getattr(final_eval, 'ai_annual_performance_summary_comment', '')),
            "SK Values 평가": {
                p: {
                    "점수": safe_convert_to_serializable(d.get("score", 0)), 
                    "평가 근거": str(d.get("reasoning", "평가 데이터 없음"))
                }
                for p, d in final_4p_data.items()
            },
            "종합 원점수": safe_convert_to_serializable(raw_score_data.get("raw_hybrid_score", "N/A")),
            "AI 추천 점수 (CL 정규화)": safe_convert_to_serializable(member.score),
            "평가 근거 요약": str(member.comment or "")
        }
        
        # 연간 핵심 성과 기여도
        performance_table = [
            {
                "Task명": str(task.task_name or ""), 
                "핵심 Task": str(task.task_performance or ""),
                "누적 달성률 (%)": safe_convert_to_serializable(task.ai_achievement_rate),
                "분석 코멘트": str(task.ai_analysis_comment_task or "")
            } for task in tasks
        ]
        
        annual_key_performance = {
            "성과 표": performance_table,
            "개인 종합 달성률": safe_convert_to_serializable(getattr(feedback, 'ai_achievement_rate', 0)),
            "종합 기여 코멘트": str(getattr(feedback, 'ai_overall_contribution_summary_comment', ''))
        }
        
        # Peer Talk 데이터
        peer_talk_data = safe_json_parse(getattr(feedback, 'ai_peer_talk_summary', None), {})
        peer_talk = {
            "강점": peer_talk_data.get("strengths", []),
            "우려": peer_talk_data.get("concerns", []),
            "협업 관찰": peer_talk_data.get("collaboration_observations", [])
        }
        
        member_detailed_evaluations.append({
            "기본 내용": basic_content,
            "AI 점수 산출 기준": ai_score_criteria,
            "연간 핵심 성과 기여도": annual_key_performance,
            "Peer Talk": peer_talk
        })
    
    korean_report = {
        "기본 정보": basic_info,
        "팀원 평가 요약표": member_summary_table,
        "협업 네트워크": collaboration_network,
        "팀원별 평가 근거": member_detailed_evaluations
    }
    
    print(f"   - 구조화된 팀 리포트 생성 완료 (팀원 {len(member_detailed_evaluations)}명)")
    return korean_report

# --- 4. DB 저장 및 메인 실행 함수 ---

def save_json_report_to_db(engine: Engine, team_evaluation_id: int, json_report: dict):
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_report = safe_convert_to_serializable(json_report)
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, default=str)
    
    query = text("UPDATE team_evaluations SET middle_report = :report WHERE team_evaluation_id = :id")
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                connection.execute(query, {"report": json_content, "id": team_evaluation_id})
                transaction.commit()
                print(f"   - ✅ Team Eval ID {team_evaluation_id}의 구조화된 리포트가 DB에 저장되었습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_structured_report(report: dict) -> bool:
    required_keys = ["기본 정보", "팀원 평가 요약표", "협업 네트워크", "팀원별 평가 근거"]
    print(f"   - 🔍 리포트 키 확인: {list(report.keys())}")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    
    # 협업 네트워크 구조 검증
    collaboration = report.get("협업 네트워크", {})
    if "협업 네트워크 표" not in collaboration or "설명" not in collaboration:
        print(f"   - ⚠️ 협업 네트워크 구조 누락")
        return False
    
    print(f"   - ✅ 리포트 검증 성공!")
    return True

def main():
    try:
        engine = get_db_engine()
        print(f"\n🗑️ 모든 기존 middle_report 데이터를 삭제합니다...")
        clear_all_middle_reports(engine)
        all_ids = fetch_all_team_evaluation_ids(engine)
        if not all_ids:
            print("처리할 팀 평가 데이터가 없습니다.")
            return

        success_count, error_count = 0, 0
        for team_eval_id in all_ids:
            print(f"\n{'='*60}\n🚀 구조화된 팀 리포트 생성 시작 (ID: {team_eval_id})\n{'='*60}")
            try:
                team_data = fetch_team_report_data(engine, team_eval_id)
                structured_report = generate_structured_team_report(team_data)
                
                if not validate_structured_report(structured_report):
                    print(f"   - ❌ 리포트 데이터 검증 실패")
                    error_count += 1
                    continue
                
                save_json_report_to_db(engine, team_eval_id, structured_report)
                success_count += 1
            except Exception as e:
                print(f"⚠️ ID {team_eval_id} 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
            time.sleep(0.5)
            
        print(f"\n🎉 구조화된 팀 리포트 생성 완료!")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        print(f"📊 총 처리: {len(all_ids)}개")

    except Exception as e:
        print(f"메인 함수 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()