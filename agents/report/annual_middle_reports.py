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

print("✅ 연말 중간평가 리포트 생성기 - 기본 라이브러리 임포트 완료")

# --- 1. 데이터베이스 연동 함수 ---

def get_db_engine() -> Engine:
    """
    config.settings의 DatabaseConfig를 사용하여 SQLAlchemy 엔진을 생성합니다.
    """
    db_config = DatabaseConfig()
    engine = create_engine(db_config.DATABASE_URL, pool_pre_ping=True)
    print("✅ 데이터베이스 엔진 생성 완료")
    return engine

def clear_existing_middle_reports(engine: Engine, teams: Optional[list] = None, period_id: Optional[int] = None):
    """
    기존 team_evaluations.middle_report 데이터를 NULL로 업데이트하여 삭제합니다.
    teams와 period_id가 주어지면 해당 팀, 해당 분기의 데이터만 삭제합니다.
    """
    try:
        if teams and period_id:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE team_evaluations 
                SET middle_report = NULL 
                WHERE middle_report IS NOT NULL 
                AND team_id IN ({placeholders})
                AND period_id = :period_id
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"🗑️ 팀 {teams}, 분기 {period_id}의 기존 team_evaluations.middle_report 데이터를 삭제합니다...")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                UPDATE team_evaluations 
                SET middle_report = NULL 
                WHERE middle_report IS NOT NULL 
                AND team_id IN ({placeholders})
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"🗑️ 팀 {teams}의 기존 team_evaluations.middle_report 데이터를 삭제합니다...")
        elif period_id:
            query = text("""
                UPDATE team_evaluations 
                SET middle_report = NULL 
                WHERE middle_report IS NOT NULL 
                AND period_id = :period_id
            """)
            params = {'period_id': period_id}
            print(f"🗑️ 분기 {period_id}의 기존 team_evaluations.middle_report 데이터를 삭제합니다...")
        else:
            query = text("UPDATE team_evaluations SET middle_report = NULL WHERE middle_report IS NOT NULL")
            params = {}
            print(f"🗑️ 모든 기존 team_evaluations.middle_report 데이터를 삭제합니다...")
        with engine.connect() as connection:
            with connection.begin() as transaction:
                result = connection.execute(query, params)
                affected_rows = result.rowcount
                transaction.commit()
                print(f"✅ 기존 team_evaluations.middle_report 데이터 {affected_rows}개 삭제 완료")
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
                SELECT team_evaluation_id 
                FROM team_evaluations 
                WHERE team_id IN ({placeholders}) AND period_id = :period_id
                ORDER BY team_evaluation_id;
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            params['period_id'] = period_id
            print(f"✅ 팀 {teams}, 분기 {period_id}의 team_evaluation_id 조회")
        elif teams:
            placeholders = ','.join([':team_id_' + str(i) for i in range(len(teams))])
            query = text(f"""
                SELECT team_evaluation_id 
                FROM team_evaluations 
                WHERE team_id IN ({placeholders})
                ORDER BY team_evaluation_id;
            """)
            params = {f'team_id_{i}': team_id for i, team_id in enumerate(teams)}
            print(f"✅ 팀 {teams}의 모든 team_evaluation_id 조회")
        elif period_id:
            query = text("""
                SELECT team_evaluation_id 
                FROM team_evaluations 
                WHERE period_id = :period_id
                ORDER BY team_evaluation_id;
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
        print(f"✅ 총 {len(ids)}개의 연말 중간평가 리포트를 생성합니다. 대상 ID: {ids}")
        return ids
    except Exception as e:
        print(f"❌ team_evaluation_id 조회 중 오류 발생: {e}")
        return []

def check_period_is_final(engine: Engine, period_id: int) -> bool:
    """
    해당 period가 연말(최종) 평가인지 확인합니다.
    """
    try:
        query = text("SELECT is_final FROM periods WHERE period_id = :period_id")
        with engine.connect() as connection:
            result = connection.execute(query, {"period_id": period_id}).first()
        if result:
            is_final = result.is_final
            if is_final:
                print(f"✅ 분기 {period_id}는 연말(최종) 평가입니다.")
            else:
                print(f"⚠️ 분기 {period_id}는 연말(최종) 평가가 아닙니다. 중간평가 리포트 생성이 적절하지 않을 수 있습니다.")
            return is_final
        else:
            print(f"⚠️ 분기 {period_id} 정보를 찾을 수 없습니다.")
            return False
    except Exception as e:
        print(f"❌ 분기 정보 조회 중 오류 발생: {e}")
        return False

def fetch_team_evaluation_basic_data(engine: Engine, team_evaluation_id: int) -> Optional[Row]:
    """팀 평가의 기본 데이터를 조회합니다."""
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

def fetch_team_member_evaluation_summary(engine: Engine, team_evaluation_id: int) -> List[Row]:
    """팀원 평가 요약표 데이터 조회"""
    try:
        query = text("""
            SELECT 
                e.emp_name,
                te.score as ai_recommended_score,
                te.ai_reason as key_contribution_summary
            FROM temp_evaluations te
            JOIN employees e ON te.emp_no = e.emp_no
            WHERE te.team_evaluation_id = :team_evaluation_id
            ORDER BY te.score DESC;
        """)
        with engine.connect() as connection:
            results = connection.execute(query, {"team_evaluation_id": team_evaluation_id}).fetchall()
        print(f"   - 팀원 평가 요약표 데이터 {len(results)}건 조회 완료")
        return list(results)
    except Exception as e:
        print(f"   - ❌ 팀원 평가 요약표 조회 중 오류 발생: {e}")
        return []

def fetch_team_members_detailed_evaluation(engine: Engine, team_evaluation_id: int) -> List[Dict]:
    """팀원별 상세 평가 데이터 조회"""
    try:
        # 팀원 기본 정보 및 점수 조회 (단순화된 쿼리)
        member_query = text("""
            SELECT 
                e.emp_no, e.emp_name, e.position, e.cl,
                te.raw_score, te.score as ai_recommended_score, te.ai_reason as key_contribution_summary
            FROM temp_evaluations te
            JOIN employees e ON te.emp_no = e.emp_no
            WHERE te.team_evaluation_id = :team_evaluation_id
            ORDER BY te.score DESC;
        """)
        
        with engine.connect() as connection:
            members = connection.execute(member_query, {"team_evaluation_id": team_evaluation_id}).fetchall()
        
        if not members:
            print(f"   - ⚠️ temp_evaluations에서 팀원 데이터를 찾을 수 없습니다.")
            return []
        
        detailed_members = []
        for member in members:
            # 각 팀원의 추가 데이터 개별 조회
            
            # final_evaluation_reports 데이터 조회
            fer_query = text("""
                SELECT ai_annual_performance_summary_comment, ai_4p_evaluation
                FROM final_evaluation_reports 
                WHERE emp_no = :emp_no AND team_evaluation_id = :team_evaluation_id
            """)
            
            # feedback_reports 데이터 조회 (Peer Talk 데이터가 있는 것 우선)
            fr_query = text("""
                SELECT fr.ai_achievement_rate, fr.ai_overall_contribution_summary_comment, fr.ai_peer_talk_summary
                FROM feedback_reports fr
                WHERE fr.emp_no = :emp_no AND fr.ai_peer_talk_summary IS NOT NULL AND fr.ai_peer_talk_summary != ''
                ORDER BY fr.created_at DESC
                LIMIT 1
            """)
            
            # Task 데이터 조회
            task_query = text("""
                SELECT tk.task_name, ts.task_performance, ts.ai_achievement_rate, ts.ai_analysis_comment_task
                FROM tasks tk
                JOIN task_summaries ts ON tk.task_id = ts.task_id
                WHERE tk.emp_no = :emp_no AND ts.period_id = 4;
            """)
            
            with engine.connect() as connection:
                # 추가 데이터 개별 조회
                fer_result = connection.execute(fer_query, {"emp_no": member.emp_no, "team_evaluation_id": team_evaluation_id}).first()
                fr_result = connection.execute(fr_query, {"emp_no": member.emp_no}).first()
                
                # Peer Talk 데이터가 없으면 다른 feedback_reports 레코드도 시도
                if not fr_result or not fr_result.ai_peer_talk_summary:
                    print(f"     - Peer Talk 데이터가 없어서 모든 feedback_reports 검색")
                    fr_fallback_query = text("""
                        SELECT fr.ai_achievement_rate, fr.ai_overall_contribution_summary_comment, fr.ai_peer_talk_summary
                        FROM feedback_reports fr
                        WHERE fr.emp_no = :emp_no
                        ORDER BY fr.created_at DESC
                        LIMIT 5
                    """)
                    fr_all_results = connection.execute(fr_fallback_query, {"emp_no": member.emp_no}).fetchall()
                    
                    # Peer Talk 데이터가 있는 첫 번째 레코드 찾기
                    for fr_record in fr_all_results:
                        if fr_record.ai_peer_talk_summary and fr_record.ai_peer_talk_summary.strip():
                            fr_result = fr_record
                            print(f"     - Peer Talk 데이터 발견: {fr_record.ai_peer_talk_summary[:100]}...")
                            break
                
                tasks = connection.execute(task_query, {"emp_no": member.emp_no}).fetchall()
            
            # 데이터 병합 (Row 객체 확장)
            member_dict = dict(member._mapping)  # Row를 dict로 변환
            member_dict.update({
                'ai_annual_performance_summary_comment': fer_result.ai_annual_performance_summary_comment if fer_result else "",
                'ai_4p_evaluation': fer_result.ai_4p_evaluation if fer_result else "{}",  # 빈 JSON 문자열로 기본값
                'ai_achievement_rate': fr_result.ai_achievement_rate if fr_result else 0,
                'ai_overall_contribution_summary_comment': fr_result.ai_overall_contribution_summary_comment if fr_result else "",
                'ai_peer_talk_summary': fr_result.ai_peer_talk_summary if fr_result else "{}"  # 빈 JSON 문자열로 기본값
            })
            
            # namedtuple 형태로 변환 (기존 코드 호환성 유지)
            from collections import namedtuple
            MemberInfo = namedtuple('MemberInfo', member_dict.keys())
            member_info = MemberInfo(**member_dict)
            
            detailed_members.append({
                "member_info": member_info,
                "tasks": tasks
            })
            
            print(f"   - {member.emp_name}님 상세 데이터 조회 완료: Tasks {len(tasks)}개")
        
        print(f"   - 팀원별 상세 평가 데이터 {len(detailed_members)}명 조회 완료")
        return detailed_members
        
    except Exception as e:
        print(f"   - ❌ 팀원별 상세 평가 데이터 조회 중 오류 발생: {e}")
        return []

def fetch_collaboration_matrix(engine: Engine, team_evaluation_id: int) -> Dict[str, Any]:
    """협업 네트워크 데이터 조회"""
    try:
        query = text("""
            SELECT ai_collaboration_matrix
            FROM team_evaluations
            WHERE team_evaluation_id = :team_evaluation_id;
        """)
        with engine.connect() as connection:
            result = connection.execute(query, {"team_evaluation_id": team_evaluation_id}).first()
        
        if result and result.ai_collaboration_matrix:
            collaboration_data = safe_json_parse(result.ai_collaboration_matrix)
            print(f"   - 협업 네트워크 데이터 조회 완료")
            return collaboration_data
        else:
            print(f"   - 협업 네트워크 데이터 없음")
            return {}
            
    except Exception as e:
        print(f"   - ❌ 협업 네트워크 데이터 조회 중 오류 발생: {e}")
        return {}

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
        print(f"   - ⚠️ JSON 파싱 실패: {json_str[:100] if json_str else 'None'}...") # 디버깅을 위해 일부 출력
        return {}

def safe_convert_to_serializable(obj):
    """모든 타입을 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict): return {k: safe_convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple): return tuple(safe_convert_to_serializable(item) for item in obj)
    if obj is None: return "" # None을 빈 문자열로 변환
    return obj

def format_cl_level(cl_value) -> str:
    """CL 레벨 포맷팅"""
    if not cl_value:
        return ""
    cl_str = str(cl_value).strip()
    if cl_str and cl_str.isdigit():
        return f"CL{cl_str}"
    elif cl_str and not cl_str.startswith("CL"):
        return f"CL{cl_str}"
    return cl_str

def generate_middle_evaluation_report(
    팀평가기본데이터: Row,
    팀원평가요약: List[Row], 
    팀원상세평가: List[Dict],
    협업네트워크: Dict[str, Any]
) -> Dict[str, Any]:
    """
    최종전 중간평가 리포트를 생성합니다. (Period 4용)
    """
    
    print(f"   - 🔍 리포트 생성 시작: 팀원상세평가 {len(팀원상세평가)}명")
    
    # 1. 기본 정보
    기본정보 = {
        "팀명": 팀평가기본데이터.team_name or "",
        "팀장명": 팀평가기본데이터.manager_name or "미지정",
        "업무_수행_기간": 팀평가기본데이터.period_name or ""
    }
    
    # 2. 팀원 평가 요약표 (협업 네트워크와 통합)
    팀원평가요약표 = []
    협업팀_요약 = ""
    
    if isinstance(협업네트워크, dict) and 협업네트워크:
        if "collaboration_matrix" in 협업네트워크:
            협업매트릭스_리스트 = 협업네트워크.get("collaboration_matrix", [])
            협업팀_요약 = 협업네트워크.get("team_summary", "")
        elif isinstance(협업네트워크.get("data"), list):  # 직접 리스트인 경우
            협업매트릭스_리스트 = 협업네트워크.get("data", [])
    else:
        협업매트릭스_리스트 = []
    
    if 협업매트릭스_리스트:
        for member_data in 협업매트릭스_리스트:
            if isinstance(member_data, dict):
                # 해당 팀원의 AI 추천 점수 찾기
                member_name = member_data.get("name", "")
                ai_score = "데이터 없음"
                
                # 이름 매칭 (괄호 제거하여 비교)
                clean_member_name = member_name.split('(')[0].strip() if '(' in member_name else member_name
                
                for member in 팀원평가요약:
                    member_emp_name = member.emp_name or ""
                    if member_emp_name == member_name or member_emp_name == clean_member_name:
                        ai_score = safe_convert_to_serializable(member.ai_recommended_score)
                        print(f"     - AI 점수 매칭 성공: {member_name} -> {ai_score}")
                        break
                
                if ai_score == "데이터 없음":
                    print(f"     - ⚠️ AI 점수 매칭 실패: {member_name} (찾을 수 있는 emp_name: {[m.emp_name for m in 팀원평가요약]})")
                
                팀원평가요약표.append({
                    "이름": member_name,
                    "AI_추천_점수": ai_score,
                    "총_Task_수": safe_convert_to_serializable(member_data.get("total_tasks", 0)),
                    "협업률": f"{safe_convert_to_serializable(member_data.get('collaboration_rate', 0))}%",
                    "핵심_협업자": member_data.get("key_collaborators", []),
                    "팀_내_역할": member_data.get("team_role", ""),
                    "Peer_Talk_평가": member_data.get("peer_talk_summary", ""),
                    "협업_편중도": f"{safe_convert_to_serializable(member_data.get('collaboration_bias', 0))}",
                    "종합_평가": member_data.get("overall_evaluation", "")
                })
    else:
        # 협업 데이터가 없는 경우 기본 메시지
        팀원평가요약표.append({
            "이름": "협업 데이터 없음",
            "AI_추천_점수": "데이터 없음",
            "총_Task_수": "데이터 없음",
            "협업률": "데이터 없음",
            "핵심_협업자": ["협업 매트릭스 데이터가 등록되지 않았습니다."],
            "팀_내_역할": "데이터 등록 필요",
            "Peer_Talk_평가": "데이터 없음",
            "협업_편중도": "데이터 없음",
            "종합_평가": "협업 네트워크 분석을 위해서는 ai_collaboration_matrix 데이터가 필요합니다."
        })
        협업팀_요약 = "협업 네트워크 데이터가 등록되지 않아 분석이 불가능합니다."
    
    # 팀원 평가 요약표 섹션 구성
    팀원평가요약표_섹션 = {
        "표": 팀원평가요약표,
        "팀_협업_요약": 협업팀_요약,
        "협업률_설명": "개인이 수행한 전체 업무 중, 다른 팀원과 실제로 협업한 업무가 차지하는 비율입니다.",
        "협업_편중도_설명": "특정 동료에게만 협업이 쏠려있는지, 혹은 여러 동료와 고르게 협업하는지를 나타내는 지표입니다."
    }
    
    # 4. 팀원별 평가 근거 (반복 섹션)
    팀원별_평가근거 = []
    
    print(f"   - 팀원별 평가근거 생성 시작: {len(팀원상세평가)}명")
    
    for idx, member_detail in enumerate(팀원상세평가):
        print(f"   - [{idx+1}] {member_detail['member_info'].emp_name} 처리 시작")
        
        member_info = member_detail["member_info"]
        tasks = member_detail["tasks"]
        
        try:
            # 기본 내용
            기본내용 = {
                "이름": member_info.emp_name or "",
                "직무": member_info.position or "",
                "CL_레벨": format_cl_level(member_info.cl)
            }
            print(f"     - 기본내용 완료: {기본내용['이름']}")
            
            # AI 점수 산출 기준
            raw_score_data = safe_json_parse(member_info.raw_score if hasattr(member_info, 'raw_score') and member_info.raw_score else "{}")
            fourp_evaluation_data = safe_json_parse(member_info.ai_4p_evaluation if hasattr(member_info, 'ai_4p_evaluation') and member_info.ai_4p_evaluation else "{}")
            
            ai_점수_산출_기준 = {
                "업적": {
                    "점수": safe_convert_to_serializable(raw_score_data.get("achievement_score", "데이터 없음")),
                    "실적_요약": getattr(member_info, 'ai_annual_performance_summary_comment', '') or "실적 요약 데이터가 없습니다."
                },
                "SK_Values": {
                    "Passive": {
                        "점수": safe_convert_to_serializable(fourp_evaluation_data.get("passionate", {}).get("score", "데이터 없음")),
                        "평가_근거": fourp_evaluation_data.get("passionate", {}).get("reasoning", "평가 근거 데이터가 없습니다.")
                    },
                    "Proactive": {
                        "점수": safe_convert_to_serializable(fourp_evaluation_data.get("proactive", {}).get("score", "데이터 없음")),
                        "평가_근거": fourp_evaluation_data.get("proactive", {}).get("reasoning", "평가 근거 데이터가 없습니다.")
                    },
                    "Professional": {
                        "점수": safe_convert_to_serializable(fourp_evaluation_data.get("professional", {}).get("score", "데이터 없음")),
                        "평가_근거": fourp_evaluation_data.get("professional", {}).get("reasoning", "평가 근거 데이터가 없습니다.")
                    },
                    "People": {
                        "점수": safe_convert_to_serializable(fourp_evaluation_data.get("people", {}).get("score", "데이터 없음")),
                        "평가_근거": fourp_evaluation_data.get("people", {}).get("reasoning", "평가 근거 데이터가 없습니다.")
                    }
                },
                "종합_원점수": safe_convert_to_serializable(raw_score_data.get("raw_hybrid_score", "데이터 없음")),
                "AI_추천_점수_CL_정규화": safe_convert_to_serializable(getattr(member_info, 'ai_recommended_score', 'N/A')),
                "평가_근거_요약": getattr(member_info, 'key_contribution_summary', '') or "평가 근거 요약이 없습니다."
            }
            print(f"     - AI 점수 산출 기준 완료")
            
            # 연간 핵심 성과 기여도
            연간_핵심_성과_표 = []
            if tasks:
                for task in tasks:
                    연간_핵심_성과_표.append({
                        "Task명": task.task_name or "",
                        "핵심_Task": task.task_performance or "",
                        "누적_달성률_퍼센트": safe_convert_to_serializable(task.ai_achievement_rate),
                        "분석_코멘트": task.ai_analysis_comment_task or ""
                    })
            else:
                # Task 데이터가 없는 경우 기본 메시지
                연간_핵심_성과_표.append({
                    "Task명": "Task 데이터 없음",
                    "핵심_Task": "Period 4 Task 데이터가 등록되지 않았습니다.",
                    "누적_달성률_퍼센트": "데이터 없음",
                    "분석_코멘트": "Task 등록 후 데이터 확인이 필요합니다."
                })
            
            연간_핵심_성과_기여도 = {
                "Task_표": 연간_핵심_성과_표,
                "개인_종합_달성률": safe_convert_to_serializable(getattr(member_info, 'ai_achievement_rate', None)) if getattr(member_info, 'ai_achievement_rate', None) else "데이터 없음",
                "종합_기여_코멘트": getattr(member_info, 'ai_overall_contribution_summary_comment', '') or "종합 기여 코멘트 데이터가 없습니다."
            }
            print(f"     - 연간 핵심 성과 기여도 완료")
            
            # Peer Talk (강화된 데이터 처리)
            peer_talk_raw = getattr(member_info, 'ai_peer_talk_summary', None)
            print(f"     - Peer Talk Raw Data: {peer_talk_raw}")
            
            if peer_talk_raw and peer_talk_raw.strip():
                peer_talk_data = safe_json_parse(peer_talk_raw)
                print(f"     - Peer Talk Parsed: {peer_talk_data}")
                
                peer_talk_섹션 = {
                    "강점": peer_talk_data.get('strengths', []) if peer_talk_data.get('strengths') else ["강점 데이터가 등록되지 않았습니다."],
                    "우려": peer_talk_data.get('concerns', []) if peer_talk_data.get('concerns') else ["우려 데이터가 등록되지 않았습니다."],
                    "협업_관찰": peer_talk_data.get('collaboration_observations', '') if peer_talk_data.get('collaboration_observations') else "협업 관찰 데이터가 등록되지 않았습니다."
                }
            else:
                print(f"     - Peer Talk 데이터가 없음")
                peer_talk_섹션 = {
                    "강점": ["Peer Talk 데이터가 feedback_reports.ai_peer_talk_summary에 등록되지 않았습니다."],
                    "우려": ["Peer Talk 데이터가 feedback_reports.ai_peer_talk_summary에 등록되지 않았습니다."],
                    "협업_관찰": "Peer Talk 데이터가 feedback_reports.ai_peer_talk_summary에 등록되지 않았습니다."
                }
            
            print(f"     - Peer Talk 섹션 완료: {peer_talk_섹션}")
            
            # 팀원별 평가 근거 완성
            팀원평가근거_항목 = {
                "기본_내용": 기본내용,
                "AI_점수_산출_기준": ai_점수_산출_기준,
                "연간_핵심_성과_기여도": 연간_핵심_성과_기여도,
                "Peer_Talk": peer_talk_섹션
            }
            
            팀원별_평가근거.append(팀원평가근거_항목)
            print(f"     - [{idx+1}] {member_info.emp_name} 완료!")
            
        except Exception as e:
            print(f"     - ❌ {member_info.emp_name} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"   - 팀원별 평가근거 최종 완료: {len(팀원별_평가근거)}개 생성")
    
    # 최종 리포트 구성
    중간평가리포트 = {
        "기본_정보": 기본정보,
        "팀원_평가_요약표": 팀원평가요약표_섹션,
        "팀원별_평가_근거": 팀원별_평가근거
    }
    
    print(f"   - 🔍 최종전 중간평가 리포트 생성 완료")
    return 중간평가리포트

# --- 3. DB 저장 및 메인 실행 함수 ---
def save_middle_report_to_db(engine: Engine, team_evaluation_id: int, json_report: Dict[str, Any]):
    """
    생성된 JSON 리포트를 team_evaluations.middle_report 컬럼에 저장합니다.
    """
    try:
        json_content = json.dumps(json_report, ensure_ascii=False, indent=2, cls=DecimalEncoder)
    except Exception as e:
        print(f"   - ❌ JSON 직렬화 오류: {e}")
        json_content = json.dumps(safe_convert_to_serializable(json_report), ensure_ascii=False, indent=2, default=str)

    query = text("""
        UPDATE team_evaluations 
        SET middle_report = :report_content 
        WHERE team_evaluation_id = :team_evaluation_id;
    """)
    
    with engine.connect() as connection:
        with connection.begin() as transaction:
            try:
                result = connection.execute(query, {"report_content": json_content, "team_evaluation_id": team_evaluation_id})
                if result.rowcount > 0:
                    transaction.commit()
                    print(f"   - ✅ Team Evaluation ID {team_evaluation_id}의 중간평가 리포트가 middle_report에 성공적으로 저장되었습니다.")
                else:
                    transaction.rollback()
                    print(f"   - ⚠️ Team Evaluation ID {team_evaluation_id}가 존재하지 않습니다.")
            except Exception as e:
                print(f"   - ❌ DB 저장 중 오류 발생: {e}")
                transaction.rollback()
                raise

def validate_middle_report(report: dict) -> bool:
    """
    중간평가 리포트 JSON 데이터의 필수 키 유효성을 검증합니다.
    """
    required_keys = [
        "기본_정보", "팀원_평가_요약표", "팀원별_평가_근거"
    ]
    
    print(f"   - 🔍 연말 중간평가 리포트 검증 시작...")
    for key in required_keys:
        if key not in report:
            print(f"   - ⚠️ 필수 키 누락: {key}")
            return False
    
    print(f"   - ✅ 연말 중간평가 리포트 검증 성공!")
    return True

def main(period_id: Optional[int] = None, teams: Optional[list] = None):
    """
    메인 실행 함수: 연말 중간평가 리포트를 생성하고 middle_report에 저장합니다.
    
    Args:
        period_id: 특정 분기 ID. None이면 모든 분기를 처리합니다.
        teams: 특정 팀 ID 리스트. None이면 모든 팀을 처리합니다.
    """
    try:
        engine = get_db_engine()

        if period_id is None and teams is None:
            # 모든 팀 평가 처리
            print(f"\n🗑️ 모든 기존 team_evaluations.middle_report 데이터를 삭제합니다...")
            clear_existing_middle_reports(engine)

            all_team_evaluation_ids = fetch_team_evaluation_ids(engine)
            if not all_team_evaluation_ids:
                print("처리할 팀 평가가 데이터베이스에 없습니다.")
                return

            success_count, error_count = 0, 0
            for current_team_evaluation_id in all_team_evaluation_ids:
                print(f"\n{'='*50}\n🚀 연말 중간평가 리포트 생성 시작 (ID: {current_team_evaluation_id})\n{'='*50}")
                try:
                    팀평가기본데이터 = fetch_team_evaluation_basic_data(engine, current_team_evaluation_id)
                    if not 팀평가기본데이터:
                        print(f"⚠️ Team Evaluation ID {current_team_evaluation_id}를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                        error_count += 1
                        continue

                    # 연말 여부 확인
                    is_final = check_period_is_final(engine, 팀평가기본데이터.period_id)
                    if not is_final:
                        print(f"⚠️ Period {팀평가기본데이터.period_id}는 연말 평가가 아닙니다. 중간평가 리포트 생성이 적절하지 않을 수 있습니다.")

                    팀원평가요약 = fetch_team_member_evaluation_summary(engine, current_team_evaluation_id)
                    팀원상세평가 = fetch_team_members_detailed_evaluation(engine, current_team_evaluation_id)
                    협업네트워크 = fetch_collaboration_matrix(engine, current_team_evaluation_id)
                    
                    중간평가리포트 = generate_middle_evaluation_report(
                        팀평가기본데이터, 팀원평가요약, 팀원상세평가, 협업네트워크
                    )

                    if not validate_middle_report(중간평가리포트):
                        print(f"   - ❌ 중간평가 리포트 데이터 검증 실패")
                        error_count += 1
                        continue

                    save_middle_report_to_db(engine, current_team_evaluation_id, 중간평가리포트)
                    success_count += 1

                except Exception as e:
                    print(f"⚠️ Team Evaluation ID {current_team_evaluation_id} 처리 중 심각한 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                
                time.sleep(0.5)  # DB 부하 방지

            print(f"\n🎉 연말 중간평가 리포트 생성 완료!")
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
            
            # 연말 여부 확인
            if period_id:
                is_final = check_period_is_final(engine, period_id)
                if not is_final:
                    print(f"⚠️ 분기 {period_id}는 연말 평가가 아닙니다. 중간평가 리포트 생성이 적절하지 않을 수 있습니다.")
            
            # 조건에 맞는 team_evaluation_id 조회
            target_team_evaluation_ids = fetch_team_evaluation_ids(engine, period_id, teams)
            if not target_team_evaluation_ids:
                print("조건에 맞는 팀 평가가 데이터베이스에 없습니다.")
                return

            success_count, error_count = 0, 0
            for current_team_evaluation_id in target_team_evaluation_ids:
                print(f"\n{'='*50}\n🚀 연말 중간평가 리포트 생성 시작 (ID: {current_team_evaluation_id})\n{'='*50}")
                try:
                    팀평가기본데이터 = fetch_team_evaluation_basic_data(engine, current_team_evaluation_id)
                    if not 팀평가기본데이터:
                        print(f"⚠️ Team Evaluation ID {current_team_evaluation_id}를 조회하는 데 실패했습니다. 다음으로 넘어갑니다.")
                        error_count += 1
                        continue

                    팀원평가요약 = fetch_team_member_evaluation_summary(engine, current_team_evaluation_id)
                    팀원상세평가 = fetch_team_members_detailed_evaluation(engine, current_team_evaluation_id)
                    협업네트워크 = fetch_collaboration_matrix(engine, current_team_evaluation_id)
                    
                    중간평가리포트 = generate_middle_evaluation_report(
                        팀평가기본데이터, 팀원평가요약, 팀원상세평가, 협업네트워크
                    )

                    if not validate_middle_report(중간평가리포트):
                        print(f"   - ❌ 중간평가 리포트 데이터 검증 실패")
                        error_count += 1
                        continue

                    save_middle_report_to_db(engine, current_team_evaluation_id, 중간평가리포트)
                    success_count += 1

                except Exception as e:
                    print(f"⚠️ Team Evaluation ID {current_team_evaluation_id} 처리 중 심각한 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    continue
                time.sleep(0.5)

            print(f"\n🎉 연말 중간평가 리포트 생성 완료!")
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