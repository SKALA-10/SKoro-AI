# ================================================================
# 모듈 10: 개인 성장 및 코칭 모듈 - 완전한 단일 파일 구현
# ================================================================

from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END
import json
import re
import sys
import os

# 기존 imports
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

from config.settings import DatabaseConfig

load_dotenv()

# DB 설정 - 다른 모듈과 동일한 방식 사용
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"LLM Client initialized: {llm_client.model_name}")

def row_to_dict(row: Row) -> Dict:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ================================================================
# Module10AgentState 정의
# ================================================================

class Module10AgentState(TypedDict):
    """모듈 10 (개인 성장 및 코칭) 상태"""
    messages: Annotated[List[HumanMessage], operator.add]
    
    # 입력 정보
    emp_no: str
    period_id: int
    report_type: str  # "quarterly" or "annual"
    
    # 수집된 데이터 (기본 5개 + 연말 추가 2개)
    basic_info: Dict
    performance_data: Dict
    peer_talk_data: Dict
    fourp_data: Dict
    collaboration_data: Dict
    
    # 연말 추가 데이터
    module7_score_data: Dict  # 팀 내 정규화 점수
    module9_final_data: Dict  # 부문 정규화 최종 점수
    
    # 중간 처리 결과
    growth_analysis: Dict
    focus_coaching_needed: bool
    focus_coaching_analysis: Dict
    
    # 최종 결과
    individual_growth_result: Dict  # 개인용 JSON
    manager_coaching_result: Dict   # 팀장용 JSON
    overall_comment: str            # 종합 총평
    storage_result: Dict
    
    # 처리 상태
    processing_status: str
    error_messages: List[str]

# ================================================================
# 데이터 수집 함수들
# ================================================================

def fetch_basic_info(emp_no: str) -> Dict:
    """기본 정보 조회"""
    with engine.connect() as connection:
        try:
            query = text("""
                SELECT emp_no, emp_name, cl, position, team_id
                FROM employees WHERE emp_no = :emp_no
            """)
            result = connection.execute(query, {"emp_no": emp_no}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"기본 정보 조회 실패: {e}")
            return {}

def fetch_performance_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """성과 데이터 수집 (모듈 2 결과)"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                query = text("""
                    SELECT fr.contribution_rate, fr.ai_overall_contribution_summary_comment,
                           AVG(ts.ai_achievement_rate) as ai_achievement_rate,
                           AVG(ts.ai_contribution_score) as avg_contribution_score
                    FROM feedback_reports fr
                    JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                    LEFT JOIN (
                        SELECT ts.*, t.emp_no 
                        FROM task_summaries ts 
                        JOIN tasks t ON ts.task_id = t.task_id
                        WHERE ts.period_id = :period_id
                    ) ts ON ts.emp_no = fr.emp_no
                    WHERE fr.emp_no = :emp_no AND te.period_id = :period_id
                    GROUP BY fr.emp_no, fr.contribution_rate, fr.ai_overall_contribution_summary_comment
                """)
            else:  # annual
                query = text("""
                    SELECT fer.contribution_rate, fer.ai_annual_achievement_rate as ai_achievement_rate,
                           fer.ai_annual_performance_summary_comment, fer.score
                    FROM final_evaluation_reports fer
                    JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                    WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
                """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"성과 데이터 조회 실패: {e}")
            return {}

def fetch_peer_talk_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """Peer Talk 데이터 수집 (모듈 4 결과)"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                table = "feedback_reports"
            else:
                table = "final_evaluation_reports"
                
            query = text(f"""
                SELECT fer.ai_peer_talk_summary
                FROM {table} fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            
            if result and result.ai_peer_talk_summary:
                try:
                    return json.loads(result.ai_peer_talk_summary)
                except json.JSONDecodeError:
                    print(f"Peer Talk JSON 파싱 실패: {emp_no}")
                    return {}
            return {}
        except Exception as e:
            print(f"Peer Talk 데이터 조회 실패: {e}")
            return {}

def fetch_fourp_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """4P 데이터 수집 (모듈 6 결과)"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                table = "feedback_reports"
            else:
                table = "final_evaluation_reports"
                
            query = text(f"""
                SELECT fer.ai_4p_evaluation
                FROM {table} fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            
            if result and result.ai_4p_evaluation:
                try:
                    return json.loads(result.ai_4p_evaluation)
                except json.JSONDecodeError:
                    print(f"4P JSON 파싱 실패: {emp_no}")
                    return {}
            return {}
        except Exception as e:
            print(f"4P 데이터 조회 실패: {e}")
            return {}

def fetch_collaboration_data(emp_no: str, period_id: int) -> Dict:
    """협업 데이터 수집 (모듈 3 결과에서 개인 부분 추출)"""
    with engine.connect() as connection:
        try:
            # 직원의 team_id 조회
            team_query = text("SELECT team_id FROM employees WHERE emp_no = :emp_no")
            team_result = connection.execute(team_query, {"emp_no": emp_no}).fetchone()
            
            if not team_result:
                return {}
                
            team_id = team_result.team_id
            
            # team_evaluations에서 협업 매트릭스 조회
            collab_query = text("""
                SELECT ai_collaboration_matrix
                FROM team_evaluations
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            collab_result = connection.execute(collab_query, {
                "team_id": team_id, 
                "period_id": period_id
            }).fetchone()
            
            if collab_result and collab_result.ai_collaboration_matrix:
                try:
                    collaboration_matrix = json.loads(collab_result.ai_collaboration_matrix)
                    
                    # collaboration_matrix에서 해당 emp_no 찾기
                    for member in collaboration_matrix.get("collaboration_matrix", []):
                        if member.get("emp_no") == emp_no:
                            return {
                                "collaboration_rate": member.get("collaboration_rate", 0),
                                "team_role": member.get("team_role", ""),
                                "key_collaborators": member.get("key_collaborators", []),
                                "collaboration_bias": member.get("collaboration_bias", "보통"),
                                "overall_evaluation": member.get("overall_evaluation", "")
                            }
                except json.JSONDecodeError:
                    print(f"협업 매트릭스 JSON 파싱 실패: {emp_no}")
                    return {}
            
            return {}
        except Exception as e:
            print(f"협업 데이터 조회 실패: {e}")
            return {}

def fetch_module7_score_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """모듈 7 팀 내 정규화 점수 데이터 수집 (연말만)"""
    if report_type != "annual":
        return {}
        
    with engine.connect() as connection:
        try:
            # temp_evaluations에서 팀 내 정규화 점수 조회
            query = text("""
                SELECT raw_score, score, ai_reason
                FROM temp_evaluations
                WHERE emp_no = :emp_no
            """)
            
            result = connection.execute(query, {"emp_no": emp_no}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"모듈 7 점수 데이터 조회 실패: {e}")
            return {}

def fetch_module9_final_score_data(emp_no: str, period_id: int, report_type: str) -> Dict:
    """모듈 9 부문 정규화 최종 점수 데이터 수집 (연말만)"""
    if report_type != "annual":
        return {}
        
    with engine.connect() as connection:
        try:
            # final_evaluation_reports에서 최종 점수
            query = text("""
                SELECT fer.score, fer.ranking, fer.cl_reason
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
            """)
            
            result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
            return row_to_dict(result) if result else {}
        except Exception as e:
            print(f"모듈 9 최종 점수 데이터 조회 실패: {e}")
            return {}

def calculate_ranking_by_achievement(emp_no: str, team_id: str, period_id: int, report_type: str) -> int:
    """팀 내 달성률 기반 순위를 동적으로 계산"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                # 분기 랭킹: task_summaries의 ai_achievement_rate 평균 기준
                query = text("""
                    WITH team_achievements AS (
                        SELECT
                            e.emp_no,
                            COALESCE(AVG(ts.ai_achievement_rate), 0) as achievement_rate
                        FROM employees e
                        LEFT JOIN (
                            SELECT t.emp_no, ts.ai_achievement_rate
                            FROM tasks t
                            JOIN task_summaries ts ON t.task_id = ts.task_id
                            WHERE ts.period_id = :period_id
                        ) ts ON e.emp_no = ts.emp_no
                        WHERE e.team_id = :team_id
                        GROUP BY e.emp_no
                    )
                    SELECT
                        emp_no,
                        RANK() OVER (ORDER BY achievement_rate DESC) as ranking
                    FROM team_achievements
                """)
            else:  # annual
                # 연간 랭킹: final_evaluation_reports의 ai_annual_achievement_rate 기준
                query = text("""
                    WITH team_achievements AS (
                        SELECT
                            e.emp_no,
                            COALESCE(fer.ai_annual_achievement_rate, 0) as achievement_rate
                        FROM employees e
                        LEFT JOIN (
                            SELECT fer_inner.emp_no, fer_inner.ai_annual_achievement_rate
                            FROM final_evaluation_reports fer_inner
                            JOIN team_evaluations te ON fer_inner.team_evaluation_id = te.team_evaluation_id
                            WHERE te.period_id = :period_id
                        ) fer ON e.emp_no = fer.emp_no
                        WHERE e.team_id = :team_id
                    )
                    SELECT
                        emp_no,
                        RANK() OVER (ORDER BY achievement_rate DESC) as ranking
                    FROM team_achievements
                """)

            params = {"team_id": team_id, "period_id": period_id}
            rank_list = connection.execute(query, params).fetchall()

            for row in rank_list:
                if row[0] == emp_no:
                    return int(row[1]) if row[1] is not None else 0

            return 0

        except Exception as e:
            print(f"달성률 기반 순위 계산 실패: {e}")
            return 0

# ================================================================
# LLM 호출 함수들
# ================================================================

def call_llm_for_growth_analysis(basic_info: Dict, performance_data: Dict, 
                                peer_talk_data: Dict, fourp_data: Dict, 
                                collaboration_data: Dict) -> Dict:
    """성장 분석을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "직원")
    cl = basic_info.get("cl", "CL2")
    position = basic_info.get("position", "직책 정보 없음")
    
    # 4P 점수 추출 및 분석
    fourp_scores = {
        "passionate": fourp_data.get("passionate", {}).get("score", 3.0),
        "proactive": fourp_data.get("proactive", {}).get("score", 3.0), 
        "professional": fourp_data.get("professional", {}).get("score", 3.0),
        "people": fourp_data.get("people", {}).get("score", 3.0)
    }
    
    # 4P 최고/최저 영역 계산
    max_4p = max(fourp_scores.keys(), key=lambda k: fourp_scores[k])
    min_4p = min(fourp_scores.keys(), key=lambda k: fourp_scores[k])
    
    system_prompt = """
    당신은 데이터 기반 성장 컨설턴트입니다. 
    직원의 구체적 데이터를 분석하여 실행 가능한 성장 방안을 제시해야 합니다.
    
    ⚠️ 중요 원칙:
    1. 일반적이거나 뻔한 제안 금지 ("팀워크 강화", "소통 개선" 등)
    2. 데이터에 근거한 구체적 분석 필수
    3. 바로 실행 가능한 액션 아이템만 제시
    4. CL/직무별 현실적 발전 경로 고려
    
    분석 방법:
    - 성장 포인트: 4P 최고점수 영역을 활용한 구체적 발전 방향
    - 보완 영역: 4P 최저점수 + Peer Talk 우려 → 명확한 스킬/행동 개선점
    - 추천 활동: 3개월 내 실행 가능한 구체적 액션 (교육명, 프로젝트명, 구체적 행동)
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    CL: {cl} (CL3=시니어급, CL2=중간급, CL1=주니어급)
    직책: {position}
    </직원 정보>

    <성과 데이터 분석>
    달성률: {performance_data.get('ai_achievement_rate', 0)}% 
    → {'목표 초과달성' if performance_data.get('ai_achievement_rate', 0) >= 100 else '목표 미달성'}
    기여도: {performance_data.get('contribution_rate', 0)}%
    팀 내 달성률 순위: {performance_data.get('ranking', 0)}위
    → {'상위권' if performance_data.get('ranking', 5) <= 3 else '중위권' if performance_data.get('ranking', 5) <= 5 else '하위권'}
    </성과 데이터 분석>

    <4P 점수 분석>
    Passionate: {fourp_scores['passionate']}점 {'(강점)' if max_4p == 'passionate' else '(보완)' if min_4p == 'passionate' else '(보통)'}
    Proactive: {fourp_scores['proactive']}점 {'(강점)' if max_4p == 'proactive' else '(보완)' if min_4p == 'proactive' else '(보통)'}
    Professional: {fourp_scores['professional']}점 {'(강점)' if max_4p == 'professional' else '(보완)' if min_4p == 'professional' else '(보통)'}
    People: {fourp_scores['people']}점 {'(강점)' if max_4p == 'people' else '(보완)' if min_4p == 'people' else '(보통)'}
    
    최고 강점: {max_4p}({fourp_scores[max_4p]}점)
    최대 보완점: {min_4p}({fourp_scores[min_4p]}점)
    </4P 점수 분석>

    <Peer Talk 핵심 인사이트>
    강점 키워드: {peer_talk_data.get('strengths', '정보 없음')}
    우려 키워드: {peer_talk_data.get('concerns', '정보 없음')}
    협업 관찰: {peer_talk_data.get('collaboration_observations', '정보 없음')}
    </Peer Talk 핵심 인사이트>

    <협업 패턴 분석>
    협업률: {collaboration_data.get('collaboration_rate', 0)}% 
    → {'과도한 협업' if collaboration_data.get('collaboration_rate', 0) >= 90 else '적정 협업' if collaboration_data.get('collaboration_rate', 0) >= 70 else '협업 부족'}
    팀 역할: {collaboration_data.get('team_role', '정보 없음')}
    협업 편중도: {collaboration_data.get('collaboration_bias', '보통')}
    </협업 패턴 분석>

    위 데이터를 바탕으로 다음 기준에 따라 분석하세요:

    📈 성장 포인트 (2-3개):
    - {max_4p} 강점({fourp_scores[max_4p]}점)을 활용한 구체적 발전 방향
    - 성과 데이터 기반 검증된 역량 영역
    - 예: "복잡한 기술 문제 해결 능력을 활용한 아키텍처 설계 역할 확대"

    🎯 보완 영역 (1-2개):  
    - {min_4p} 영역({fourp_scores[min_4p]}점) 개선을 위한 명확한 스킬/행동
    - Peer Talk 우려사항 기반 구체적 개선점
    - 예: "코드리뷰 시 설명 방식 개선을 통한 지식 전달 스킬 향상"

    🚀 추천 활동 (3개):
    - 3개월 내 실행 가능한 구체적 액션
    - {cl} + {position} 레벨에 적합한 현실적 활동
    - 예: "사내 기술 세미나 발표 1회 진행", "타팀과의 API 설계 협업 프로젝트 참여"

    JSON 응답:
    {{
        "growth_points": [
            "{max_4p} 강점을 활용한 구체적이고 실행 가능한 발전 방향 1",
            "성과 데이터 기반 검증된 역량을 확장하는 방향 2"
        ],
        "improvement_areas": [
            "{min_4p} 영역 개선을 위한 명확하고 측정 가능한 행동 1",
            "Peer Talk 우려사항 해결을 위한 구체적 스킬 향상 2"
        ],
        "recommended_activities": [
            "{cl} {position}에게 적합한 3개월 내 실행 가능한 구체적 액션 1",
            "측정 가능한 결과를 낼 수 있는 구체적 활동 2", 
            "협업/성과 개선에 직접 도움이 되는 실행 가능한 액션 3"
        ]
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        return json.loads(json_output)
        
    except Exception as e:
        print(f"성장 분석 LLM 호출 실패: {e}")
        return {
            "growth_points": ["데이터 분석 중 오류 발생"],
            "improvement_areas": ["데이터 분석 중 오류 발생"], 
            "recommended_activities": ["데이터 분석 중 오류 발생"]
        }

def call_llm_for_focus_coaching_analysis(peer_talk_data: Dict, performance_data: Dict, 
                                       collaboration_data: Dict, fourp_data: Dict) -> Dict:
    """집중 코칭 필요성 분석을 위한 LLM 호출"""
    
    system_prompt = """
    당신은 HR 전문가입니다. 직원의 데이터를 분석하여 집중 코칭이 필요한지 판단해주세요.
    
    집중 코칭 필요 기준:
    1. 성과 이슈: 달성률 70% 미만, 팀 내 하위권
    2. 협업 이슈: 협업률 60% 미만, Peer Talk 심각한 우려사항 2개 이상
    3. 태도 이슈: People 점수 3.0 미만, 부정적 피드백 다수
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <분석 데이터>
    달성률: {performance_data.get('ai_achievement_rate', 0)}%
    달성률 순위: {performance_data.get('ranking', 0)}위
    협업률: {collaboration_data.get('collaboration_rate', 0)}%
    People 점수: {fourp_data.get('people', {}).get('score', 3.0)}점
    
    Peer Talk 우려사항: {peer_talk_data.get('concerns', '없음')}
    협업 관찰: {peer_talk_data.get('collaboration_observations', '없음')}
    </분석 데이터>

    JSON 응답:
    {{
        "focus_coaching_needed": true/false,
        "issue_summary": "핵심 이슈 요약 (집중 코칭 필요한 경우만)",
        "root_cause_analysis": "근본 원인 분석",
        "risk_factors": "리스크 요소 및 점검 포인트", 
        "coaching_plan": "구체적인 집중 코칭 계획"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        return json.loads(json_output)
        
    except Exception as e:
        print(f"집중 코칭 분석 LLM 호출 실패: {e}")
        return {
            "focus_coaching_needed": False,
            "issue_summary": "",
            "root_cause_analysis": "",
            "risk_factors": "", 
            "coaching_plan": ""
        }

def call_llm_for_individual_result(basic_info: Dict, growth_analysis: Dict, 
                                 performance_data: Dict, report_type: str) -> Dict:
    """개인용 결과 생성을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "")
    
    system_prompt = """
    당신은 직원 개인에게 성장 피드백을 제공하는 HR 전문가입니다.
    격려하고 동기부여하는 톤으로 개인 친화적인 피드백을 작성해주세요.
    
    작성 원칙:
    - "당신의", "귀하의" 등 개인 대상 표현 사용
    - 긍정적이고 건설적인 표현
    - 구체적인 개선 방안 제시
    - 격려와 동기부여 포함
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    period_text = "분기" if report_type == "quarterly" else "연간"
    
    human_prompt = f"""
    <기본 정보>
    이름: {emp_name}
    평가 유형: {period_text}
    </기본 정보>

    <성장 분석 결과>
    성장 포인트: {growth_analysis.get('growth_points', [])}
    보완 영역: {growth_analysis.get('improvement_areas', [])}
    추천 활동: {growth_analysis.get('recommended_activities', [])}
    </성장 분석 결과>

    <성과 데이터>
    달성률: {performance_data.get('ai_achievement_rate', 0)}%
    기여도: {performance_data.get('contribution_rate', 0)}%
    </성과 데이터>

    JSON 응답:
    {{
        "growth_points": [
            "당신의 강점을 개인 친화적 톤으로 표현한 항목 1",
            "강점 항목 2",
            "강점 항목 3"
        ],
        "improvement_areas": [
            "발전 가능성이 큰 영역을 건설적 톤으로 표현한 항목 1", 
            "보완 영역 2"
        ],
        "recommended_activities": [
            "구체적이고 실행 가능한 추천 활동 1",
            "추천 활동 2",
            "추천 활동 3"
        ]
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        return json.loads(json_output)
        
    except Exception as e:
        print(f"개인용 결과 생성 LLM 호출 실패: {e}")
        return {
            "growth_points": ["결과 생성 중 오류 발생"],
            "improvement_areas": ["결과 생성 중 오류 발생"],
            "recommended_activities": ["결과 생성 중 오류 발생"]
        }

def call_llm_for_overall_comment(basic_info: Dict, performance_data: Dict, 
                                peer_talk_data: Dict, fourp_data: Dict, 
                                collaboration_data: Dict, growth_analysis: Dict,
                                module7_score_data: Dict, module9_final_data: Dict,
                                report_type: str) -> str:
    """전체 레포트 종합 총평 생성을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "")
    cl = basic_info.get("cl", "CL2")
    position = basic_info.get("position", "직책 정보 없음")
    
    # 4P 점수 추출
    fourp_scores = {
        "passionate": fourp_data.get("passionate", {}).get("score", 3.0),
        "proactive": fourp_data.get("proactive", {}).get("score", 3.0), 
        "professional": fourp_data.get("professional", {}).get("score", 3.0),
        "people": fourp_data.get("people", {}).get("score", 3.0)
    }
    
    fourp_avg = sum(fourp_scores.values()) / len(fourp_scores)
    
    system_prompt = """
    당신은 종합 성과 평가 전문가입니다.
    직원의 모든 평가 결과를 종합하여 전체적인 총평을 작성해주세요.
    
    총평 작성 원칙:
    1. 개인 친화적이고 격려하는 톤 사용
    2. 모든 모듈의 결과를 균형있게 반영
    3. 구체적 성과와 데이터 언급
    4. 향후 성장 방향 제시
    5. 250-300자 분량
    
    반드시 일반적인 문구("열심히 하세요", "앞으로도 화이팅")는 피하고
    구체적인 성과와 개선 방향을 포함해주세요.
    
    결과는 문자열로만 응답하세요.
    """
    
    period_text = "분기" if report_type == "quarterly" else "연간"
    
    # 점수 정보 (연말만)
    score_summary = ""
    if report_type == "annual":
        if module7_score_data.get("score"):
            team_score = module7_score_data.get("score", 0)
            score_summary += f"팀 내 정규화 {team_score}점"
        
        if module9_final_data.get("score"):
            final_score = module9_final_data.get("score", 0)
            ranking = module9_final_data.get("ranking", 0)
            score_summary += f", 부문 정규화 후 최종 {final_score}점(팀 내 {ranking}위)"
    
    human_prompt = f"""
    <종합 평가 데이터>
    직원: {emp_name}({cl} {position})
    평가 기간: {period_text}
    
    📊 성과 결과 (모듈 2):
    - 달성률: {performance_data.get('ai_achievement_rate', 0)}%
    - 기여도: {performance_data.get('contribution_rate', 0)}%
    - 팀 내 달성률 순위: {performance_data.get('ranking', 0)}위
    
    🤝 협업 분석 (모듈 3):
    - 협업률: {collaboration_data.get('collaboration_rate', 0)}%
    - 팀 역할: {collaboration_data.get('team_role', '정보 없음')}
    - 협업 편중도: {collaboration_data.get('collaboration_bias', '보통')}
    
    👥 Peer Talk (모듈 4):
    - 강점: {peer_talk_data.get('strengths', '정보 없음')}
    - 우려사항: {peer_talk_data.get('concerns', '정보 없음')}
    
    🎯 4P 평가 (모듈 6):
    - Passionate: {fourp_scores['passionate']}점
    - Proactive: {fourp_scores['proactive']}점  
    - Professional: {fourp_scores['professional']}점
    - People: {fourp_scores['people']}점
    - 평균: {fourp_avg:.1f}점
    
    {f'📈 점수 평가 (모듈 7,9): {score_summary}' if score_summary else ''}
    
    🚀 성장 제안 (모듈 10):
    - 성장 포인트: {len(growth_analysis.get('growth_points', []))}개
    - 보완 영역: {len(growth_analysis.get('improvement_areas', []))}개
    - 추천 활동: {len(growth_analysis.get('recommended_activities', []))}개
    </종합 평가 데이터>

    위 모든 결과를 종합하여 {emp_name}님께 드리는 {period_text} 종합 총평을 작성해주세요.
    구체적인 성과 수치와 강점을 언급하고, 향후 발전 방향을 제시하는 격려의 메시지로 작성해주세요.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        content = response.content if isinstance(response.content, str) else str(response.content)
        return content.strip()
        
    except Exception as e:
        print(f"종합 총평 LLM 호출 실패: {e}")
        return f"{emp_name}님의 {period_text} 종합 총평 생성 중 오류가 발생했습니다."

def call_llm_for_manager_result(basic_info: Dict, growth_analysis: Dict, 
                              performance_data: Dict, collaboration_data: Dict,
                              focus_coaching_analysis: Dict, focus_coaching_needed: bool) -> Dict:
    """팀장용 결과 생성을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "")
    emp_no = basic_info.get("emp_no", "")
    
    system_prompt = """
    당신은 팀장에게 팀원 관리 정보를 제공하는 HR 전문가입니다.
    객관적이고 분석적인 톤으로 관리자 관점의 코칭 정보를 작성해주세요.
    
    작성 원칙:
    - "해당 직원의", "○○○님의" 등 관리 대상 표현 사용
    - 팀 운영과 인사 관리 포커스
    - 객관적이고 사실 기반의 분석적 표현
    - 의사결정 지원 정보 제공
    
    ⚠️ 중요: 반드시 제공된 JSON 구조를 그대로 사용하세요.
    - emp_no와 name 필드는 정확히 제공된 값으로 설정
    - JSON 구조를 변경하지 마세요
    - 하나의 직원에 대한 하나의 general_coaching 항목만 생성하세요
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <직원 정보>
    사번: {emp_no}
    이름: {emp_name}
    </직원 정보>

    <성장 분석 결과>
    성장 포인트: {growth_analysis.get('growth_points', [])}
    보완 영역: {growth_analysis.get('improvement_areas', [])}
    추천 활동: {growth_analysis.get('recommended_activities', [])}
    </성장 분석 결과>

    <성과 및 협업 데이터>
    달성률: {performance_data.get('ai_achievement_rate', 0)}%
    기여도: {performance_data.get('contribution_rate', 0)}%
    달성률 순위: {performance_data.get('ranking', 0)}위
    협업률: {collaboration_data.get('collaboration_rate', 0)}%
    팀 역할: {collaboration_data.get('team_role', '정보 없음')}
    </성과 및 협업 데이터>

    <집중 코칭 필요성>
    집중 코칭 필요: {focus_coaching_needed}
    집중 코칭 분석: {focus_coaching_analysis if focus_coaching_needed else '해당 없음'}
    </집중 코칭 필요성>

    위 데이터를 바탕으로 다음 JSON 구조로 정확히 응답하세요:

    {{
        "general_coaching": [
            {{
                "emp_no": "{emp_no}",
                "name": "{emp_name}",
                "strengths": "핵심 강점을 관리자 관점으로 요약",
                "improvement_points": "성장 보완점을 관리 관점으로 설명",
                "collaboration_style": "협업 특성 및 팀 내 역할 분석",
                "performance_summary": "성과 기여 요약",
                "next_quarter_coaching": "다음 분기 코칭 제안사항"
            }}
        ],
        "focused_coaching": []
    }}

    ⚠️ 주의사항:
    1. emp_no와 name은 반드시 "{emp_no}"와 "{emp_name}"으로 설정
    2. general_coaching은 하나의 항목만 생성
    3. focused_coaching은 빈 배열로 설정 (별도 처리됨)
    4. JSON 구조를 변경하지 마세요
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        result = json.loads(json_output)
        
        # 🔥 중요: LLM 응답 후 emp_no와 name을 강제로 설정
        if "general_coaching" in result and result["general_coaching"]:
            for coaching_item in result["general_coaching"]:
                coaching_item["emp_no"] = emp_no
                coaching_item["name"] = emp_name
        
        # focused_coaching 처리
        if focus_coaching_needed and focus_coaching_analysis:
            focused_coaching_item = {
                "emp_no": emp_no,
                "name": emp_name,
                "issue_summary": focus_coaching_analysis.get("issue_summary", ""),
                "root_cause_analysis": focus_coaching_analysis.get("root_cause_analysis", ""),
                "risk_factors": focus_coaching_analysis.get("risk_factors", ""),
                "coaching_plan": focus_coaching_analysis.get("coaching_plan", "")
            }
            result["focused_coaching"] = [focused_coaching_item]
        else:
            result["focused_coaching"] = []
            
        return result
        
    except Exception as e:
        print(f"팀장용 결과 생성 LLM 호출 실패: {e}")
        return {
            "general_coaching": [{
                "emp_no": emp_no,
                "name": emp_name,
                "strengths": "결과 생성 중 오류 발생",
                "improvement_points": "결과 생성 중 오류 발생",
                "collaboration_style": "결과 생성 중 오류 발생",
                "performance_summary": "결과 생성 중 오류 발생",
                "next_quarter_coaching": "결과 생성 중 오류 발생"
            }],
            "focused_coaching": []
        }

# ================================================================
# DB 저장 함수들
# ================================================================

def save_individual_result(emp_no: str, period_id: int, report_type: str, 
                         individual_result: Dict, overall_comment: str) -> bool:
    """개인용 결과 + 종합 총평 저장"""
    with engine.connect() as connection:
        try:
            if report_type == "quarterly":
                # feedback_reports 테이블에 저장
                query = text("""
                    UPDATE feedback_reports 
                    SET ai_growth_coaching = :result,
                        overall_comment = :overall_comment
                    WHERE emp_no = :emp_no 
                    AND team_evaluation_id = (
                        SELECT team_evaluation_id 
                        FROM team_evaluations 
                        WHERE period_id = :period_id 
                        AND team_id = (SELECT team_id FROM employees WHERE emp_no = :emp_no)
                    )
                """)
            else:
                # final_evaluation_reports 테이블에 저장
                query = text("""
                    UPDATE final_evaluation_reports 
                    SET ai_growth_coaching = :result,
                        overall_comment = :overall_comment
                    WHERE emp_no = :emp_no 
                    AND team_evaluation_id = (
                        SELECT team_evaluation_id 
                        FROM team_evaluations 
                        WHERE period_id = :period_id 
                        AND team_id = (SELECT team_id FROM employees WHERE emp_no = :emp_no)
                    )
                """)
            
            result = connection.execute(query, {
                "emp_no": emp_no,
                "period_id": period_id,
                "result": json.dumps(individual_result, ensure_ascii=False),
                "overall_comment": overall_comment
            })
            
            connection.commit()
            return result.rowcount > 0
            
        except Exception as e:
            print(f"개인용 결과 저장 실패: {e}")
            connection.rollback()
            return False

def save_manager_result(emp_no: str, period_id: int, manager_result: Dict) -> bool:
    """팀장용 결과 저장 (team_evaluations.ai_team_coaching에 누적)"""
    with engine.connect() as connection:
        try:
            # 기존 team_coaching 데이터 조회
            team_id_query = text("SELECT team_id FROM employees WHERE emp_no = :emp_no")
            team_result = connection.execute(team_id_query, {"emp_no": emp_no}).fetchone()
            
            if not team_result:
                return False
                
            team_id = team_result.team_id
            
            # 기존 ai_team_coaching 데이터 조회
            existing_query = text("""
                SELECT ai_team_coaching 
                FROM team_evaluations 
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            existing_result = connection.execute(existing_query, {
                "team_id": team_id,
                "period_id": period_id
            }).fetchone()
            
            if existing_result and existing_result.ai_team_coaching:
                # 기존 데이터가 있으면 누적
                try:
                    existing_data = json.loads(existing_result.ai_team_coaching)
                except json.JSONDecodeError:
                    existing_data = {"general_coaching": [], "focused_coaching": []}
            else:
                # 기존 데이터가 없으면 새로 생성
                existing_data = {"general_coaching": [], "focused_coaching": []}
            
            # 현재 직원 데이터 추가/업데이트
            # general_coaching에서 기존 직원 데이터 제거
            existing_data["general_coaching"] = [
                gc for gc in existing_data["general_coaching"] 
                if gc.get("emp_no") != emp_no
            ]
            # focused_coaching에서도 기존 직원 데이터 제거
            existing_data["focused_coaching"] = [
                fc for fc in existing_data["focused_coaching"] 
                if fc.get("emp_no") != emp_no
            ]
            
            # 새 데이터 추가
            existing_data["general_coaching"].extend(manager_result["general_coaching"])
            existing_data["focused_coaching"].extend(manager_result["focused_coaching"])
            
            # DB 업데이트
            update_query = text("""
                UPDATE team_evaluations 
                SET ai_team_coaching = :result
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            result = connection.execute(update_query, {
                "team_id": team_id,
                "period_id": period_id,
                "result": json.dumps(existing_data, ensure_ascii=False)
            })
            
            connection.commit()
            return result.rowcount > 0
            
        except Exception as e:
            print(f"팀장용 결과 저장 실패: {e}")
            connection.rollback()
            return False

# ================================================================
# 서브모듈 함수들
# ================================================================

def data_collection_submodule(state: Module10AgentState) -> Module10AgentState:
    """1. 데이터 수집 서브모듈 (종합 총평용 데이터 포함)"""
    
    emp_no = state["emp_no"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    
    try:
        print(f"🔍 모듈 10 데이터 수집 시작: {emp_no} ({report_type})")
        
        # 기본 5개 데이터 소스 수집
        basic_info = fetch_basic_info(emp_no)
        if not basic_info or not basic_info.get("team_id"):
            raise ValueError(f"{emp_no}의 기본 정보 또는 팀 정보를 찾을 수 없습니다.")

        team_id = basic_info["team_id"]

        performance_data = fetch_performance_data(emp_no, period_id, report_type)

        # 달성률 기반으로 실시간 순위 계산
        ranking = calculate_ranking_by_achievement(emp_no, team_id, period_id, report_type)
        performance_data['ranking'] = ranking
        print(f"   📊 달성률 기반 순위 계산 완료: {ranking}위")

        peer_talk_data = fetch_peer_talk_data(emp_no, period_id, report_type)
        fourp_data = fetch_fourp_data(emp_no, period_id, report_type)
        collaboration_data = fetch_collaboration_data(emp_no, period_id)
        
        # 연말 추가 데이터 수집
        module7_score_data = fetch_module7_score_data(emp_no, period_id, report_type)
        module9_final_data = fetch_module9_final_score_data(emp_no, period_id, report_type)
        
        total_sources = 5 + (2 if report_type == "annual" else 0)
        print(f"   ✅ {total_sources}개 데이터 소스 수집 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="데이터 수집 완료")],
            "basic_info": basic_info,
            "performance_data": performance_data,
            "peer_talk_data": peer_talk_data,
            "fourp_data": fourp_data,
            "collaboration_data": collaboration_data,
            "module7_score_data": module7_score_data,
            "module9_final_data": module9_final_data,
            "processing_status": "data_collected"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"데이터 수집 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": [str(e)]
        })
        return updated_state

def growth_analysis_submodule(state: Module10AgentState) -> Module10AgentState:
    """2. 성장 분석 서브모듈"""
    
    try:
        print(f"📊 성장 분석 시작")
        
        growth_analysis = call_llm_for_growth_analysis(
            state["basic_info"],
            state["performance_data"], 
            state["peer_talk_data"],
            state["fourp_data"],
            state["collaboration_data"]
        )
        
        print(f"   ✅ 성장 분석 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="성장 분석 완료")],
            "growth_analysis": growth_analysis,
            "processing_status": "growth_analyzed"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 성장 분석 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"성장 분석 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def focus_coaching_selection_submodule(state: Module10AgentState) -> Module10AgentState:
    """3. 집중 코칭 대상 선정 서브모듈"""
    
    try:
        print(f"🎯 집중 코칭 필요성 분석 시작")
        
        focus_analysis = call_llm_for_focus_coaching_analysis(
            state["peer_talk_data"],
            state["performance_data"],
            state["collaboration_data"],
            state["fourp_data"]
        )
        
        focus_needed = focus_analysis.get("focus_coaching_needed", False)
        print(f"   ✅ 집중 코칭 필요성: {focus_needed}")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"집중 코칭 분석 완료: {focus_needed}")],
            "focus_coaching_needed": focus_needed,
            "focus_coaching_analysis": focus_analysis,
            "processing_status": "focus_analyzed"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 집중 코칭 분석 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"집중 코칭 분석 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def individual_result_generation_submodule(state: Module10AgentState) -> Module10AgentState:
    """4. 개인용 결과 생성 서브모듈 (overall_comment 포함)"""
    
    try:
        print(f"👤 개인용 결과 생성 시작")
        
        # 개인용 성장 제안 결과 생성
        individual_result = call_llm_for_individual_result(
            state["basic_info"],
            state["growth_analysis"],
            state["performance_data"],
            state["report_type"]
        )
        
        # 종합 총평 생성 (모든 모듈 결과 포함)
        overall_comment = call_llm_for_overall_comment(
            state["basic_info"],
            state["performance_data"],
            state["peer_talk_data"],
            state["fourp_data"],
            state["collaboration_data"],
            state["growth_analysis"],
            state["module7_score_data"],
            state["module9_final_data"],
            state["report_type"]
        )
        
        print(f"   ✅ 개인용 결과 + 종합 총평 생성 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="개인용 결과 생성 완료")],
            "individual_growth_result": individual_result,
            "overall_comment": overall_comment,
            "processing_status": "individual_generated"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 개인용 결과 생성 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"개인용 결과 생성 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def manager_result_generation_submodule(state: Module10AgentState) -> Module10AgentState:
    """5. 팀장용 결과 생성 서브모듈"""
    
    try:
        print(f"👨‍💼 팀장용 결과 생성 시작")
        
        manager_result = call_llm_for_manager_result(
            state["basic_info"],
            state["growth_analysis"],
            state["performance_data"],
            state["collaboration_data"],
            state["focus_coaching_analysis"],
            state["focus_coaching_needed"]
        )
        
        print(f"   ✅ 팀장용 결과 생성 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="팀장용 결과 생성 완료")],
            "manager_coaching_result": manager_result,
            "processing_status": "manager_generated"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 팀장용 결과 생성 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"팀장용 결과 생성 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def storage_submodule(state: Module10AgentState) -> Module10AgentState:
    """6. DB 저장 서브모듈 (종합 총평 포함)"""
    
    try:
        print(f"💾 DB 저장 시작")
        
        emp_no = state["emp_no"]
        period_id = state["period_id"]
        report_type = state["report_type"]
        
        # 개인용 결과 + 종합 총평 저장
        individual_saved = save_individual_result(
            emp_no, period_id, report_type, 
            state["individual_growth_result"],
            state["overall_comment"]
        )
        
        # 팀장용 결과 저장
        manager_saved = save_manager_result(
            emp_no, period_id,
            state["manager_coaching_result"]
        )
        
        storage_result = {
            "individual_saved": individual_saved,
            "manager_saved": manager_saved,
            "updated_records": int(individual_saved) + int(manager_saved)
        }
        
        print(f"   ✅ 저장 완료: 개인용({individual_saved}), 팀장용({manager_saved})")
        print(f"   📝 종합 총평 저장: {len(state['overall_comment'])}자")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="DB 저장 완료")],
            "storage_result": storage_result,
            "processing_status": "completed"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"DB 저장 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)],
            "storage_result": {"individual_saved": False, "manager_saved": False, "updated_records": 0}
        })
        return updated_state

# ================================================================
# 워크플로우 생성
# ================================================================

def create_module10_graph():
    """모듈 10 그래프 생성 및 반환"""
    module10_workflow = StateGraph(Module10AgentState)
    
    # 노드 추가 (State 키와 겹치지 않도록 이름 변경)
    module10_workflow.add_node("collect_data", data_collection_submodule)
    module10_workflow.add_node("analyze_growth", growth_analysis_submodule)
    module10_workflow.add_node("select_focus_coaching", focus_coaching_selection_submodule)
    module10_workflow.add_node("generate_individual_result", individual_result_generation_submodule)
    module10_workflow.add_node("generate_manager_result", manager_result_generation_submodule)
    module10_workflow.add_node("store_results", storage_submodule)
    
    # 엣지 정의 (순차 실행)
    module10_workflow.add_edge(START, "collect_data")
    module10_workflow.add_edge("collect_data", "analyze_growth")
    module10_workflow.add_edge("analyze_growth", "select_focus_coaching")
    module10_workflow.add_edge("select_focus_coaching", "generate_individual_result")
    module10_workflow.add_edge("generate_individual_result", "generate_manager_result")
    module10_workflow.add_edge("generate_manager_result", "store_results")
    module10_workflow.add_edge("store_results", END)
    
    return module10_workflow.compile()

# ================================================================
# 실행 함수들
# ================================================================

def run_module10_evaluation(emp_no: str, period_id: int, report_type: str = "quarterly"):
    """모듈 10 개인 성장 및 코칭 분석 실행"""
    
    print(f"🚀 모듈 10 개인 성장 및 코칭 분석 시작: {emp_no} ({report_type})")
    
    # State 정의
    state = Module10AgentState(
        messages=[HumanMessage(content=f"모듈 10 시작: {emp_no}")],
        emp_no=emp_no,
        period_id=period_id,
        report_type=report_type,
        basic_info={},
        performance_data={},
        peer_talk_data={},
        fourp_data={},
        collaboration_data={},
        module7_score_data={},
        module9_final_data={},
        growth_analysis={},
        focus_coaching_needed=False,
        focus_coaching_analysis={},
        individual_growth_result={},
        manager_coaching_result={},
        overall_comment="",
        storage_result={},
        processing_status="started",
        error_messages=[]
    )
    
    # 그래프 생성 및 실행
    module10_graph = create_module10_graph()
    
    try:
        result = module10_graph.invoke(state)
        
        print("✅ 모듈 10 개인 성장 및 코칭 분석 완료!")
        print(f"📊 처리 상태: {result.get('processing_status')}")
        
        if result.get('storage_result'):
            storage = result['storage_result']
            print(f"💾 저장 결과: {storage.get('updated_records', 0)}개 레코드 업데이트")
            
        if result.get('error_messages'):
            print(f"⚠️ 오류 메시지: {result['error_messages']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 모듈 10 실행 실패: {e}")
        return None


def run_team_module10_evaluation(team_id: str, period_id: int, report_type: str = "quarterly"):
    """팀 단위 모듈 10 실행"""
    
    print(f"🚀 팀 단위 모듈 10 실행: {team_id} ({report_type})")
    
    # 팀원 목록 조회
    with engine.connect() as connection:
        query = text("SELECT emp_no, emp_name FROM employees WHERE team_id = :team_id")
        results = connection.execute(query, {"team_id": team_id}).fetchall()
        team_members = [row_to_dict(row) for row in results]
    
    if not team_members:
        print(f"❌ 팀원이 없습니다: {team_id}")
        return None
    
    print(f"📋 대상 팀원: {len(team_members)}명")
    
    results = {}
    success_count = 0
    
    for member in team_members:
        emp_no = member["emp_no"]
        emp_name = member["emp_name"]
        
        print(f"\n{'='*30}")
        print(f"처리 중: {emp_name}({emp_no})")
        
        result = run_module10_evaluation(emp_no, period_id, report_type)
        results[emp_no] = result
        
        if result and result.get('processing_status') == 'completed':
            success_count += 1
    
    print(f"\n🎯 팀 단위 실행 완료:")
    print(f"   성공: {success_count}/{len(team_members)}명")
    
    return results

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def test_module10(emp_no: Optional[str] = None, period_id: int = 4, report_type: str = "quarterly"):
    """모듈 10 테스트"""
    if not emp_no:
        # 테스트용 직원 자동 선택
        with engine.connect() as connection:
            query = text("""
                SELECT e.emp_no, e.emp_name 
                FROM employees e
                JOIN final_evaluation_reports fer ON e.emp_no = fer.emp_no
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE te.period_id = :period_id
                LIMIT 1
            """)
            result = connection.execute(query, {"period_id": period_id}).fetchone()
            
            if result:
                emp_no = result.emp_no
                print(f"🧪 테스트 직원 자동 선택: {result.emp_name}({emp_no})")
            else:
                print("❌ 테스트할 직원이 없습니다")
                return
    
    if emp_no is None:
        print("❌ emp_no가 None입니다")
        return
        
    return run_module10_evaluation(emp_no, period_id, report_type)

def get_teams_with_data(period_id: int = 4) -> List[str]:
    """데이터가 있는 팀 목록 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT e.team_id
            FROM employees e
            JOIN final_evaluation_reports fer ON e.emp_no = fer.emp_no
            JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
            WHERE te.period_id = :period_id
            ORDER BY e.team_id
        """)
        results = connection.execute(query, {"period_id": period_id}).fetchall()
        return [row.team_id for row in results]

def display_result_summary(result: Dict):
    """결과 요약 출력"""
    if not result:
        print("❌ 결과가 없습니다.")
        return
    
    emp_no = result.get('emp_no', 'Unknown')
    status = result.get('processing_status', 'Unknown')
    
    print(f"\n📊 {emp_no} 결과 요약:")
    print(f"   상태: {status}")
    
    if status == 'completed':
        individual_result = result.get('individual_growth_result', {})
        manager_result = result.get('manager_coaching_result', {})
        overall_comment = result.get('overall_comment', '')
        
        print(f"   성장 포인트: {len(individual_result.get('growth_points', []))}개")
        print(f"   보완 영역: {len(individual_result.get('improvement_areas', []))}개")
        print(f"   추천 활동: {len(individual_result.get('recommended_activities', []))}개")
        print(f"   집중 코칭: {'필요' if result.get('focus_coaching_needed') else '불필요'}")
        print(f"   종합 총평: {len(overall_comment)}자")
        
        storage = result.get('storage_result', {})
        print(f"   저장 상태: {storage.get('updated_records', 0)}개 레코드")
    
    if result.get('error_messages'):
        print(f"   ⚠️ 오류: {len(result['error_messages'])}건")

# ================================================================
# 데이터 정리 함수들
# ================================================================

def clean_ai_team_coaching_data(team_id: str, period_id: int):
    """기존 ai_team_coaching 데이터에서 빈 emp_no 항목들을 제거"""
    with engine.connect() as connection:
        try:
            # 기존 데이터 조회
            query = text("""
                SELECT ai_team_coaching 
                FROM team_evaluations 
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            result = connection.execute(query, {
                "team_id": team_id,
                "period_id": period_id
            }).fetchone()
            
            if not result or not result.ai_team_coaching:
                print(f"정리할 데이터가 없습니다: {team_id}")
                return
            
            try:
                data = json.loads(result.ai_team_coaching)
            except json.JSONDecodeError:
                print(f"JSON 파싱 실패: {team_id}")
                return
            
            # 빈 emp_no 항목들 제거
            original_general_count = len(data.get("general_coaching", []))
            original_focused_count = len(data.get("focused_coaching", []))
            
            data["general_coaching"] = [
                item for item in data.get("general_coaching", [])
                if item.get("emp_no") and item.get("emp_no").strip()
            ]
            
            data["focused_coaching"] = [
                item for item in data.get("focused_coaching", [])
                if item.get("emp_no") and item.get("emp_no").strip()
            ]
            
            cleaned_general_count = len(data.get("general_coaching", []))
            cleaned_focused_count = len(data.get("focused_coaching", []))
            
            # 업데이트
            update_query = text("""
                UPDATE team_evaluations 
                SET ai_team_coaching = :result
                WHERE team_id = :team_id AND period_id = :period_id
            """)
            
            connection.execute(update_query, {
                "team_id": team_id,
                "period_id": period_id,
                "result": json.dumps(data, ensure_ascii=False)
            })
            
            connection.commit()
            
            print(f"✅ 데이터 정리 완료: {team_id}")
            print(f"   general_coaching: {original_general_count} → {cleaned_general_count}")
            print(f"   focused_coaching: {original_focused_count} → {cleaned_focused_count}")
            
        except Exception as e:
            print(f"❌ 데이터 정리 실패: {e}")
            connection.rollback()

def clean_all_team_coaching_data(period_id: int = 4):
    """모든 팀의 ai_team_coaching 데이터 정리"""
    teams = get_teams_with_data(period_id)
    
    print(f"🧹 {len(teams)}개 팀의 데이터 정리 시작...")
    
    for team_id in teams:
        clean_ai_team_coaching_data(team_id, period_id)
    
    print("✅ 모든 팀 데이터 정리 완료!")

# ================================================================
# 메인 실행 부분
# ================================================================

if __name__ == "__main__":
    print("🚀 모듈 10: 개인 성장 및 코칭 모듈 준비 완료!")
    print("\n🔥 주요 기능:")
    print("✅ 7개 데이터 소스 통합 분석 (기본 5개 + 연말 2개)")
    print("✅ LLM 기반 성장 포인트 및 보완 영역 추출")
    print("✅ 집중 코칭 대상 자동 선정")
    print("✅ 개인용/팀장용 차별화된 결과 생성")
    print("✅ 종합 총평 생성 (모든 모듈 결과 통합)")
    print("✅ JSON + TEXT 형태로 DB 저장")
    
    print("\n📋 실행 명령어:")
    print("1. run_module10_evaluation('E002', 4, 'quarterly')      # 개별 실행 (분기)")
    print("2. run_module10_evaluation('E002', 4, 'annual')        # 개별 실행 (연말)")
    print("3. run_team_module10_evaluation('1', 4, 'annual')      # 팀 단위 실행")
    print("4. test_module10()                                     # 테스트 실행")
    print("5. get_teams_with_data(4)                              # 데이터 있는 팀 조회")
    print("6. display_result_summary(result)                      # 결과 요약 출력")
    print("7. clean_ai_team_coaching_data('1', 4)                # 특정 팀 데이터 정리")
    print("8. clean_all_team_coaching_data(4)                    # 모든 팀 데이터 정리")
    
    print("\n📊 DB 저장 구조:")
    print("- ai_growth_coaching: 성장 제안 3개 항목 (JSON)")
    print("- overall_comment: 전체 레포트 종합 총평 (TEXT)")
    print("- ai_team_coaching: 팀장용 코칭 정보 (JSON)")
    
    print("\n🎯 필요한 DB 스키마:")
    print("ALTER TABLE feedback_reports ADD COLUMN overall_comment TEXT;")
    print("ALTER TABLE final_evaluation_reports ADD COLUMN overall_comment TEXT;")
    
    print("\n🔧 수정 사항:")
    print("✅ LLM 응답 후 emp_no/name 강제 설정")
    print("✅ JSON 구조 명확화")
    print("✅ 빈 emp_no 데이터 정리 함수 추가")
    
    # 자동 테스트 (필요시 주석 해제)
    # test_module10()

    run_module10_evaluation('E002', 4, 'annual')