import os
from typing import TypedDict, Literal, Optional, List, Dict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from sqlalchemy import create_engine, text
from langgraph.graph import StateGraph, END, START

from dotenv import load_dotenv
load_dotenv()

# 기존 설정 가져오기 (상대 경로로 시도)
try:
    from config.settings import settings
except ImportError:
    # config 모듈이 없는 경우 기본값 사용
    class DefaultSettings:
        DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/skoro_db"
    settings = DefaultSettings()

# =============================================================================
# ChatState 정의
# =============================================================================

class ChatState(TypedDict):
    # 사용자 정보
    user_id: str
    chat_mode: Literal["default", "appeal_to_manager"]
    user_input: str
    role: Literal["MANAGER", "MEMBER"]
    team_id: str
    appeal_complete: Optional[bool]
    
    # 검색된 문서들
    retrieved_docs: List[Dict]
    
    # 대화 히스토리 관리
    qna_dialog_log: List[str]
    dialog_log: List[str]
    
    # 플래그 및 결과
    summary_draft: Optional[str]
    llm_response: Optional[str]

# =============================================================================
# 챗봇 설정 (완전한 .env 기반)
# =============================================================================

class ChatbotConfig:
    def __init__(self):
        # .env 파일에서 필요한 값들 로드
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        self.index_reports_name = os.getenv("PINECONE_INDEX_REPORTS", "skoro-chatbot")
        self.index_policy_name = os.getenv("PINECONE_INDEX_POLICY", "eval-policy-docs")
        self.index_appeals_name = os.getenv("PINECONE_INDEX_APPEALS", "appeal-cases")
        
        print(f"🔧 설정 로드 중...")
        print(f"  - Pinecone API Key: {'설정됨' if self.pinecone_api_key else '없음'}")
        print(f"  - 임베딩 모델: {self.embedding_model_name}")
        print(f"  - 리포트 인덱스: {self.index_reports_name}")
        
        # 필수값 체크 (개발용으로 완화)
        if not self.pinecone_api_key:
            print("⚠️ PINECONE_API_KEY가 .env 파일에 설정되지 않았습니다.")
            print("💡 테스트를 위해 기본값을 사용합니다.")
            self.pinecone_api_key = "test-key"  # 테스트용
        
        try:
            # LLM 설정 (OPENAI_API_KEY는 ChatOpenAI가 자동으로 찾음)
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
            print("✅ LLM 초기화 완료")
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {str(e)}")
            self.llm = None
        
        try:
            # 임베딩 모델
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            print("✅ 임베딩 모델 초기화 완료")
        except Exception as e:
            print(f"⚠️ 임베딩 모델 초기화 실패: {str(e)}")
            self.embedding_model = None
        
        try:
            # Pinecone 설정
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index_reports = self.pc.Index(self.index_reports_name)
            self.index_policy = self.pc.Index(self.index_policy_name)
            self.index_appeals = self.pc.Index(self.index_appeals_name)
            print("✅ Pinecone 초기화 완료")
        except Exception as e:
            print(f"⚠️ Pinecone 초기화 실패: {str(e)}")
            self.index_reports = None
            self.index_policy = None
            self.index_appeals = None
        
        try:
            # DB 설정
            self.engine = create_engine(settings.DATABASE_URL)
            print("✅ DB 연결 초기화 완료")
        except Exception as e:
            print(f"⚠️ DB 초기화 실패: {str(e)}")
            self.engine = None

# =============================================================================
# 전역 객체 초기화
# =============================================================================

print("🚀 챗봇 시스템 초기화 중...")
config = ChatbotConfig()

# 전역에서 사용할 객체들
llm = config.llm
embedding_model = config.embedding_model
index_reports = config.index_reports
index_policy = config.index_policy
index_appeals = config.index_appeals
engine = config.engine

# =============================================================================
# 유틸리티 함수들
# =============================================================================

def get_user_metadata(user_id: str) -> dict:
    """사용자 메타데이터 조회"""
    if not engine:
        print("⚠️ DB 연결이 없어 기본값을 사용합니다.")
        return {"role": "MEMBER", "team_id": "default", "team_name": "default"}
    
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT e.role, e.team_id, t.team_name
                FROM employees e
                JOIN teams t ON e.team_id = t.team_id
                WHERE e.emp_no = :user_id
            """)
            result = conn.execute(query, {"user_id": user_id}).fetchone()
            if result:
                return {
                    "role": result.role,
                    "team_id": result.team_id,
                    "team_name": result.team_name
                }
            else:
                return {"role": "MEMBER", "team_id": "default", "team_name": "default"}
    except Exception as e:
        print(f"❌ 사용자 정보 조회 실패: {str(e)}")
        return {"role": "MEMBER", "team_id": "default", "team_name": "default"}

# =============================================================================
# 세션 관리자
# =============================================================================

class SessionManager:
    """메모리 기반 세션 관리"""
    
    def __init__(self):
        self.sessions = {}
        print("✅ 세션 매니저 초기화 완료")
    
    def get_session_state(self, user_id: str, chat_mode: str) -> Dict:
        session_key = f"{user_id}_{chat_mode}"
        return self.sessions.get(session_key, {})
    
    def save_session_state(self, user_id: str, chat_mode: str, state: Dict):
        session_key = f"{user_id}_{chat_mode}"
        self.sessions[session_key] = {
            "qna_dialog_log": state.get("qna_dialog_log", []),
            "dialog_log": state.get("dialog_log", []),
            "updated_at": datetime.now().isoformat()
        }
    
    def clear_session(self, user_id: str, chat_mode: str):
        session_key = f"{user_id}_{chat_mode}"
        if session_key in self.sessions:
            del self.sessions[session_key]

session_manager = SessionManager()

# =============================================================================
# LangGraph 노드들
# =============================================================================

def initialize_state(state: ChatState) -> ChatState:
    """상태 초기화"""
    print(f"🔧 상태 초기화: {state['user_id']}")
    
    user_metadata = get_user_metadata(state["user_id"])
    
    state["role"] = user_metadata["role"]
    state["team_id"] = user_metadata["team_id"]
    
    if "retrieved_docs" not in state:
        state["retrieved_docs"] = []
    if "qna_dialog_log" not in state:
        state["qna_dialog_log"] = []
    if "dialog_log" not in state:
        state["dialog_log"] = []
    if "appeal_complete" not in state:
        state["appeal_complete"] = False
    
    return state

def route_chat_mode(state: ChatState) -> str:
    """라우팅"""
    if state["chat_mode"] == "appeal_to_manager":
        if state.get("appeal_complete", False):
            return "summary_generator"
        else:
            return "appeal_dialogue"
    else:
        return "qna_agent"
    

def qna_agent_node(state: ChatState) -> ChatState:
    """QnA 에이전트 - 간소화된 버전"""
    print("💬 QnA 에이전트 실행 중...")
    
    query = state["user_input"]
    user_metadata = {
        "emp_no": state["user_id"],
        "role": state["role"],
        "team_name": state["team_id"]
    }

    # 이번 사용자 발화 기록
    state["qna_dialog_log"].append(f"사용자: {query}")

    # RAG 검색이 가능한지 확인
    if not (embedding_model and index_reports and index_policy and index_appeals):
        print("⚠️ RAG 검색 환경이 완전하지 않아 기본 응답을 사용합니다.")
        simple_response = f"현재 시스템 점검 중입니다. '{query}'에 대한 상세한 답변은 잠시 후 제공하겠습니다."
        state["retrieved_docs"] = []
        state["llm_response"] = simple_response
        state["qna_dialog_log"].append(f"챗봇: {simple_response}")
        return state

    try:
        # ✅ 통합 검색 함수 사용 (중복 제거!)
        from .rag_retriever import search_for_qna
        
        retrieved_docs = search_for_qna(query, user_metadata)

        # 컨텍스트 구성
        context = "\n\n".join(
            f"[출처: {doc['source']}] {doc['content'][:500]}" for doc in retrieved_docs[:5]
        )

        # 이전 대화(최근 10턴) 포함하여 프롬프트 구성
        conversation_context = "\n".join(state["qna_dialog_log"][-10:])

        # LLM 프롬프트 구성
        if context.strip():
            prompt = f"""
당신은 SK그룹 성과평가 기준에 기반한 AI 상담 챗봇입니다.
사용자의 질문에 대해 아래 문서를 참고하여 정확하고 신뢰성 있게 답변해주세요.

[이전 대화]
{conversation_context}

[사용자 질문]
{query}

[참고 문서]
{context}

[응답 형식]
- 가능한 객관적 근거를 포함하여 설명
- 출처(리포트, 정책, 이의제기 등)에 따라 근거 다를 경우 간략히 언급
- 기준이 불명확하면 "정책 문서에 정의되어 있지 않음"이라고 말함
- user_id(emp_no)에 해당하는 직원, 즉 본인 외에 다른 직원은 익명으로 작성해주세요.
"""
        else:
            # 검색 결과가 없을 때
            prompt = f"""
당신은 SK그룹 성과평가 상담 챗봇입니다.
사용자의 질문에 대해 일반적인 성과평가 가이드라인을 바탕으로 답변해주세요.

[이전 대화]
{conversation_context}

[사용자 질문]
{query}

관련 문서를 찾을 수 없어 일반적인 가이드라인으로 답변드리겠습니다.
친절하고 도움이 되는 답변을 해주세요.
"""

        # LLM 호출
        if llm:
            print("🤖 LLM 응답 생성 중...")
            try:
                llm_response = llm.predict(prompt)
                print(f"✅ LLM 응답 생성 완료 ({len(llm_response)}자)")
            except Exception as e:
                print(f"⚠️ LLM 호출 실패: {str(e)}")
                llm_response = f"죄송합니다. '{query}'에 대한 답변을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        else:
            llm_response = f"현재 시스템 점검 중입니다. '{query}'에 대한 답변은 잠시 후 제공하겠습니다."

        # 상태 업데이트
        state["retrieved_docs"] = retrieved_docs
        state["llm_response"] = llm_response.strip()
        state["qna_dialog_log"].append(f"챗봇: {llm_response.strip()}")

        print("✅ QnA 에이전트 완료")

    except ImportError as e:
        print(f"⚠️ RAG 모듈 import 실패: {str(e)}")
        # 기본 응답 fallback (기존과 동일)
        
    except Exception as e:
        print(f"❌ QnA 에이전트 처리 중 오류: {str(e)}")
        # 에러 처리 (기존과 동일)

    return state   



def appeal_dialogue_node(state: ChatState) -> ChatState:
    """이의제기 대화 노드 - 간소화된 버전"""
    print("💬 이의제기 대화 노드 실행 중...")
    
    query = state["user_input"]
    user_metadata = {
        "emp_no": state["user_id"],
        "role": state["role"],
        "team_name": state["team_id"]
    }

    # 현재 사용자 입력 추가
    state["dialog_log"].append(f"사용자: {query}")

    # 팩트 체크 키워드 목록
    FACT_TRIGGER_KEYWORDS = [
        "점수", "등급", "기준", "평가", "순위", "컷오프", "평가표", "채점", "근거",
        "기여", "비중", "수치", "비율", "정량", "정성", "성과", "퍼센트", "%",
        "기준치", "초과", "미달", "도달", "충족", "달성", "충분", "부족",
        "어디에", "문서", "기록", "정책", "규정", "왜", "무엇 때문에", "무슨 이유"
    ]

    normalized_query = query.lower().replace(" ", "")
    needs_fact = any(k.replace(" ", "") in normalized_query for k in FACT_TRIGGER_KEYWORDS)

    # RAG 검색이 가능한지 확인
    if not (embedding_model and index_reports):
        print("⚠️ RAG 검색 환경이 완전하지 않아 기본 응답을 사용합니다.")
        simple_response = f"현재 시스템 점검 중입니다. '{query}'에 대한 상세한 답변은 잠시 후 제공하겠습니다."
        state["retrieved_docs"] = []
        state["llm_response"] = simple_response
        state["dialog_log"].append(f"챗봇: {simple_response}")
        return state

    try:
        # RAG 검색 (필요할 때만)
        retrieved_docs = []
        context = "(참고 문서 없음)"
        
        if needs_fact:
            print("🔍 팩트 체크를 위한 문서 검색 중...")
            
            # ✅ 통합 검색 함수 사용 (중복 제거!)
            from .rag_retriever import search_for_appeal
            
            retrieved_docs = search_for_appeal(query, user_metadata)
            
            if retrieved_docs:
                context = "\n\n".join(
                    f"[출처: {doc['source']}] {doc['content'][:500]}" for doc in retrieved_docs[:3]
                )
                print(f"✅ 팩트 체크용 문서 {len(retrieved_docs)}개 검색 완료")
            else:
                print("📝 관련 팩트 문서를 찾지 못했습니다.")
        else:
            print("💭 일반적인 대화 모드 (팩트 체크 불필요)")

        # 대화 히스토리 생성 (최근 5개까지만 사용)
        history = "\n".join(state["dialog_log"][-5:])

        # 프롬프트 구성
        prompt = f"""
당신은 SK 성과관리 AI 챗봇으로서, 사용자가 평가에 대해 느낀 억울함이나 문제의식을 명확히 표현하도록 돕는 역할입니다.
직접적인 정답 제공과 동시에 공감과 질문을 통해 사용자의 입장을 더 끌어내세요.

[대화 이력]
{history}

[사용자 최신 질문]
{query}

[참고 문서]
{context}

[응답 가이드]
- 공감 또는 질문으로 시작
- 사용자가 더 자세히 설명하게 유도
- 부드럽고 친절한 말투 사용
- 문서 기준 내용이 있으면 부드럽게 반영 ("문서 기준으로 보면 ~일 수 있어요.")
- user_id(emp_no)에 해당하는 직원, 즉 본인 외에 다른 직원은 익명으로 작성해주세요.
"""

        # LLM 호출
        if llm:
            print("🤖 LLM 응답 생성 중...")
            try:
                llm_response = llm.predict(prompt)
                print(f"✅ LLM 응답 생성 완료 ({len(llm_response)}자)")
            except Exception as e:
                print(f"⚠️ LLM 호출 실패: {str(e)}")
                llm_response = f"죄송합니다. '{query}'에 대한 답변을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        else:
            llm_response = f"현재 시스템 점검 중입니다. '{query}'에 대한 답변은 잠시 후 제공하겠습니다."

        # 상태 업데이트
        state["dialog_log"].append(f"챗봇: {llm_response.strip()}")
        state["llm_response"] = llm_response.strip()
        state["retrieved_docs"] = retrieved_docs

        print("✅ 이의제기 대화 노드 완료")

    except ImportError as e:
        print(f"⚠️ RAG 모듈 import 실패: {str(e)}")
        # 기본 응답 fallback
        simple_response = "현재 일부 기능이 점검 중입니다. 기본적인 상담은 가능하니 말씀해주세요."
        state["retrieved_docs"] = []
        state["llm_response"] = simple_response
        state["dialog_log"].append(f"챗봇: {simple_response}")
        
    except Exception as e:
        print(f"❌ 이의제기 대화 노드 처리 중 오류: {str(e)}")
        error_response = "죄송합니다. 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        state["retrieved_docs"] = []
        state["llm_response"] = error_response
        state["dialog_log"].append(f"챗봇: {error_response}")

    return state


def summary_generator_node(state: ChatState) -> ChatState:
    """요약 생성 노드 (원본 로직 그대로)"""
    conversation = "\n".join(state["dialog_log"])

    prompt = f"""
당신은 SK그룹 성과평가 시스템의 AI 요약 챗봇입니다.
아래 대화는 한 구성원과 챗봇 간의 이의제기 대화입니다.

이 대화를 바탕으로, **아직 해결되지 않은 의문**이나 **이의제기 내용을 요약**해 주세요.
단, 다음 조건을 반드시 지켜주세요:

[작성 조건]
- 누구인지 식별할 수 없도록 **완전한 익명 표현** 사용 (1인칭, 실명, 직무명, 팀명 금지)
- 욕설, 불만은 정중하게 정제
- **두괄식**으로 요약 (핵심 주장 → 상세 설명)
- 여전히 의문이 남는 내용만 요약
- **수긍한 내용은 절대 포함하지 마세요**
- 문서 기반 사실관계에 대한 근거는 요약하지 않음 (대화 중 수긍한 내용은 제외)
- **의문이 구체적으로 드러나도록** 서술
- 요약문은 예의와 논리를 갖추어야 합니다.

[대화 기록]
{conversation}

[최종 출력]
- 팀장이 쉽게 이해할 수 있도록 구성원이 제기한 핵심 의문을 **정중하고 논리적으로 요약**한 문장만 작성해주세요.
- **구성원의 말투나 개성을 흉내내지 말고**, 객관적인 행정용 보고 스타일로 정리해주세요.
"""

    # LLM 호출
    llm_response = config.llm.predict(prompt)

    state["summary_draft"] = llm_response.strip()
    state["llm_response"] = llm_response.strip()

    return state

# def summary_generator_node(state: ChatState) -> ChatState:
#     """요약 생성"""
#     print("📝 요약 생성 중...")
    
#     conversation = "\n".join(state["dialog_log"])
    
#     # 간단한 요약
#     simple_summary = f"이의제기 요약: 사용자가 제기한 주요 의문사항들을 정리했습니다."
    
#     # LLM 사용 (가능한 경우)
#     if llm:
#         try:
#             prompt = f"""
# 다음 이의제기 대화를 요약해주세요:

# {conversation}

# 객관적이고 정중한 톤으로 핵심 의문사항만 요약해주세요.
# """
#             llm_response = llm.predict(prompt)
#         except Exception as e:
#             print(f"⚠️ LLM 호출 실패: {str(e)}")
#             llm_response = simple_summary
#     else:
#         llm_response = simple_summary
    
#     state["summary_draft"] = llm_response.strip()
#     state["llm_response"] = llm_response.strip()
    
#     return state

# =============================================================================
# LangGraph 워크플로우 생성
# =============================================================================

def create_chatbot_workflow():
    """챗봇 워크플로우 생성"""
    print("🔄 LangGraph 워크플로우 생성 중...")
    
    workflow = StateGraph(ChatState)
    
    # 노드 추가
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("qna_agent", qna_agent_node)
    workflow.add_node("appeal_dialogue", appeal_dialogue_node)
    workflow.add_node("summary_generator", summary_generator_node)
    
    # 시작점 설정
    workflow.add_edge(START, "initialize")
    
    # 조건부 엣지 (라우팅)
    workflow.add_conditional_edges(
        "initialize",
        route_chat_mode,
        {
            "qna_agent": "qna_agent",
            "appeal_dialogue": "appeal_dialogue",
            "summary_generator": "summary_generator"
        }
    )
    
    # 모든 노드는 END로 종료
    workflow.add_edge("qna_agent", END)
    workflow.add_edge("appeal_dialogue", END)
    workflow.add_edge("summary_generator", END)
    
    compiled_workflow = workflow.compile()
    print("✅ LangGraph 워크플로우 생성 완료!")
    
    return compiled_workflow