# =============================================================================
# agent.py - LangGraph 에이전트 및 워크플로우 
# =============================================================================

from typing import TypedDict, Literal, Optional, List, Dict
from datetime import datetime
import warnings
from langgraph.graph import StateGraph, START, END

from .llm_utils import ChatbotConfig, get_user_metadata, analyze_question_intent
from .rag_retriever import UnifiedRAGRetriever, search_documents_with_access_control

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 1. State 정의
# =============================================================================

class ChatState(TypedDict):
    user_id: str
    chat_mode: Literal["default", "appeal_to_manager"]
    user_input: str
    role: Literal["MANAGER", "MEMBER"]
    team_id: str
    appeal_complete: Optional[bool]
    retrieved_docs: List[Dict]
    qna_dialog_log: List[str]
    dialog_log: List[str]
    summary_draft: Optional[str]
    llm_response: Optional[str]

# =============================================================================
# 2. 설정 및 초기화
# =============================================================================

config = ChatbotConfig()
rag_retriever = UnifiedRAGRetriever(config)

# =============================================================================
# 3. LangGraph 노드들
# =============================================================================

def initialize_state(state: ChatState) -> ChatState:
    """상태 초기화 노드"""
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
    """채팅 모드에 따른 라우팅"""
    if state["chat_mode"] == "appeal_to_manager":
        if state.get("appeal_complete", False):
            return "summary_generator"
        else:
            return "appeal_dialogue"
    else:
        return "qna_agent"

def qna_agent_node(state: ChatState) -> ChatState:
    """RAG 검색기를 사용하는 개선된 QnA 에이전트 노드"""
    query = state["user_input"]
    user_metadata = {
        "emp_no": state["user_id"],
        "role": state["role"],
        "team_name": state["team_id"]
    }

    state["qna_dialog_log"].append(f"사용자: {query}")

    # ✅ 문맥 분석 - 이전 대화에서 언급된 키워드 추출
    previous_context = "\n".join(state["qna_dialog_log"][:-1])  # 현재 입력 제외
    
    # 이전 대화에서 중요한 키워드들 추출
    context_keywords = []
    if "점수" in previous_context or "등급" in previous_context:
        context_keywords.extend(["점수", "등급", "평가"])
    if "업무" in previous_context or "달성률" in previous_context:
        context_keywords.extend(["달성률", "성과", "업무"])
    if "팀" in previous_context:
        context_keywords.append("팀")
    
    # ✅ 확장된 검색 쿼리 생성
    enhanced_query = query
    if context_keywords:
        enhanced_query = f"{query} {' '.join(context_keywords[:3])}"
        print(f"🔍 확장된 쿼리: '{enhanced_query}'")

    # 🎯 RAG 검색기를 통한 문서 검색 (권한 제어 자동 적용)
    report_results = search_documents_with_access_control(
        query=enhanced_query, 
        user_metadata=user_metadata, 
        filter_type="report", 
        top_k=5
    )["matches"]

    # 정책 문서 검색 (RAG 검색기 사용)
    try:
        policy_results = rag_retriever.search_policies(enhanced_query, top_k=3)
    except:
        policy_results = []

    # 이의제기 사례 검색 (RAG 검색기 사용)
    try:
        appeal_results = rag_retriever.search_appeals(enhanced_query, top_k=2)
    except:
        appeal_results = []

    all_matches = report_results + policy_results + appeal_results
    retrieved_docs = [{
        "content": match["metadata"]["content"],
        "source": match["metadata"].get("type", "unknown"),
        "score": match["score"]
    } for match in all_matches]

    print(f" 총 검색 결과: {len(retrieved_docs)}개 문서")

    # ✅ 더 많은 내용 포함하는 컨텍스트 생성
    context = "\n\n".join(
        f"[출처: {doc['source']}]\n{doc['content'][:2000]}" 
        for doc in retrieved_docs[:4]  # 3개에서 4개로 증가
    )

    # ✅ 전체 대화 히스토리 활용 (최근 것만이 아니라 중요한 부분 포함)
    full_conversation = "\n".join(state["qna_dialog_log"])
    
    # 사용자가 이전에 물어본 중요한 질문들 추출
    important_previous_questions = []
    for msg in state["qna_dialog_log"]:
        if msg.startswith("사용자:"):
            question = msg[4:].strip()
            # 중요한 질문들만 필터링
            if any(keyword in question for keyword in ["점수", "등급", "달성률", "성과", "평가", "누구", "어떻게"]):
                important_previous_questions.append(question)
    
    # ✅ 대화 맥락 요약
    conversation_summary = ""
    if len(important_previous_questions) > 1:
        conversation_summary = f"\n**이전 대화 요약**: 사용자가 {', '.join(important_previous_questions[-2:])}에 대해 문의했습니다."

    # ✅ 현재 질문의 의도 파악
    question_intent = analyze_question_intent(query, previous_context)

    # ✅ 개선된 프롬프트 - 문맥 강화
    prompt = f"""
당신은 SK그룹 성과평가 기준에 기반한 AI 상담 챗봇입니다.
사용자({state["user_id"]})와의 지속적인 대화에서 문맥을 이해하고 연관성 있는 답변을 제공해주세요.

** 현재 질문 의도**: {question_intent}

** 대화 흐름 파악:**{conversation_summary}

** 전체 대화 기록:**
{full_conversation}


** 현재 질문:**
{query}

** 참고 문서 ({len(retrieved_docs)}개 - 권한 검증 완료):**
{context}

**중요 지침:**
1. **대화 연속성**: 이전 질문과 답변을 기억하고 연결하여 답변
2. **구체적 수치 제공**: JSON 데이터의 점수, 달성률, 등급 등을 정확히 추출
3. **업무명 정확히 표시**: "업무1", "업무2" 대신 실제 업무명 사용
4. **문맥 기반 추론**: 이전 대화를 바탕으로 사용자의 진짜 궁금증 파악
5. **반복 방지**: 이미 답변한 내용은 간략히 언급하고 새로운 정보 제공
6. **관련 정보 연결**: 현재 질문과 관련된 추가 유용한 정보도 함께 제공
7. **직접적으로**: 질문에 바로 답하기
8. **형식 금지**: 번호, 불릿포인트, **굵은글씨** 등 사용 안함

**답변 스타일:**
- 사용자가 정말 알고 싶어하는 핵심을 파악해서 답변
- 단순 반복보다는 발전적이고 연결된 정보 제공
- 본인 외 다른 직원은 익명으로 처리



**JSON 데이터 처리:**
- "Task명" → 실제 업무명으로 표시
- "누적_달성률", "달성률" → 성과 수치로 활용
- "점수", "등급", "기여도" → 구체적 수치로 명시
- 시간대별 데이터가 있으면 변화 추이도 설명
"""

    llm_response = config.llm.predict(prompt)
    
    state["retrieved_docs"] = retrieved_docs
    state["llm_response"] = llm_response.strip()
    state["qna_dialog_log"].append(f"챗봇: {llm_response.strip()}")
    
    return state

def appeal_dialogue_node(state: ChatState) -> ChatState:
    """RAG 검색기를 사용하는 권한 제어 적용된 이의제기 대화 노드"""
    query = state["user_input"]
    user_metadata = {
        "emp_no": state["user_id"],
        "role": state["role"],
        "team_name": state["team_id"]
    }

    state["dialog_log"].append(f"사용자: {query}")

    # 팩트 체크 키워드 목록
    FACT_TRIGGER_KEYWORDS = [
        "점수", "등급", "기준", "평가", "순위", "컷오프", "평가표", "채점", "근거",
        "기여", "비중", "수치", "비율", "정량", "정성", "성과", "퍼센트", "%",
        "기준치", "초과", "미달", "도달", "충족", "달성", "충분", "부족",
        "어디에", "문서", "기록", "정책", "규정", "왜", "무엇 때문에", "무슨 이유",
        "몇점", "누구", "정보", "데이터", "결과"
    ]

    normalized_query = query.lower().replace(" ", "")
    needs_fact = any(k.replace(" ", "") in normalized_query for k in FACT_TRIGGER_KEYWORDS)

    # 🎯 RAG 검색 (필요할 때만) - 권한 제어 자동 적용
    retrieved_docs = []
    fact_info = ""
    
    if needs_fact:
        print(f"🔍 팩트 체크 필요: '{query}'")
        try:
            # RAG 검색기를 통한 권한 제어 검색
            results = search_documents_with_access_control(
                query=query,
                user_metadata=user_metadata,
                filter_type="report",
                top_k=3
            )["matches"]

            retrieved_docs = [{
                "content": r["metadata"]["content"],
                "source": r["metadata"].get("type", "unknown"),
                "score": r["score"]
            } for r in results]

            if retrieved_docs:
                fact_context = "\n\n".join(f"{doc['content'][:800]}" for doc in retrieved_docs[:2])
                fact_info = f"[참고 문서 - 권한 검증 완료]\n{fact_context}"
                print(f"📊 팩트 데이터 검색 완료: {len(retrieved_docs)}개 문서")
            else:
                fact_info = "(접근 가능한 참고 문서 없음)"
                print("⚠️ 접근 가능한 팩트 데이터 없음")
        except Exception as e:
            fact_info = "(참고 문서 검색 실패)"
            print(f"❌ 팩트 검색 실패: {str(e)}")
    else:
        fact_info = "(팩트 체크 불필요)"
        print("ℹ️ 팩트 체크 불필요한 질문")

    # ✅ 전체 대화 히스토리 포함 (최근 것만이 아니라 전체)
    full_conversation = "\n".join(state["dialog_log"])
    
    # ✅ 사용자의 핵심 불만사항 추출
    user_messages = [msg for msg in state["dialog_log"] if msg.startswith("사용자:")]
    user_concerns = "\n".join(user_messages[-3:])  # 최근 3개 사용자 메시지

    # ✅ 개선된 프롬프트 - 문맥 강화
    prompt = f"""
당신은 SK 성과관리 AI 챗봇으로서, 사용자가 평가에 대해 느낀 억울함이나 문제의식을 명확히 표현하도록 돕는 역할입니다.

** 현재 사용자의 주요 관심사 파악:**
{user_concerns}

** 전체 대화 맥락:**
{full_conversation}

** 사용자 최신 입력:**
{query}

** 참고 데이터 (권한 검증 완료):**
{fact_info}

**중요한 응답 규칙:**
1. **대화의 연속성 유지**: 이전에 언급된 내용(야근, 노고 등)을 기억하고 연결
2. **사용자 감정 공감**: 억울함, 아쉬움 등의 감정을 충분히 인정
3. **구체적 정보 제공**: 점수나 데이터 요청 시 정확한 정보 먼저 제공 (권한 범위 내에서)
4. **자연스러운 대화**: 앞서 말한 내용을 반복해서 물어보거나 답하지 않기
5. **감정 유도**: 사용자가 더 자세한 불만을 표현하도록 유도
6. **권한 존중**: 본인만 접근 가능한 데이터임을 자연스럽게 반영
7. 전체 대화 맥락을 고려해 답변

**응답 가이드:**
- 사용자가 이미 말한 문제를 기억하고 이를 바탕으로 응답
- 부드럽고 친근한 말투 유지
- 본인 외 다른 직원은 익명으로 처리

**답변 스타일:**
- **간결함**: 최대 2-3문장으로 답변
- **친근함**: 길고 형식적인 문장보다 짧고 자연스럽게
- **적절한 공감**: 과도한 위로보다 적절한 수준의 공감

**답변 길이 제한:**
- 일반 질문: 1-2문장
- 복잡한 질문: 최대 3-4문장
- 불만/감정 표현: 짧은 공감 + 핵심 질문 1개

**금지사항:**
- "함께 고민해볼 수 있을 것 같아요" (반복됨)
- "어떤 아이디어가 있으신가요?" (해결책 강요)
- 이전과 똑같은 문장 사용
- 3문장 이상의 긴 답변
"""

    # LLM 호출
    llm_response = config.llm.predict(prompt)

    # 상태 업데이트
    state["dialog_log"].append(f"챗봇: {llm_response.strip()}")
    state["llm_response"] = llm_response.strip()
    state["retrieved_docs"] = retrieved_docs

    return state

def summary_generator_node(state: ChatState) -> ChatState:
    """전체 문맥을 반영하는 요약 생성 노드"""
    # ✅ 전체 대화 기록 사용
    full_conversation = "\n".join(state["dialog_log"])
    
    # 사용자 발언만 추출해서 핵심 불만 파악
    user_statements = []
    for msg in state["dialog_log"]:
        if msg.startswith("사용자:"):
            user_statements.append(msg[4:].strip())  # "사용자:" 제거
    
    user_concerns = "\n".join(f"- {stmt}" for stmt in user_statements)

    prompt = f"""
당신은 SK그룹 성과평가 시스템의 AI 요약 전문가입니다.
아래 대화를 바탕으로 구성원의 이의제기 내용을 팀장이 이해하기 쉽게 요약해주세요.

** 구성원이 제기한 주요 내용:**
{user_concerns}

** 전체 대화 기록:**
{full_conversation}

** 요약 작성 조건:**
1. **완전한 익명 표현** 사용 ("구성원"으로 통일)
2. **욕설, 감정적 표현은 정중하게 정제**
3. **핵심 이슈를 두괄식**으로 요약
4. **여전히 의문이 남는 내용만** 포함 (수긍한 내용 제외)
5. **논리적이고 예의바른 톤** 유지
6. **시간순으로 일관된 스토리** 구성
7. **특정될 수 있는 숫자 제거** (달성률, 기여도 등)
8. 핵심만 한 문단으로 간단히 요약

** 요약 내용:**
핵심 이슈 (한 문장)
구성원이 제기한 문제점과 구성원이 원하는 바 간략히 요약


팀장이 상황을 명확히 이해하고 적절한 피드백을 제공할 수 있도록 작성해주세요.
"""

    llm_response = config.llm.predict(prompt)
    state["summary_draft"] = llm_response.strip()
    state["llm_response"] = llm_response.strip()
    return state

# =============================================================================
# 4. 세션 관리 클래스
# =============================================================================

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session_state(self, user_id: str, chat_mode: str) -> Dict:
        session_key = f"{user_id}_{chat_mode}"
        return self.sessions.get(session_key, {})
    
    def save_session_state(self, user_id: str, chat_mode: str, state: ChatState):
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

    def auto_generate_summary_on_exit(self, user_id: str, chatbot_instance) -> Optional[str]:
        """이의제기 모드 종료 시 자동으로 요약 생성"""
        session_key = f"{user_id}_appeal_to_manager"
        
        if session_key not in self.sessions:
            return None
            
        session_data = self.sessions[session_key]
        dialog_log = session_data.get("dialog_log", [])
        
        # 대화가 2개 이상의 메시지가 있을 때만 요약 생성
        if len(dialog_log) >= 4:  # 사용자 2회 + 챗봇 2회 이상
            print("🔄 이의제기 대화 종료 - 자동으로 요약을 생성합니다...")
            
            try:
                response = chatbot_instance.chat(
                    user_id=user_id,
                    chat_mode="appeal_to_manager",
                    user_input="요약해주세요",
                    appeal_complete=True
                )
                
                if response["type"] == "appeal_summary":
                    return response["summary"]
            except Exception as e:
                print(f"❌ 자동 요약 생성 실패: {str(e)}")
                
        return None

session_manager = SessionManager()

# =============================================================================
# 5. LangGraph 워크플로우 구성
# =============================================================================

def create_chatbot_workflow():
    """챗봇 워크플로우 생성"""
    workflow = StateGraph(ChatState)
    
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("qna_agent", qna_agent_node)
    workflow.add_node("appeal_dialogue", appeal_dialogue_node)
    workflow.add_node("summary_generator", summary_generator_node)
    
    workflow.add_edge(START, "initialize")
    
    workflow.add_conditional_edges(
        "initialize",
        route_chat_mode,
        {
            "qna_agent": "qna_agent",
            "appeal_dialogue": "appeal_dialogue",
            "summary_generator": "summary_generator"
        }
    )
    
    workflow.add_edge("qna_agent", END)
    workflow.add_edge("appeal_dialogue", END)
    workflow.add_edge("summary_generator", END)
    
    return workflow.compile()

# =============================================================================
# 6. 메인 챗봇 클래스
# =============================================================================

class SKChatbot:
    def __init__(self):
        self.workflow = create_chatbot_workflow()
    
    def chat(self, user_id: str, chat_mode: str, user_input: str, appeal_complete: bool = False) -> Dict:
        saved_state = session_manager.get_session_state(user_id, chat_mode)
        
        current_state = {
            "user_id": user_id,
            "chat_mode": chat_mode,
            "user_input": user_input,
            "appeal_complete": appeal_complete,
            "qna_dialog_log": saved_state.get("qna_dialog_log", []),
            "dialog_log": saved_state.get("dialog_log", []),
        }
        
        result_state = self.workflow.invoke(current_state)
        session_manager.save_session_state(user_id, chat_mode, result_state)
        
        if chat_mode == "appeal_to_manager" and appeal_complete:
            return {
                "type": "appeal_summary",
                "summary": result_state["summary_draft"],
                "user_id": user_id
            }
        elif chat_mode == "appeal_to_manager":
            return {
                "type": "appeal_dialogue",
                "response": result_state["llm_response"],
                "user_id": user_id
            }
        else:
            return {
                "type": "qna_response",
                "response": result_state["llm_response"],
                "user_id": user_id
            }
    
    def get_session_history(self, user_id: str, chat_mode: str) -> List[str]:
        saved_state = session_manager.get_session_state(user_id, chat_mode)
        if chat_mode == "appeal_to_manager":
            return saved_state.get("dialog_log", [])
        else:
            return saved_state.get("qna_dialog_log", [])
    
    def clear_session(self, user_id: str, chat_mode: str):
        session_manager.clear_session(user_id, chat_mode)