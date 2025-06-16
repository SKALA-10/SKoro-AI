# =============================================================================
# main_chatbot.py - SKChatbot 클래스 정의
# =============================================================================

"""
SKChatbot 메인 클래스 - 외부에서 호출하는 인터페이스
"""

from typing import Dict, List
from .sk_chatbot import create_chatbot_workflow, session_manager

class SKChatbot:
    """SK 성과평가 챗봇 - LangGraph 기반"""
    
    def __init__(self):
        print("🤖 SKChatbot 초기화 중...")
        try:
            self.workflow = create_chatbot_workflow()
            print("✅ LangGraph 워크플로우 로드 완료!")
        except Exception as e:
            print(f"❌ 워크플로우 초기화 실패: {str(e)}")
            raise
    
    def chat(self, 
             user_id: str, 
             chat_mode: str, 
             user_input: str, 
             appeal_complete: bool = False) -> Dict:
        """
        챗봇과 대화
        
        Args:
            user_id: 사용자 ID (emp_no)
            chat_mode: "default" 또는 "appeal_to_manager"
            user_input: 사용자 입력
            appeal_complete: 이의제기 완료 플래그
        
        Returns:
            응답 딕셔너리
        """
        
        print(f"📝 처리 중: {user_id} | {chat_mode} | {user_input[:30]}...")
        
        try:
            # 1. 세션에서 기존 상태 로드
            saved_state = session_manager.get_session_state(user_id, chat_mode)
            
            # 2. 현재 요청으로 상태 구성
            current_state = {
                "user_id": user_id,
                "chat_mode": chat_mode,
                "user_input": user_input,
                "appeal_complete": appeal_complete,
                # 기존 대화 히스토리 로드
                "qna_dialog_log": saved_state.get("qna_dialog_log", []),
                "dialog_log": saved_state.get("dialog_log", []),
            }
            
            # 3. LangGraph 워크플로우 실행
            result_state = self.workflow.invoke(current_state)
            
            # 4. 세션에 결과 저장
            session_manager.save_session_state(user_id, chat_mode, result_state)
            
            # 5. 응답 형식 결정
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
                
        except Exception as e:
            print(f"❌ 챗봇 처리 중 오류: {str(e)}")
            return {
                "type": "error",
                "message": f"처리 중 오류가 발생했습니다: {str(e)}",
                "user_id": user_id
            }
    
    def get_session_history(self, user_id: str, chat_mode: str) -> List[str]:
        """세션 히스토리 조회"""
        try:
            saved_state = session_manager.get_session_state(user_id, chat_mode)
            if chat_mode == "appeal_to_manager":
                return saved_state.get("dialog_log", [])
            else:
                return saved_state.get("qna_dialog_log", [])
        except Exception as e:
            print(f"❌ 히스토리 조회 실패: {str(e)}")
            return []
    
    def clear_session(self, user_id: str, chat_mode: str):
        """특정 세션 클리어"""
        try:
            session_manager.clear_session(user_id, chat_mode)
            print(f"✅ {user_id}의 {chat_mode} 세션 클리어 완료")
        except Exception as e:
            print(f"❌ 세션 클리어 실패: {str(e)}")

# 테스트용 함수
def test_skChatbot():
    """SKChatbot 테스트"""
    print("🧪 SKChatbot 테스트 시작...")
    
    try:
        # 인스턴스 생성
        chatbot = SKChatbot()
        
        # 간단한 채팅 테스트
        response = chatbot.chat(
            user_id="test_user",
            chat_mode="default",
            user_input="테스트 메시지입니다"
        )
        
        print(f"✅ 테스트 성공! 응답: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 파일을 직접 실행할 때 테스트
    test_skChatbot()