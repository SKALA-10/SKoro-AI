# =============================================================================
# main_chatbot.py - 메인 실행 함수
# =============================================================================

import os
from datetime import datetime
from .agent import SKChatbot, session_manager

# =============================================================================
# 7. 대화형 테스트 인터페이스
# =============================================================================

class InteractiveChatbotTest:
    def __init__(self):
        self.chatbot = SKChatbot()
        self.current_user = None
        self.current_mode = "default"
        self.session_active = False
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        print("🤖 " + "="*60)
        print("   SK 성과평가 AI 챗봇 - 대화형 테스트")
        print("="*63)
        print(f"👤 현재 사용자: {self.current_user}")
        print(f"🔧 현재 모드: {self.current_mode}")
        print(f"📅 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*63)
        
    def show_commands(self):
        print("\\n📋 사용 가능한 명령어:")
        print("  /user <사번>     - 사용자 변경 (예: /user sk0002)")
        print("  /mode default    - QnA 모드로 변경")
        print("  /mode appeal     - 이의제기 모드로 변경")
        print("  /history         - 대화 히스토리 보기")
        print("  /clear           - 현재 세션 클리어")
        print("  /summary         - 이의제기 요약 (appeal 모드에서만)")
        print("  /help            - 도움말")
        print("  /quit            - 종료")
        print("-" * 63)
    
    def handle_command(self, command: str) -> bool:
        command = command.strip()
        
        if command == "/quit":
            return False
        elif command == "/help":
            self.show_commands()
        elif command.startswith("/user "):
            new_user = command[6:].strip()
            if new_user:
                self.current_user = new_user
                print(f"✅ 사용자를 '{new_user}'로 변경했습니다.")
            else:
                print("❌ 사용자 ID를 입력해주세요. (예: /user sk0002)")
        elif command == "/mode default":
            self.current_mode = "default"
            print("✅ QnA 모드로 변경했습니다.")
        elif command == "/mode appeal":
            self.current_mode = "appeal_to_manager"
            print("✅ 이의제기 모드로 변경했습니다.")
        elif command == "/history":
            self.show_history()
        elif command == "/clear":
            self.clear_session()
        elif command == "/summary":
            if self.current_mode == "appeal_to_manager":
                self.generate_summary()
            else:
                print("❌ 요약은 이의제기 모드에서만 가능합니다.")
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
        
        return True
    
    def show_history(self):
        if not self.current_user:
            print("❌ 먼저 사용자를 설정해주세요.")
            return
        
        history = self.chatbot.get_session_history(self.current_user, self.current_mode)
        
        if not history:
            print("📝 대화 히스토리가 없습니다.")
            return
        
        print(f"\\n📜 대화 히스토리 ({len(history)}개):")
        print("-" * 50)
        for i, msg in enumerate(history, 1):
            print(f"{i:2d}. {msg}")
        print("-" * 50)
    
    def clear_session(self):
        if not self.current_user:
            print("❌ 먼저 사용자를 설정해주세요.")
            return
        
        self.chatbot.clear_session(self.current_user, self.current_mode)
        print("✅ 현재 세션이 클리어되었습니다.")
    
    def generate_summary(self):
        if not self.current_user:
            print("❌ 먼저 사용자를 설정해주세요.")
            return
        
        print("🔄 이의제기 요약을 생성하고 있습니다...")
        
        try:
            response = self.chatbot.chat(
                user_id=self.current_user,
                chat_mode=self.current_mode,
                user_input="요약해주세요",
                appeal_complete=True
            )
            
            if response["type"] == "appeal_summary":
                print(f"\\n📋 이의제기 요약:")
                print("=" * 50)
                print(response["summary"])
                print("=" * 50)
            else:
                print("❌ 요약 생성에 실패했습니다.")
        except Exception as e:
            print(f"❌ 요약 생성 중 오류 발생: {str(e)}")
    
    def chat_with_bot(self, user_input: str):
        if not self.current_user:
            print("❌ 먼저 사용자를 설정해주세요.")
            return
        
        print("🤖 답변을 생성하고 있습니다...")
        
        try:
            response = self.chatbot.chat(
                user_id=self.current_user,
                chat_mode=self.current_mode,
                user_input=user_input
            )
            
            print(f"\\n🤖 챗봇:")
            print("-" * 50)
            print(response["response"])
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
    
    def run(self):
        self.clear_screen()
        self.print_header()
        
        print("🎉 SK 성과평가 AI 챗봇에 오신 것을 환영합니다!")
        print("💡 시작하려면 먼저 사용자를 설정해주세요. (예: /user sk0002)")
        self.show_commands()
        
        while True:
            try:
                user_input = input(f"\\n💬 입력하세요: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                else:
                    self.chat_with_bot(user_input)
                    
            except KeyboardInterrupt:
                print("\\n\\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\\n❌ 예상치 못한 오류: {str(e)}")
                continue
        
        print("🔚 대화가 종료되었습니다. 감사합니다!")

# =============================================================================
# 8. 실행 함수
# =============================================================================

def start_interactive_test():
    test = InteractiveChatbotTest()
    test.run()

if __name__ == "__main__":
    start_interactive_test()