# =============================================================================
# 수정된 interactive_test.py - import 경로 문제 해결
# =============================================================================

"""
chatbot/interactive_test.py 파일 내용
"""

# 현재 디렉토리에서 직접 import
from agents.evaluation.modules.chatbot.main_chatbot import SKChatbot

def interactive_chat():
    """대화형 챗봇 테스트"""
    print("🤖 SK 성과평가 챗봇 대화형 테스트")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요\n")
    
    try:
        chatbot = SKChatbot()
        print("✅ 챗봇 초기화 완료!\n")
    except Exception as e:
        print(f"❌ 챗봇 초기화 실패: {str(e)}")
        print("💡 sk_chatbot.py와 main_chatbot.py 파일이 올바르게 설정되어 있는지 확인해주세요.")
        return
    
    user_id = input("사용자 ID를 입력하세요 (예: sk0003): ").strip()
    if not user_id:
        user_id = "sk0003"  # 기본값
        print(f"기본값 사용: {user_id}")
    
    print("\n채팅 모드를 선택하세요:")
    print("1. default (일반 QnA)")
    print("2. appeal_to_manager (이의제기)")
    
    mode_choice = input("모드 선택 (1 또는 2, 기본값: 1): ").strip()
    chat_mode = "default" if mode_choice != "2" else "appeal_to_manager"
    
    print(f"\n=== {chat_mode} 모드 시작 ===")
    print("💡 사용 가능한 명령어:")
    print("  - 'quit', 'exit', 'q': 종료")
    print("  - 'history', 'h': 대화 히스토리 보기")
    print("  - '요약해줘' (이의제기 모드에서): 요약 생성")
    print("="*50)
    
    turn_count = 0
    
    while True:
        try:
            user_input = input(f"[{turn_count+1}턴] 당신: ").strip()
            
            # 종료 명령어 체크
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 대화를 종료합니다.")
                break
            
            # 히스토리 명령어 체크
            if user_input.lower() in ['history', 'h', '히스토리']:
                print("\n=== 📜 대화 히스토리 ===")
                try:
                    history = chatbot.get_session_history(user_id, chat_mode)
                    if history:
                        for i, msg in enumerate(history, 1):
                            print(f"{i}. {msg}")
                    else:
                        print("아직 대화 히스토리가 없습니다.")
                except Exception as e:
                    print(f"히스토리 조회 실패: {str(e)}")
                print("=" * 30 + "\n")
                continue
            
            # 빈 입력 체크
            if not user_input:
                print("💬 메시지를 입력해주세요.\n")
                continue
            
            # 요약 요청 체크
            appeal_complete = False
            if chat_mode == "appeal_to_manager" and any(keyword in user_input for keyword in ['요약', '정리', '마무리']):
                appeal_complete = True
                print("📝 요약을 생성하고 있습니다...")
            
            # 챗봇 호출
            print("🤔 생각 중...")
            response = chatbot.chat(
                user_id=user_id,
                chat_mode=chat_mode,
                user_input=user_input,
                appeal_complete=appeal_complete
            )
            
            # 응답 출력
            if response["type"] == "appeal_summary":
                print(f"\n🤖 챗봇: [📋 요약 완료]\n")
                print("=" * 50)
                print(response['summary'])
                print("=" * 50)
                print("\n✅ 요약이 완료되었습니다. 대화를 종료합니다.")
                break
            elif response["type"] == "error":
                print(f"🤖 챗봇: ❌ {response.get('message', '오류가 발생했습니다.')}\n")
            else:
                content = response.get("response", response.get("summary", "응답을 생성할 수 없습니다."))
                print(f"🤖 챗봇: {content}\n")
            
            turn_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⏹️ 사용자에 의해 대화가 중단되었습니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            print("🔄 다시 시도해주세요.\n")
    
    # 최종 히스토리 출력
    print("\n" + "="*50)
    print("📊 최종 대화 통계")
    print("="*50)
    try:
        history = chatbot.get_session_history(user_id, chat_mode)
        print(f"총 대화 횟수: {len(history)}개")
        print(f"사용자 ID: {user_id}")
        print(f"채팅 모드: {chat_mode}")
        
        if history:
            print(f"\n📜 전체 대화 히스토리:")
            for i, msg in enumerate(history, 1):
                print(f"  {i}. {msg}")
        else:
            print("대화 히스토리가 없습니다.")
            
    except Exception as e:
        print(f"최종 히스토리 조회 실패: {str(e)}")
    
    print("\n👋 SK 성과평가 챗봇을 이용해주셔서 감사합니다!")

def quick_test():
    """빠른 테스트 함수"""
    print("⚡ 빠른 연결 테스트")
    try:
        chatbot = SKChatbot()
        
        test_response = chatbot.chat(
            user_id="test_user",
            chat_mode="default",
            user_input="안녕하세요, 테스트입니다."
        )
        
        print(f"✅ 테스트 성공!")
        print(f"응답 타입: {test_response['type']}")
        print(f"응답 내용: {test_response.get('response', '없음')[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 SK 성과평가 챗봇 대화형 테스트 시작\n")
    
    # 먼저 빠른 테스트
    if quick_test():
        print("\n" + "="*50)
        print("🎉 기본 연결 테스트 통과! 대화형 테스트를 시작합니다.")
        print("="*50)
        interactive_chat()
    else:
        print("\n💡 문제 해결 방법:")
        print("1. sk_chatbot.py 파일이 올바르게 작성되어 있는지 확인")
        print("2. main_chatbot.py 파일이 올바르게 작성되어 있는지 확인")
        print("3. .env 파일이 설정되어 있는지 확인")
        print("4. 필요한 패키지가 설치되어 있는지 확인")
        print("\n🔧 다음 명령어로 개별 테스트를 해보세요:")
        print("   python sk_chatbot.py")
        print("   python main_chatbot.py")

# =============================================================================
# 추가: 간단한 명령행 인터페이스
# =============================================================================

def simple_chat():
    """더 간단한 채팅 인터페이스"""
    print("💬 간단 채팅 모드")
    
    try:
        chatbot = SKChatbot()
    except Exception as e:
        print(f"초기화 실패: {str(e)}")
        return
    
    user_id = "demo_user"
    chat_mode = "default"
    
    print(f"사용자: {user_id} | 모드: {chat_mode}")
    print("'bye'를 입력하면 종료됩니다.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['bye', 'goodbye', '바이']:
            break
            
        if not user_input:
            continue
            
        try:
            response = chatbot.chat(user_id, chat_mode, user_input)
            content = response.get("response", "응답 없음")
            print(f"Bot: {content}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")
    
    print("Goodbye! 👋")

# 실행 옵션 추가
if __name__ == "__main__" and len(__import__('sys').argv) > 1:
    if __import__('sys').argv[1] == "simple":
        simple_chat()
    else:
        interactive_chat()