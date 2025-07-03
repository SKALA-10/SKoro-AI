"""
팀장용 레포트 톤 보정 Agent 메인 실행 함수
Team manager report tone correction agent main execution function
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from db_utils import engine
from agent import ManagerReportToneAgent, quick_test_manager

# 환경 변수 로드
load_dotenv()

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"LLM Client initialized: {llm_client.model_name}")

def main_manager():
    """팀장용 Agent 메인 실행 함수"""
    # Agent 초기화
    manager_agent = ManagerReportToneAgent(engine, llm_client)
    
    # 모든 레포트 처리 및 DB 저장
    print("=== 팀장용 레포트 톤 보정 Agent 실행 ===")
    corrections = manager_agent.process_all_reports()
    
    # 결과 출력
    print(f"\n📊 처리 완료: 총 {len(corrections)}개 팀장용 레포트")
    
    # 상세 통계 출력
    manager_agent.print_summary()
    
    # 평균 처리 시간 계산
    if corrections:
        avg_time = sum(c.processing_time for c in corrections) / len(corrections)
        print(f"\n⏱️ 평균 처리 시간: {avg_time:.2f}초")
        
        # 성공한 톤 보정 개수
        successful_corrections = sum(1 for c in corrections if c.corrected_text != c.original_text)
        print(f"🎯 실제 톤 보정된 레포트: {successful_corrections}개")
    
    return corrections

if __name__ == "__main__":
    # 빠른 테스트 실행
    print("=== 팀장용 시스템 테스트 시작 ===")
    test_result = quick_test_manager(engine, llm_client)
    
    if test_result:
        print("\n=== 팀장용 메인 프로세스 시작 ===")
        corrections = main_manager()
    else:
        print("\n❌ 팀장용 시스템 테스트 실패, 메인 프로세스를 실행하지 않습니다.")