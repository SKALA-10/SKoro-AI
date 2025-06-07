from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 전역 LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

print("✅ LLM 설정 완료")