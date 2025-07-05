# =============================================================================
# llm_utils.py - 프롬프트 관련 함수들
# =============================================================================

import os
import sys
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# DB 설정
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

class ChatbotConfig:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        )
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        self.index_reports = self.pc.Index(os.getenv("PINECONE_INDEX_REPORTS"))
        self.index_policy = self.pc.Index(os.getenv("PINECONE_INDEX_POLICY"))
        self.index_appeals = self.pc.Index(os.getenv("PINECONE_INDEX_APPEALS"))


def get_user_metadata(user_id: str) -> dict:
    """사용자 메타데이터 조회"""
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
                    "emp_no": user_id,
                    "role": result.role,
                    "team_id": result.team_id,
                    "team_name": result.team_name
                }
            else:
                return {
                    "emp_no": user_id,
                    "role": "MEMBER", 
                    "team_id": "default", 
                    "team_name": "default"
                }
    except Exception as e:
        print(f"❌ 사용자 정보 조회 실패: {str(e)}")
        return {
            "emp_no": user_id,
            "role": "MEMBER", 
            "team_id": "default", 
            "team_name": "default"
        }

def analyze_question_intent(current_query: str, previous_context: str) -> str:
    """현재 질문의 의도를 분석하는 함수"""
    
    # 기본 의도 카테고리
    if any(keyword in current_query for keyword in ["점수", "몇점", "등급"]):
        if "점수" in previous_context:
            return "점수 재확인 또는 상세 설명 요청"
        else:
            return "개인 점수/등급 조회"
    
    elif any(keyword in current_query for keyword in ["달성률", "성과", "결과"]):
        return "업무 성과 및 달성률 조회"
    
    elif any(keyword in current_query for keyword in ["누구", "나", "내", "이름"]):
        if previous_context:
            return "개인 신원 재확인 (대화 맥락에서)"
        else:
            return "개인 신원 확인"
    
    elif any(keyword in current_query for keyword in ["왜", "이유", "어떻게"]):
        return "평가 기준이나 산출 방식 문의"
    
    elif any(keyword in current_query for keyword in ["팀", "동료", "비교"]):
        return "팀 또는 동료와의 비교 분석"
    
    elif any(keyword in current_query for keyword in ["업무", "프로젝트", "task"]):
        return "개별 업무별 성과 조회"
    
    else:
        if previous_context:
            return "이전 대화와 연관된 추가 질문"
        else:
            return "일반적인 성과 관련 문의"