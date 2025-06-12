# =============================================================================
# rag_retriever.py - 완전한 RAG 검색 함수들
# =============================================================================

"""
RAG 검색 관련 함수들을 모아둔 파일
"""

from typing import Dict, List, Tuple

# sk_chatbot에서 전역 객체들 import
from sk_chatbot import embedding_model, index_reports, index_policy, index_appeals

def search_documents_with_access_control(
    query: str,
    user_metadata: dict,
    filter_type: str = None,
    top_k: int = 5
) -> Dict:
    """권한 기반 문서 검색"""
    emp_no = user_metadata["emp_no"]
    role = user_metadata["role"]
    team = user_metadata.get("team_name", user_metadata.get("team"))

    # 필터 조건 구성
    if role == "MANAGER":
        base_filter = {"team": team}
    else:
        base_filter = {"emp_no": emp_no}

    if filter_type:
        base_filter["type"] = filter_type

    # 쿼리 임베딩 생성
    query_embedding = embedding_model.embed_query(query)

    # Pinecone 검색
    results = index_reports.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=base_filter
    )

    return results

def search_policy_documents(query: str, top_k: int = 3) -> List[Dict]:
    """정책 문서 검색"""
    query_embedding = embedding_model.embed_query(query)
    
    results = index_policy.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default",
        filter={"type": "policy"}
    )
    
    return results["matches"]

def search_appeal_cases(query: str, top_k: int = 2) -> List[Dict]:
    """이의제기 사례 검색"""
    query_embedding = embedding_model.embed_query(query)
    
    results = index_appeals.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
    )
    
    return results["matches"]

# =============================================================================
# 통합 검색 함수들 (새로 추가)
# =============================================================================

def search_all_documents(
    query: str, 
    user_metadata: dict,
    include_reports: bool = True,
    include_policy: bool = True, 
    include_appeals: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    모든 문서 타입을 한 번에 검색하는 통합 함수
    
    Args:
        query: 검색 쿼리
        user_metadata: 사용자 메타데이터 (권한 확인용)
        include_reports: 리포트 문서 검색 여부
        include_policy: 정책 문서 검색 여부
        include_appeals: 이의제기 사례 검색 여부
    
    Returns:
        (report_results, policy_results, appeal_results) 튜플
    """
    
    report_results = []
    policy_results = []
    appeal_results = []
    
    # 1. 리포트 문서 검색
    if include_reports:
        try:
            print("📄 리포트 문서 검색 중...")
            report_results = search_documents_with_access_control(
                query=query,
                user_metadata=user_metadata,
                filter_type="report",
                top_k=5
            )["matches"]
            print(f"✅ 리포트 문서 {len(report_results)}개 검색 완료")
        except Exception as e:
            print(f"⚠️ 리포트 검색 실패: {str(e)}")
            report_results = []
    
    # 2. 정책 문서 검색
    if include_policy:
        try:
            print("📋 정책 문서 검색 중...")
            policy_results = search_policy_documents(query, top_k=3)
            print(f"✅ 정책 문서 {len(policy_results)}개 검색 완료")
        except Exception as e:
            print(f"⚠️ 정책 문서 검색 실패: {str(e)}")
            policy_results = []
    
    # 3. 이의제기 사례 검색
    if include_appeals:
        try:
            print("📝 이의제기 사례 검색 중...")
            appeal_results = search_appeal_cases(query, top_k=2)
            print(f"✅ 이의제기 사례 {len(appeal_results)}개 검색 완료")
        except Exception as e:
            print(f"⚠️ 이의제기 사례 검색 실패: {str(e)}")
            appeal_results = []
    
    total_count = len(report_results) + len(policy_results) + len(appeal_results)
    print(f"🔍 총 {total_count}개 문서 검색 완료")
    
    return report_results, policy_results, appeal_results

def search_for_qna(query: str, user_metadata: dict) -> List[Dict]:
    """
    QnA 모드용 검색 - 모든 문서 타입 포함
    
    Returns:
        통합된 검색 결과 리스트
    """
    report_results, policy_results, appeal_results = search_all_documents(
        query=query,
        user_metadata=user_metadata,
        include_reports=True,
        include_policy=True,
        include_appeals=True
    )
    
    # 결과 통합 및 포맷팅
    all_matches = report_results + policy_results + appeal_results
    
    retrieved_docs = [{
        "content": match["metadata"]["content"],
        "source": match["metadata"].get("type", "unknown"),
        "score": match["score"]
    } for match in all_matches]
    
    return retrieved_docs

def search_for_appeal(query: str, user_metadata: dict) -> List[Dict]:
    """
    이의제기 모드용 검색 - 리포트 문서만 (팩트 체크용)
    
    Returns:
        리포트 검색 결과 리스트
    """
    report_results, _, _ = search_all_documents(
        query=query,
        user_metadata=user_metadata,
        include_reports=True,
        include_policy=False,
        include_appeals=False
    )
    
    retrieved_docs = [{
        "content": match["metadata"]["content"],
        "source": match["metadata"].get("type", "unknown"),
        "score": match["score"]
    } for match in report_results]
    
    return retrieved_docs