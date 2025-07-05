# =============================================================================
# rag_retriever.py - RAG 검색 및 권한 제어
# =============================================================================

from typing import List, Dict
from .llm_utils import ChatbotConfig

class UnifiedRAGRetriever:
    """권한 제어가 통합된 RAG 검색기"""
    
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.embedding_model = config.embedding_model
        self.index_reports = config.index_reports
        self.index_policy = config.index_policy
        self.index_appeals = config.index_appeals
    
    def _apply_access_control(self, user_metadata: dict, base_filter: dict = None) -> dict:
        """새로운 accessible_by 필드 기반 권한 제어"""
        emp_no = user_metadata.get("emp_no")
  
        if base_filter is None:
            base_filter = {}
    
        # accessible_by 배열에 사용자 emp_no가 포함된 문서만 필터링
        access_filter = {"accessible_by": {"$in": [emp_no]}}
    
        final_filter = {**base_filter, **access_filter}
        print(f"🔍 새로운 권한 필터: accessible_by에 '{emp_no}' 포함")
    
        return final_filter

    def _validate_access_rights(self, matches: List, user_metadata: dict) -> List:
        """새로운 accessible_by 필드 기반 권한 재검증"""
        emp_no = user_metadata.get("emp_no")
        validated_matches = []
    
        for match in matches:
            metadata = match.get('metadata', {})
            accessible_by = metadata.get('accessible_by', [])
            doc_type = metadata.get('type', 'unknown')
            doc_emp_no = metadata.get('emp_no', 'unknown')
        
            if emp_no in accessible_by:
                validated_matches.append(match)
                print(f"    ✅ 접근 허용: {doc_type} (accessible_by: {accessible_by})")
            else:
                print(f"    🚫 접근 차단: {doc_type} (accessible_by: {accessible_by})")
    
        return validated_matches
    
    def search_reports(self, query: str, user_metadata: dict, top_k: int = 5) -> List[Dict]:
        """권한 제어가 적용된 리포트 검색"""
        print(f"📊 리포트 검색: '{query}'")
        
        try:
            # 권한 제어 필터 적용
            filter_dict = self._apply_access_control(
                user_metadata, 
                base_filter={"type": {"$in": ["individual_report", "team_report"]}}
            )
            
            # 검색 실행
            query_embedding = self.embedding_model.embed_query(query)
            results = self.index_reports.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            matches = results.get('matches', [])
            print(f"  - 1차 검색 결과: {len(matches)}개")
            
            # 권한 재검증
            validated_matches = self._validate_access_rights(matches, user_metadata)
            print(f"  - 권한 검증 후: {len(validated_matches)}개")
            
            return self._format_matches(validated_matches, "report")
            
        except Exception as e:
            print(f"❌ 리포트 검색 실패: {str(e)}")
            return []
    
    def search_policies(self, query: str, top_k: int = 3) -> List[Dict]:
        """정책 문서 검색 (모든 사용자 접근 가능)"""
        print(f"📋 정책 검색: '{query}'")
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            results = self.index_policy.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="default",
                filter={"type": "policy"}
            )
            
            matches = results.get("matches", [])
            print(f"  - 정책 문서: {len(matches)}개")
            
            return self._format_matches(matches, "policy")
            
        except Exception as e:
            print(f"❌ 정책 검색 실패: {str(e)}")
            return []
    
    def search_appeals(self, query: str, top_k: int = 2) -> List[Dict]:
        """이의제기 사례 검색 (모든 사용자 접근 가능)"""
        print(f"📝 이의제기 사례 검색: '{query}'")
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            results = self.index_appeals.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="default"
            )
            
            matches = results.get("matches", [])
            print(f"  - 이의제기 사례: {len(matches)}개")
            
            return self._format_matches(matches, "appeal")
            
        except Exception as e:
            print(f"❌ 이의제기 사례 검색 실패: {str(e)}")
            return []
    
    def comprehensive_search(self, query: str, user_metadata: dict, enhanced_query: str = None) -> Dict[str, List[Dict]]:
        """종합 검색 - 모든 소스에서 권한 제어 적용"""
        search_query = enhanced_query or query
        print(f"\n🔍 종합 검색 시작: '{search_query}'")
        print(f"👤 사용자: {user_metadata.get('emp_no')} ({user_metadata.get('role')})")
        
        results = {
            "reports": self.search_reports(search_query, user_metadata, top_k=5),
            "policies": self.search_policies(search_query, top_k=3),
            "appeals": self.search_appeals(search_query, top_k=2)
        }
        
        total_docs = sum(len(docs) for docs in results.values())
        print(f"🎯 종합 검색 완료: 총 {total_docs}개 문서")
        
        return results
    
    def get_best_matches(self, query: str, user_metadata: dict, enhanced_query: str = None, total_k: int = 4) -> List[Dict]:
        """최적의 문서들을 종합해서 반환 (권한 제어 적용)"""
        results = self.comprehensive_search(query, user_metadata, enhanced_query)
        
        # 모든 결과를 점수순으로 정렬
        all_docs = []
        for source_type, docs in results.items():
            all_docs.extend(docs)
        
        # 점수순 정렬 후 상위 문서 반환
        sorted_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)
        final_docs = sorted_docs[:total_k]
        
        print(f"📋 최종 선별: {len(final_docs)}개 최적 문서")
        return final_docs
    
    def _format_matches(self, matches: List, source_type: str) -> List[Dict]:
        """검색 결과를 통일된 형태로 포맷팅"""
        formatted = []
        
        for match in matches:
            metadata = match.get("metadata", {})
            formatted.append({
                "content": metadata.get("content", ""),
                "source": source_type,
                "score": match.get("score", 0.0),
                "metadata": metadata
            })
        
        return formatted

def search_documents_with_access_control(query: str, user_metadata: dict, filter_type: str = None, top_k: int = 5):
    """RAG 검색기를 사용하는 권한 제어 검색 - 기존 인터페이스 유지"""
    from llm_utils import ChatbotConfig
    config = ChatbotConfig()
    rag_retriever = UnifiedRAGRetriever(config)
    
    if filter_type == "report":
        matches = rag_retriever.search_reports(query, user_metadata, top_k)
        return {"matches": matches}
    else:
        # 종합 검색
        report_matches = rag_retriever.search_reports(query, user_metadata, top_k=3)
        policy_matches = rag_retriever.search_policies(query, top_k=2)
        appeal_matches = rag_retriever.search_appeals(query, top_k=1)
        
        all_matches = report_matches + policy_matches + appeal_matches
        return {"matches": all_matches}