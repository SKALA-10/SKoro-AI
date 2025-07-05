def rebuild_vectordb_with_correct_access_control():
    """올바른 권한 제어 기반 벡터DB 재구축"""
    print("🚀 올바른 권한 제어 기반 벡터DB 재구축 시작...")
    print("📋 권한 규칙:")
    print("   MANAGER: 본인 개별 리포트 + 팀원들 개별 리포트 + 팀 평가 리포트")
    print("   MEMBER: 본인 개별 리포트만")
    print("   ❌ 팀원끼리는 서로 데이터 접근 불가")
    print("-" * 60)
    
    # 1. 기존 벡터DB 삭제
    try:
        print("🗑️ 기존 벡터DB 삭제 중...")
        index.delete(delete_all=True)
        print("✅ 기존 벡터DB 삭제 완료")
    except Exception as e:
        print(f"⚠️ 기존 벡터DB 삭제 실패: {e}")
    
    with engine.connect() as conn:
        # 2. 모든 직원 정보 조회
        emp_query = text("SELECT emp_no, role, team_id FROM employees")
        employees = conn.execute(emp_query).fetchall()
        
        print(f"📊 총 {len(employees)}명의 직원 데이터 처리 예정")
        
        all_documents = []
        
        for emp_row in employees:
            emp_no = emp_row.emp_no
            emp_role = emp_row.role
            team_id = emp_row.team_id
            
            print(f"\n👤 {emp_no} ({emp_role}) 처리 중...")
            
            try:
                # 메타데이터 생성
                metadata = build_metadata(emp_no, conn)
                
                # === MEMBER 권한: 본인 개별 리포트만 ===
                if emp_role == "MEMBER":
                    print(f"  📝 MEMBER 권한: 본인 개별 리포트만 생성")
                    individual_report = build_individual_report(emp_no, conn)
                    
                    # 청킹
                    chunks = chunk_by_section_and_length(individual_report, max_length=1500)
                    
                    # Document 객체 생성 - 본인만 접근 가능
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_index": i,
                                "type": "individual_report",
                                "doc_id": f"{emp_no}_individual_{i}",
                                "accessible_by": [emp_no]  # ✅ 본인만 접근 가능
                            }
                        )
                        all_documents.append(doc)
                    
                    print(f"    → 개별 리포트: {len(chunks)}개 청크 (본인만 접근)")
                
                # === MANAGER 권한: 본인 + 팀원 개별 리포트 + 팀 평가 ===
                elif emp_role == "MANAGER":
                    print(f"  📝 MANAGER 권한: 확장된 데이터 생성")
                    
                    # 팀 멤버 목록 조회
                    team_members = get_team_members(team_id, conn)
                    print(f"    팀 멤버: {len(team_members)}명")
                    
                    # 1) 매니저 본인 개별 리포트 - 본인만 접근 가능
                    individual_report = build_individual_report(emp_no, conn)
                    chunks = chunk_by_section_and_length(individual_report, max_length=1500)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_index": i,
                                "type": "individual_report",
                                "doc_id": f"{emp_no}_individual_{i}",
                                "accessible_by": [emp_no]  # ✅ 매니저 본인만 접근
                            }
                        )
                        all_documents.append(doc)
                    
                    print(f"    → 본인 개별 리포트: {len(chunks)}개 청크 (본인만)")
                    
                    # 2) 팀원들 개별 리포트 - 해당 팀원 + 매니저만 접근 가능
                    team_member_count = 0
                    for member_no in team_members:
                        if member_no != emp_no:  # 본인 제외
                            member_report = build_individual_report(member_no, conn)
                            member_metadata = build_metadata(member_no, conn)
                            member_chunks = chunk_by_section_and_length(member_report, max_length=1500)
                            
                            for i, chunk in enumerate(member_chunks):
                                doc = Document(
                                    page_content=chunk,
                                    metadata={
                                        **member_metadata,
                                        "chunk_index": i,
                                        "type": "individual_report",
                                        "doc_id": f"{member_no}_individual_{i}",
                                        "accessible_by": [member_no, emp_no]  # ✅ 해당 팀원 + 매니저만
                                    }
                                )
                                all_documents.append(doc)
                            
                            team_member_count += len(member_chunks)
                    
                    print(f"    → 팀원 개별 리포트: {team_member_count}개 청크 (각 팀원+매니저만)")
                    
                    # 3) 팀 평가 리포트 - 매니저만 접근 가능
                    team_report = get_team_evaluation_text(team_id, conn)
                    team_chunks = chunk_by_section_and_length(team_report, max_length=1500)
                    
                    for i, chunk in enumerate(team_chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_index": i,
                                "type": "team_report",
                                "doc_id": f"{emp_no}_team_{i}",
                                "accessible_by": [emp_no]  # ✅ 매니저만 접근
                            }
                        )
                        all_documents.append(doc)
                    
                    print(f"    → 팀 평가 리포트: {len(team_chunks)}개 청크 (매니저만)")
                
                # === ADMIN 권한 처리 ===
                elif emp_role == "ADMIN":
                    print(f"  📝 ADMIN 권한: 본인 개별 리포트만 생성")
                    individual_report = build_individual_report(emp_no, conn)
                    chunks = chunk_by_section_and_length(individual_report, max_length=1500)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_index": i,
                                "type": "individual_report",
                                "doc_id": f"{emp_no}_individual_{i}",
                                "accessible_by": [emp_no]  # ✅ 본인만 접근
                            }
                        )
                        all_documents.append(doc)
                    
                    print(f"    → 개별 리포트: {len(chunks)}개 청크 (본인만)")
                
                print(f"  ✅ {emp_no} 처리 완료")
                
            except Exception as e:
                print(f"  ❌ {emp_no} 처리 실패: {str(e)}")
                continue
        
        print(f"\n📊 총 {len(all_documents)}개 문서 생성 완료")
        
        # 3. 권한 검증 로그 출력
        print_access_verification(all_documents)
        
        # 4. Pinecone에 업로드
        upload_documents_to_pinecone(all_documents, batch_size=50)
        
        print("\n🎉 올바른 권한 제어 기반 벡터DB 구축 완료!")
        print_final_access_summary(conn)

def print_access_verification(documents: List[Document]):
    """권한 설정 검증 로그 출력"""
    print("\n" + "="*60)
    print("🔍 권한 설정 검증")
    print("="*60)
    
    access_stats = {}
    role_stats = {}
    
    for doc in documents:
        accessible_by = doc.metadata.get("accessible_by", [])
        doc_owner = doc.metadata.get("emp_no", "Unknown")
        doc_role = doc.metadata.get("role", "Unknown")
        doc_type = doc.metadata.get("type", "Unknown")
        
        # 통계 수집
        access_count = len(accessible_by)
        if access_count not in access_stats:
            access_stats[access_count] = 0
        access_stats[access_count] += 1
        
        if doc_role not in role_stats:
            role_stats[doc_role] = {"individual": 0, "team": 0}
        if doc_type == "individual_report":
            role_stats[doc_role]["individual"] += 1
        elif doc_type == "team_report":
            role_stats[doc_role]["team"] += 1
        
        # 권한 검증
        if doc_role == "MEMBER":
            if accessible_by != [doc_owner]:
                print(f"🚫 권한 오류: MEMBER {doc_owner}의 문서가 다른 사용자에게 공개됨: {accessible_by}")
        elif doc_role == "MANAGER":
            if doc_type == "team_report" and accessible_by != [doc_owner]:
                print(f"🚫 권한 오류: MANAGER {doc_owner}의 팀 리포트가 다른 사용자에게 공개됨: {accessible_by}")
    
    print(f"📊 접근 권한 분포:")
    for count, docs in access_stats.items():
        print(f"   {count}명 접근 가능: {docs}개 문서")
    
    print(f"\n📊 역할별 문서 분포:")
    for role, counts in role_stats.items():
        print(f"   {role}: 개별 리포트 {counts['individual']}개, 팀 리포트 {counts['team']}개")

def print_final_access_summary(conn):
    """최종 권한 제어 요약 출력"""
    print("\n" + "="*60)
    print("📋 최종 권한 제어 요약")
    print("="*60)
    
    try:
        with conn:
            # 역할별 직원 수 조회
            role_query = text("SELECT role, COUNT(*) as count FROM employees GROUP BY role")
            role_results = conn.execute(role_query).fetchall()
            
            for row in role_results:
                role = row.role
                count = row.count
                
                if role == "MANAGER":
                    print(f"👨‍💼 MANAGER ({count}명):")
                    print("   ✅ 본인 개별 리포트만 접근")
                    print("   ✅ 팀원들 개별 리포트 접근 (각각 개별적으로)")
                    print("   ✅ 팀 평가 리포트 접근")
                    print("   ❌ 다른 팀 데이터 접근 불가")
                elif role == "MEMBER":
                    print(f"👤 MEMBER ({count}명):")
                    print("   ✅ 본인 개별 리포트만 접근")
                    print("   ❌ 팀원 개별 리포트 접근 불가")
                    print("   ❌ 팀 평가 리포트 접근 불가")
                elif role == "ADMIN":
                    print(f"🔧 ADMIN ({count}명):")
                    print("   ✅ 본인 개별 리포트만 접근")
                
                print()
                
            print("🔒 권한 제어 규칙:")
            print("   - 각 MEMBER는 본인 데이터만 접근")
            print("   - MANAGER는 팀원들 데이터를 개별적으로 접근 (서로 공유 안됨)")
            print("   - 팀 평가는 해당 팀 MANAGER만 접근")
            
    except Exception as e:
        print(f"❌ 요약 출력 실패: {str(e)}")

# =============================================================================
# UnifiedRAGRetriever 클래스 (검색용) - 수정된 권한 제어 버전
# =============================================================================

class UnifiedRAGRetriever:
    """권한 제어가 통합된 RAG 검색기 - 수정된 버전"""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = config.embedding_model
        self.index_reports = config.index_reports
        self.index_policy = config.index_policy
        self.index_appeals = config.index_appeals
    
    def _apply_access_control(self, user_metadata: dict, base_filter: dict = None) -> dict:
        """새로운 accessible_by 필드 기반 권한 제어 필터 생성"""
        emp_no = user_metadata.get("emp_no")
        
        if base_filter is None:
            base_filter = {}
        
        # ✅ Pinecone 배열 필터 문법 사용
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
            
            # accessible_by 리스트에 사용자가 포함되어 있는지 확인
            if emp_no in accessible_by:
                validated_matches.append(match)
                print(f"    ✅ 접근 허용: {doc_emp_no}의 {doc_type} (accessible_by: {accessible_by})")
            else:
                print(f"    🚫 접근 차단: {doc_emp_no}의 {doc_type} (accessible_by: {accessible_by})")
        
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

def quick_access_test(test_emp_no: str = "SK0028"):
    """권한 제어 테스트"""
    print(f"🧪 {test_emp_no} 권한 제어 테스트")
    
    # 임시 사용자 메타데이터
    user_metadata = {
        "emp_no": test_emp_no,
        "role": "MEMBER",  # MEMBER로 테스트
        "team_name": "W4팀"
    }
    
    # 검색 테스트
    rag_retriever = UnifiedRAGRetriever(config)
    results = rag_retriever.search_reports("성과 점수", user_metadata, top_k=10)
    
    print(f"검색 결과: {len(results)}개")
    for result in results:
        accessible_by = result["metadata"].get("accessible_by", [])
        doc_emp_no = result["metadata"].get("emp_no", "Unknown")
        print(f"  - {doc_emp_no}의 문서 (접근권한: {accessible_by})")

# 실행 함수
def run_correct_rebuild():
    """올바른 권한 제어로 재구축 실행"""
    print("🚀 올바른 권한 제어 기반 벡터DB 재구축을 시작합니다.")
    print("⚠️ 이전 잘못된 권한 설정을 수정합니다.")
    
    confirm = input("계속하시겠습니까? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 작업이 취소되었습니다.")
        return
    
    try:
        rebuild_vectordb_with_correct_access_control()
        print("\n✅ 올바른 권한 제어 벡터DB 재구축이 완료되었습니다!")
        
        # 테스트 실행
        print("\n🧪 권한 제어 테스트 실행...")
        quick_access_test("SK0028")
        
    except Exception as e:
        print(f"\n❌ 벡터DB 재구축 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    run_correct_rebuild()