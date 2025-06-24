from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from sqlalchemy import text

# 필요한 라이브러리 import
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.docstore.document import Document

# Hugging Face 모델을 사용한 임베딩 생성
embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 벡터화 함수 (HuggingFace 임베딩 사용)
def get_embeddings(texts):
    embeddings = embedding_model.embed_documents(texts)
    return np.array(embeddings)


def get_task_text(emp_no: str, conn) -> str:
    result = conn.execute(text("""
        SELECT 
            e.emp_no,
            e.emp_name,
            e.cl,
            e.position,
            t.task_name,
            t.target_level,
            t.progress,
            tk.kpi_name,
            tk.kpi_description,
            ts.task_summary,
            ts.task_performance,
            g.grade_s,
            g.grade_a,
            g.grade_b,
            g.grade_c,
            g.grade_d,
            g.grade_rule
        FROM employees e
        JOIN tasks t ON e.emp_no = t.emp_no
        LEFT JOIN team_kpis tk ON t.team_kpi_id = tk.team_kpi_id
        LEFT JOIN task_summaries ts ON t.task_id = ts.task_id
        LEFT JOIN grades g ON g.task_id = t.task_id
        WHERE e.emp_no = :emp_no
    """), {"emp_no": emp_no}).mappings().all()  # ⬅ 여기 .mappings().all()로 변경

    if not result:
        return f"{emp_no}에 대한 업무 데이터가 없습니다."

    emp_name = result[0]['emp_name']
    cl = result[0]['cl']
    position = result[0]['position']

    task_text = [f"직원번호: {emp_no}", f"이름: {emp_name}", f"직무: {position}", f"CL: {cl}"]

    for i, row in enumerate(result, 1):
        task_text.append(f"\n[업무{i}]")
        task_text.append(f"과제명: {row['task_name']}")
        task_text.append(f"목표 수준: {row['target_level']}")
        task_text.append(f"진행률: {row['progress']}%")
        if row['kpi_name']:
            task_text.append(f"KPI: {row['kpi_name']} - {row['kpi_description']}")
        if row['task_summary']:
            task_text.append(f"업무 요약:\n{row['task_summary']}")
        if row['task_performance']:
            task_text.append(f"성과:\n{row['task_performance']}")
        if row['grade_s']:
            task_text.append("성과 등급 기준:")
            task_text.append(f"S: {row['grade_s']}, A: {row['grade_a']}, B: {row['grade_b']}, C: {row['grade_c']}, D: {row['grade_d']}")
            task_text.append(f"평가기준 설명: {row['grade_rule']}")

    return "\n".join(task_text)


def get_feedback_report_text(emp_no: str, conn) -> str:
    query = text("""
        SELECT report
        FROM feedback_reports
        WHERE emp_no = :emp_no
    """)
    result = conn.execute(query, {"emp_no": emp_no}).fetchone()
    return result[0] if result else ""


def get_final_report_text(emp_no: str, conn) -> str:
    query = text("""
        SELECT report
        FROM final_evaluation_reports
        WHERE emp_no = :emp_no
    """)
    result = conn.execute(query, {"emp_no": emp_no}).fetchone()
    return result[0] if result else ""


def get_team_evaluation_text(emp_no: str, conn) -> str:
    query = text("""
        SELECT te.report
        FROM team_evaluations te
        JOIN employees e ON te.team_id = e.team_id
        WHERE e.emp_no = :emp_no
        ORDER BY te.period_id DESC
        LIMIT 1
    """)
    result = conn.execute(query, {"emp_no": emp_no}).fetchone()
    return result[0] if result else ""


def build_full_report_text(emp_no: str, conn) -> str:
    sections = []

    # 개인 업무 요약
    task_section = get_task_text(emp_no, conn)
    sections.append(f"🔹 [개인 업무 요약]\n{task_section}")

    # 최종 평가 보고서
    final_section = get_final_report_text(emp_no, conn)
    sections.append(f"🔹 [최종 평가 보고서]\n{final_section}")

    # 피드백 보고서
    feedback_section = get_feedback_report_text(emp_no, conn)
    sections.append(f"🔹 [피드백 보고서]\n{feedback_section}")

    # 팀 평가 보고서
    team_section = get_team_evaluation_text(emp_no, conn)
    sections.append(f"🔹 [팀 평가 보고서]\n{team_section}")

    return "\n\n".join(sections)


def chunk_by_section_and_length(full_text: str, max_length: int = 1000) -> List[str]:
    """
    섹션 헤더를 기준으로 나눈 뒤, 각 섹션이 max_length를 초과하면 문단 단위로 쪼갬
    """
    chunks = []
    for section in full_text.split("🔹 "):
        if not section.strip():
            continue
        section_title, *section_body = section.strip().split("\n", 1)
        body = section_body[0] if section_body else ""
        if len(body) <= max_length:
            chunks.append(f"🔹 {section_title}\n{body}")
        else:
            # 문단 단위 추가 분할
            paragraphs = body.split("\n\n")
            temp = ""
            for para in paragraphs:
                if len(temp) + len(para) < max_length:
                    temp += para + "\n\n"
                else:
                    chunks.append(f"🔹 {section_title}\n{temp.strip()}")
                    temp = para + "\n\n"
            if temp.strip():
                chunks.append(f"🔹 {section_title}\n{temp.strip()}")
    return chunks

# ✅ HuggingFace 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# ✅ Pinecone 클라이언트 인스턴스 생성 (수정됨)
pc = Pinecone(
    api_key="pcsk_7Xg9Hi_CsCfZ8PEeHVqnwzmgJqWvVXUiaZhxpfFjdZbUGyFWfEfkDdhQmM898otzMhd7Jj"
)

# ✅ 인덱스 불러오기 (ServerlessSpec 제거 - Serverless는 자동 관리)
index = pc.Index("skoro-chatbot")

# ✅ 메타데이터 생성 함수 (동일)
def build_metadata(emp_no: str, conn) -> dict:
    result = conn.execute(text("""
        SELECT e.emp_name, e.cl, e.position, e.role, e.team_id,
               t.team_name, h.headquarter_name, p.part_name
        FROM employees e
        JOIN teams t ON e.team_id = t.team_id
        JOIN headquarters h ON t.headquarter_id = h.headquarter_id
        JOIN parts p ON h.part_id = p.part_id
        WHERE e.emp_no = :emp_no
    """), {"emp_no": emp_no}).fetchone()

    return {
        "emp_no": emp_no,
        "emp_name": result.emp_name,
        "cl": result.cl,
        "position": result.position,
        "role": result.role,
        "team": result.team_name,
        "headquarter": result.headquarter_name,
        "part": result.part_name
    }

# ✅ Pinecone 업로드 함수 (새로 추가)
def upload_documents_to_pinecone(docs, batch_size=100):
    """문서들을 배치 단위로 Pinecone에 업로드"""
    print(f"총 {len(docs)}개 문서를 Pinecone에 업로드 중...")
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        
        # 배치 텍스트 추출
        texts = [doc.page_content for doc in batch]
        
        # 배치 임베딩 생성 (KR-SBERT 사용)
        embeddings = embedding_model.embed_documents(texts)
        
        # Pinecone 업서트 형식으로 변환
        vectors = []
        for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append({
                "id": doc.metadata["doc_id"],  # 기존 doc_id 사용
                "values": embedding,
                "metadata": {
                    # 모든 메타데이터 포함
                    **doc.metadata,
                    "content": doc.page_content  # 원본 텍스트도 저장
                }
            })
        
        # Pinecone에 업서트
        index.upsert(vectors=vectors)
        print(f"배치 {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} 완료 ({len(batch)}개 문서)")
    
    print("✅ 모든 문서 업로드 완료!")
    
    # 업로드 확인
    stats = index.describe_index_stats()
    print(f"📊 인덱스 통계: {stats['total_vector_count']}개 벡터 저장됨")

# ✅ 실제 수행 파이프라인 (수정됨)
with engine.connect() as conn:
    emp_nos = conn.execute(text("SELECT emp_no FROM employees")).fetchall()
    emp_nos = [row[0] for row in emp_nos]

    all_docs = []

    for emp_no in emp_nos:
        full_text = build_full_report_text(emp_no, conn)
        chunks = chunk_by_section_and_length(full_text)
        metadata = build_metadata(emp_no, conn)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "type": "report",  # 현재는 report, 추후 feedback/final로 확장 가능
                    "doc_id": f"{emp_no}_report_{i}"
                }
            )
            all_docs.append(doc)

    # ✅ Pinecone에 직접 업로드 (PineconeStore 대신)
    upload_documents_to_pinecone(all_docs)

# ✅ 검색 테스트 함수 (추가)
def test_search(query: str, top_k: int = 5):
    """검색 테스트"""
    # 쿼리 임베딩 생성
    query_embedding = embedding_model.embed_query(query)
    
    # Pinecone 검색
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    print(f"\n🔍 쿼리: '{query}'")
    print("검색 결과:")
    for match in results['matches']:
        print(emp_no, "→ chunk 개수:", len(chunks))

        print(f"  - Score: {match['score']:.4f}")
        print(f"    직원: {match['metadata']['emp_name']}")
        print(f"    내용: {match['metadata']['content'][:100]}...")
        print()