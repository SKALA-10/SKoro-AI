# -*- coding: utf-8 -*-
"""
모듈8: 팀 성과 비교 분석 모듈 - 완전 구현

기능:
1. 클러스터 통계 존재 확인 및 계산
2. 우리팀 + 유사팀 성과 데이터 수집  
3. KPI별 유사도 매칭 + 비교 분석
4. LLM 기반 팀 성과 코멘트 생성
5. JSON 결과 DB 저장
6. LangGraph 워크플로우 구조
"""

import sys
import os
import json
import re
import statistics
import time
import logging
from typing import Dict, List, Optional, Any, Literal, TypedDict
from dataclasses import dataclass
from datetime import datetime

# 환경 설정
current_file_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_file_dir, '../../../../'))
CACHE_DIR = os.path.join(ROOT_DIR, 'data', 'cache')

# 캐시 디렉토리 생성 (없으면)
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.append(ROOT_DIR)

# 필수 라이브러리 임포트
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row

# DB 연결
from config.settings import DatabaseConfig

# LLM 관련
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage

# 유사도 분석
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 팀 성과 비교 분석기 임포트
from shared.team_performance_comparator import TeamPerformanceComparator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx 및 관련 라이브러리 로그 레벨 조정 (HTTP 요청 로그 숨김)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# DB 설정 및 LLM 클라이언트 초기화
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
logger.info(f"LLM Client initialized: {llm_client.model_name}")

# ===== 상태 정의 =====
class Module8AgentState(TypedDict):
    """
    모듈 8 (팀 성과 비교 모듈)의 내부 상태를 정의합니다.
    """
    messages: List[HumanMessage]
    
    # 기본 정보
    team_id: int
    period_id: int
    report_type: Literal["quarterly", "annual_manager"]
    
    # 클러스터링 결과
    our_team_cluster_id: int
    similar_teams: List[int]
    cluster_stats: Dict
    
    # 성과 데이터
    our_team_kpis: List[Dict]
    our_team_overall_rate: float
    similar_teams_performance: List[Dict]
    
    # 비교 분석 결과
    kpi_comparison_results: List[Dict]
    team_performance_summary: Dict
    
    # 최종 결과
    team_performance_comment: str
    final_comparison_json: Dict
    
    # 업데이트된 ID
    updated_team_evaluation_id: Optional[int]

# ===== 에러 처리 클래스 =====
class Module8ValidationError(Exception):
    pass

class Module8DataIntegrityError(Exception):
    pass

# ===== 유틸리티 함수 =====
def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def robust_llm_call(prompt: str, validation_func, max_retries: int = 3, context: str = ""):
    """견고한 LLM 호출"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = llm_client.invoke(prompt)
            content = str(response.content)
            validated_result = validation_func(content)
            return validated_result
            
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call attempt {attempt + 1} failed for {context}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 지수 백오프
    
    logger.error(f"All LLM attempts failed for {context}: {last_error}")
    raise Module8ValidationError(f"Failed after {max_retries} attempts: {last_error}")

# ===== 데이터 조회 함수들 =====

def fetch_team_kpis_data(team_id: int, period_id: int) -> Optional[Dict]:
    """팀 KPI 데이터와 종합 달성률 조회"""
    with engine.connect() as connection:
        # team_evaluations.average_achievement_rate 조회
        overall_query = text("""
            SELECT te.average_achievement_rate as overall_rate
            FROM team_evaluations te
            WHERE te.team_id = :team_id AND te.period_id = :period_id
        """)
        
        overall_result = connection.execute(overall_query, {
            "team_id": team_id, 
            "period_id": period_id
        }).fetchone()
        
        if not overall_result:
            return None
        
        # KPI 목록 조회 - team_kpi_id 추가
        kpi_query = text("""
            SELECT 
                tk.team_kpi_id,
                tk.kpi_name,
                tk.kpi_description,
                tk.ai_kpi_progress_rate as rate,
                tk.weight
            FROM team_kpis tk
            WHERE tk.team_id = :team_id AND tk.year = 2024
            ORDER BY tk.team_kpi_id
        """)
        
        kpi_results = connection.execute(kpi_query, {"team_id": team_id}).fetchall()
        
        kpis = []
        for row in kpi_results:
            kpis.append({
                "team_kpi_id": row.team_kpi_id,
                "kpi_name": row.kpi_name,
                "kpi_description": row.kpi_description or "",
                "rate": row.rate or 0,
                "weight": row.weight or 0
            })
        
        return {
            "team_id": team_id,
            "overall_rate": overall_result.overall_rate or 0,
            "kpis": kpis
        }

def fetch_multiple_teams_kpis(team_ids: List[int]) -> List[Dict]:
    """여러 팀의 KPI 데이터 배치 조회"""
    if not team_ids:
        return []
    
    with engine.connect() as connection:
        team_ids_str = ','.join(map(str, team_ids))
        query = text(f"""
            SELECT 
                tk.team_id,
                tk.team_kpi_id,
                tk.kpi_name,
                tk.kpi_description,
                tk.ai_kpi_progress_rate as rate,
                tk.weight
            FROM team_kpis tk
            WHERE tk.team_id IN ({team_ids_str}) AND tk.year = 2024
            ORDER BY tk.team_id, tk.team_kpi_id
        """)
        
        results = connection.execute(query).fetchall()
        
        # 팀별로 그룹화
        teams_kpis = {}
        for row in results:
            team_id = row.team_id
            if team_id not in teams_kpis:
                teams_kpis[team_id] = []
            
            teams_kpis[team_id].append({
                "team_id": team_id,
                "team_kpi_id": row.team_kpi_id,
                "kpi_name": row.kpi_name,
                "kpi_description": row.kpi_description or "",
                "rate": row.rate or 0,
                "weight": row.weight or 0
            })
        
        # 리스트로 평탄화
        all_kpis = []
        for team_kpis in teams_kpis.values():
            all_kpis.extend(team_kpis)
        
        return all_kpis

def fetch_team_evaluation_id(team_id: int, period_id: int) -> Optional[int]:
    """team_evaluation_id 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT team_evaluation_id 
            FROM team_evaluations 
            WHERE team_id = :team_id AND period_id = :period_id
        """)
        result = connection.execute(query, {
            "team_id": team_id, 
            "period_id": period_id
        }).scalar_one_or_none()
        return result

# ===== KPI 비교 분석 함수들 =====

def find_similar_kpis_by_text_similarity(our_kpi: Dict, similar_teams_kpis: List[Dict], 
                                       threshold: float = 0.3) -> List[Dict]:
    """텍스트 유사도 기반 KPI 매칭"""
    our_kpi_text = f"{our_kpi['kpi_name']} {our_kpi['kpi_description']}"
    
    matched_kpis = []
    
    for kpi in similar_teams_kpis:
        kpi_text = f"{kpi['kpi_name']} {kpi['kpi_description']}"
        
        # TF-IDF 유사도 계산
        vectorizer = TfidfVectorizer(stop_words=None)
        try:
            tfidf_matrix = vectorizer.fit_transform([our_kpi_text, kpi_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            if similarity >= threshold:
                matched_kpis.append({
                    "kpi": kpi,
                    "similarity": similarity
                })
        except:
            # 벡터화 실패 시 건너뛰기
            continue
    
    return matched_kpis

def get_comparison_result_detailed(our_rate: float, stats: Dict) -> str:
    """통계적 기준으로 상세한 비교 결과 판정"""
    avg = stats["avg_rate"]
    std = stats["std_rate"]
    
    if std == 0:  # 표준편차가 0인 경우
        if our_rate > avg:
            return "우수"
        elif our_rate == avg:
            return "평균"
        else:
            return "개선 필요"
    
    if our_rate >= avg + 1.5 * std:
        return "매우 우수"
    elif our_rate >= avg + 0.5 * std:
        return "우수"
    elif our_rate >= avg - 0.5 * std:
        return "평균"
    elif our_rate >= avg - 1.5 * std:
        return "개선 필요"
    else:
        return "크게 개선 필요"

def compare_kpis_with_similar_teams(our_kpis: List[Dict], similar_teams_kpis: List[Dict]) -> List[Dict]:
    """KPI별 유사도 매칭 및 비교"""
    comparison_results = []
    min_sample_size = 3
    
    for our_kpi in our_kpis:
        # 유사 KPI 찾기
        similar_kpis = find_similar_kpis_by_text_similarity(our_kpi, similar_teams_kpis)
        
        if len(similar_kpis) >= min_sample_size:
            # 충분한 샘플 → 평균 계산
            similar_rates = [matched["kpi"]["rate"] for matched in similar_kpis]
            similar_avg = statistics.mean(similar_rates)
            similar_std = statistics.stdev(similar_rates) if len(similar_rates) > 1 else 0
            
            # 통계적 비교
            comparison_result = get_comparison_result_detailed(
                our_kpi["rate"], {"avg_rate": similar_avg, "std_rate": similar_std}
            )
            
            comparison_results.append({
                "team_kpi_id": our_kpi["team_kpi_id"],
                "kpi_name": our_kpi["kpi_name"],
                "our_rate": our_kpi["rate"],
                "similar_avg_rate": round(similar_avg, 1),
                "similar_kpis_count": len(similar_kpis),
                "comparison_result": comparison_result
            })
        else:
            # 샘플 부족 → 비교 불가
            comparison_results.append({
                "team_kpi_id": our_kpi["team_kpi_id"],
                "kpi_name": our_kpi["kpi_name"],
                "our_rate": our_kpi["rate"],
                "similar_avg_rate": None,
                "similar_kpis_count": len(similar_kpis),
                "comparison_result": "-"
            })
    
    return comparison_results

# ===== LLM 함수 =====

def call_llm_for_team_performance_comment(our_overall_rate: float, cluster_stats: Dict, 
                                        kpi_comparison_results: List[Dict], 
                                        similar_teams_count: int) -> str:
    """팀 성과 비교 코멘트 생성"""
    
    # KPI별 상세 분석 문자열 생성
    kpi_details = ""
    for kpi in kpi_comparison_results:
        if kpi["similar_avg_rate"] is not None:
            kpi_details += f"- {kpi['kpi_name']}: {kpi['our_rate']}% vs {kpi['similar_avg_rate']}% (유사팀 평균) → {kpi['comparison_result']}\n"
        else:
            kpi_details += f"- {kpi['kpi_name']}: {kpi['our_rate']}% (유사 KPI 없음) → -\n"
    
    system_prompt = """
    당신은 SK의 팀 성과 분석 전문가입니다.
    다음 정보를 바탕으로 팀장에게 제공할 객관적이고 건설적인 성과 분석 코멘트를 작성해주세요.
    
    다음 구성으로 250-300자 분량의 코멘트를 작성해주세요:
    1. 종합 달성률 평가 (유사팀 대비)
    2. KPI별 강점/특징 분석
    3. 팀 성과의 전반적 특성
    4. 간단한 개선 방향 (필요시)
    
    결과는 다음 JSON 형식으로만 응답해주세요:
    {
      "comment": "[250-300자 분량의 팀 성과 분석 코멘트]"
    }
    """
    
    human_prompt = f"""
    <팀 성과 정보>
    종합 달성률: {our_overall_rate}% (유사팀 평균: {cluster_stats['avg_rate']}%)
    클러스터 내 위치: {similar_teams_count}개 유사팀과 비교
    </팀 성과 정보>
    
    <KPI별 상세 분석>
    {kpi_details}
    </KPI별 상세 분석>
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    def validate_comment_response(response: str) -> str:
        try:
            json_output_raw = response
            json_output = _extract_json_from_llm_response(json_output_raw)
            llm_parsed_data = json.loads(json_output)
            
            comment = llm_parsed_data.get("comment", "")
            if not comment:
                raise ValueError("LLM이 빈 코멘트를 반환했습니다.")
            
            return comment
            
        except Exception as e:
            logger.error(f"Comment validation failed: {e}")
            # 폴백 코멘트
            overall_comparison = get_comparison_result_detailed(
                our_overall_rate, cluster_stats
            )
            return f"귀하의 팀은 종합 달성률 {our_overall_rate}%로 유사팀 평균({cluster_stats['avg_rate']}%) 대비 {overall_comparison} 성과를 보이고 있습니다. 유사팀 {similar_teams_count}개와의 비교 결과를 바탕으로 지속적인 성장을 위한 전략 수립이 필요합니다."
    
    chain = prompt | llm_client
    
    try:
        response: AIMessage = chain.invoke({})
        content = str(response.content)
        return validate_comment_response(content)
        
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        # 폴백 코멘트
        overall_comparison = get_comparison_result_detailed(
            our_overall_rate, cluster_stats
        )
        return f"귀하의 팀은 종합 달성률 {our_overall_rate}%로 유사팀 평균({cluster_stats['avg_rate']}%) 대비 {overall_comparison} 성과를 보이고 있습니다. 유사팀 {similar_teams_count}개와의 비교 결과를 바탕으로 지속적인 성장을 위한 전략 수립이 필요합니다."

# ===== DB 저장 함수 =====

def save_team_comparison_results(team_evaluation_id: int, comparison_json: Dict) -> bool:
    """팀 비교 결과 DB 저장"""
    with engine.connect() as connection:
        # comparison_result 추출
        comparison_result = comparison_json.get("overall", {}).get("comparison_result", "")
        
        query = text("""
            UPDATE team_evaluations
            SET 
                ai_team_comparison = :comparison_json,
                relative_performance = :comparison_result
            WHERE team_evaluation_id = :team_evaluation_id
        """)
        
        result = connection.execute(query, {
            "team_evaluation_id": team_evaluation_id,
            "comparison_json": json.dumps(comparison_json, ensure_ascii=False),
            "comparison_result": comparison_result
        })
        connection.commit()
        return result.rowcount > 0

# ===== 서브모듈 함수 정의 =====

def check_cluster_stats_submodule(state: Module8AgentState) -> Module8AgentState:
    """1. 클러스터 통계 존재 확인"""
    period_id = state["period_id"]
    
    # TeamPerformanceComparator 인스턴스 생성
    comparator = TeamPerformanceComparator(cache_dir=CACHE_DIR)
    
    # 클러스터 통계 상태 확인
    status = comparator.get_cluster_status(period_id)
    
    if status["cache_file_exists"]:
        message = f"클러스터 통계 확인 완료: 기존 캐시 사용 (Q{period_id})"
        logger.info(f"✅ 클러스터 캐시 파일 존재 - Q{period_id}")
    else:
        message = f"클러스터 통계 없음: 새로 계산 예정 (Q{period_id})"
        logger.info(f"📊 클러스터 분석 필요 - Q{period_id}")
    
    return {
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=message)]
    }

def calculate_cluster_stats_submodule(state: Module8AgentState) -> Module8AgentState:
    """2. 필요시 전사 클러스터링 + 성과 통계 계산"""
    team_id = state["team_id"]
    period_id = state["period_id"]
    
    logger.info(f"🔄 클러스터 분석 시작 - 팀 {team_id}")
    
    # TeamPerformanceComparator로 클러스터 분석 실행
    comparator = TeamPerformanceComparator(cache_dir=CACHE_DIR)
    result_data = comparator.analyze_team_cluster_performance(team_id, period_id)
    
    if not result_data["success"]:
        logger.error(f"❌ 클러스터 분석 실패: {result_data['error']}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"클러스터 분석 실패: {result_data['error']}")
            ]
        }
    
    team_cluster_info = result_data["team_cluster_info"]
    logger.info(f"✅ 클러스터 분석 완료 - 클러스터 {team_cluster_info['cluster_id']}, 유사팀 {len(team_cluster_info['similar_teams'])}개")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"클러스터 분석 완료: 클러스터 {team_cluster_info['cluster_id']}, 유사팀 {len(team_cluster_info['similar_teams'])}개")
        ],
        "our_team_cluster_id": team_cluster_info["cluster_id"],
        "similar_teams": team_cluster_info["similar_teams"],
        "cluster_stats": team_cluster_info["cluster_stats"]
    }

def team_performance_collection_submodule(state: Module8AgentState) -> Module8AgentState:
    """3. 우리팀 + 유사팀 성과 데이터 수집"""
    team_id = state["team_id"]
    period_id = state["period_id"]
    similar_teams = state["similar_teams"]
    
    logger.info(f"📋 성과 데이터 수집 중 - 팀 {team_id} + 유사팀 {len(similar_teams)}개")
    
    # 우리팀 데이터 수집
    our_team_data = fetch_team_kpis_data(team_id, period_id)
    if not our_team_data:
        logger.error(f"❌ 우리팀 데이터 조회 실패 - 팀 {team_id}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"우리팀 성과 데이터 조회 실패: 팀 {team_id}")
            ]
        }
    
    # 유사팀들 KPI 데이터 수집
    similar_teams_kpis = fetch_multiple_teams_kpis(similar_teams)
    
    logger.info(f"✅ 성과 데이터 수집 완료 - 우리팀 KPI {len(our_team_data['kpis'])}개, 유사팀 KPI {len(similar_teams_kpis)}개")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"팀 성과 데이터 수집 완료: 우리팀 KPI {len(our_team_data['kpis'])}개, 유사팀 KPI {len(similar_teams_kpis)}개")
        ],
        "our_team_kpis": our_team_data["kpis"],
        "our_team_overall_rate": our_team_data["overall_rate"],
        "similar_teams_performance": similar_teams_kpis
    }

def kpi_comparison_submodule(state: Module8AgentState) -> Module8AgentState:
    """4. KPI별 유사도 매칭 + 비교 분석"""
    our_team_kpis = state["our_team_kpis"]
    similar_teams_performance = state["similar_teams_performance"]
    
    logger.info(f"🔍 KPI 비교 분석 중 - {len(our_team_kpis)}개 KPI")
    
    # KPI별 비교 분석 실행
    kpi_comparison_results = compare_kpis_with_similar_teams(our_team_kpis, similar_teams_performance)
    
    # 비교 가능한 KPI 개수 계산
    comparable_kpis = len([kpi for kpi in kpi_comparison_results if kpi["comparison_result"] != "-"])
    
    logger.info(f"✅ KPI 비교 분석 완료 - {len(kpi_comparison_results)}개 중 {comparable_kpis}개 비교 가능")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"KPI 비교 분석 완료: {len(kpi_comparison_results)}개 KPI 중 {comparable_kpis}개 비교 가능")
        ],
        "kpi_comparison_results": kpi_comparison_results
    }

def generate_team_comment_submodule(state: Module8AgentState) -> Module8AgentState:
    """5. LLM 기반 팀 성과 코멘트 생성"""
    our_team_overall_rate = state["our_team_overall_rate"]
    cluster_stats = state["cluster_stats"]
    kpi_comparison_results = state["kpi_comparison_results"]
    similar_teams = state["similar_teams"]
    
    logger.info(f"🤖 LLM 코멘트 생성 중 - 종합 달성률 {our_team_overall_rate}%")
    
    # LLM으로 팀 성과 코멘트 생성
    team_comment = call_llm_for_team_performance_comment(
        our_team_overall_rate, cluster_stats, kpi_comparison_results, len(similar_teams)
    )
    
    # 최종 비교 JSON 구성
    final_comparison_json = {
        "overall": {
            "our_rate": our_team_overall_rate,
            "similar_avg_rate": cluster_stats["avg_rate"],
            "similar_teams_count": len(similar_teams),
            "comparison_result": get_comparison_result_detailed(our_team_overall_rate, cluster_stats),
            "comment": team_comment
        },
        "kpis": kpi_comparison_results
    }
    
    logger.info(f"✅ LLM 코멘트 생성 완료 - {len(team_comment)}자")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"LLM 코멘트 생성 완료 ({len(team_comment)}자)")
        ],
        "team_performance_comment": team_comment,
        "final_comparison_json": final_comparison_json
    }

def save_results_submodule(state: Module8AgentState) -> Module8AgentState:
    """6. JSON 결과 DB 저장"""
    team_id = state["team_id"]
    period_id = state["period_id"]
    final_comparison_json = state["final_comparison_json"]
    
    logger.info(f"💾 DB 저장 중 - 팀 {team_id}")
    
    # team_evaluation_id 조회
    team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
    
    if not team_evaluation_id:
        logger.error(f"❌ team_evaluation_id 조회 실패 - 팀 {team_id}, Q{period_id}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"team_evaluation_id 조회 실패: 팀 {team_id}, 분기 {period_id}")
            ]
        }
    
    # DB 저장
    success = save_team_comparison_results(team_evaluation_id, final_comparison_json)
    
    if success:
        logger.info(f"✅ DB 저장 완료 - team_evaluations[{team_evaluation_id}]")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"DB 저장 완료: team_evaluations[{team_evaluation_id}] 업데이트")
            ],
            "updated_team_evaluation_id": team_evaluation_id
        }
    else:
        logger.error(f"❌ DB 저장 실패 - team_evaluations[{team_evaluation_id}]")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"DB 저장 실패: team_evaluations[{team_evaluation_id}]")
            ]
        }

# ===== LangGraph 워크플로우 구성 =====

# 모듈 8 워크플로우 정의
module8_workflow = StateGraph(Module8AgentState)

# 노드 추가
module8_workflow.add_node("check_cluster_stats", check_cluster_stats_submodule)
module8_workflow.add_node("calculate_cluster_stats", calculate_cluster_stats_submodule)
module8_workflow.add_node("team_performance_collection", team_performance_collection_submodule)
module8_workflow.add_node("kpi_comparison", kpi_comparison_submodule)
module8_workflow.add_node("generate_team_comment", generate_team_comment_submodule)
module8_workflow.add_node("save_results", save_results_submodule)

# 엣지 정의
module8_workflow.add_edge(START, "check_cluster_stats")
module8_workflow.add_edge("check_cluster_stats", "calculate_cluster_stats")
module8_workflow.add_edge("calculate_cluster_stats", "team_performance_collection")
module8_workflow.add_edge("team_performance_collection", "kpi_comparison")
module8_workflow.add_edge("kpi_comparison", "generate_team_comment")
module8_workflow.add_edge("generate_team_comment", "save_results")
module8_workflow.add_edge("save_results", END)

# 모듈 8 그래프 컴파일
module8_graph = module8_workflow.compile()

# ===== 메인 파이프라인 함수 =====

def execute_module8_pipeline(team_id: int, period_id: int, report_type: str = "quarterly") -> Dict[str, Any]:
    """모듈 8 팀 성과 비교 평가 실행"""
    
    start_time = datetime.now()
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 모듈 8: 팀 성과 비교 분석 시작")
    logger.info(f"{'='*60}")
    logger.info(f"📍 설정 정보:")
    logger.info(f"   팀 ID: {team_id}")
    logger.info(f"   기간 ID: {period_id}")
    logger.info(f"   리포트 타입: {report_type}")
    
    try:
        initial_state = {
            "team_id": team_id,
            "period_id": period_id,
            "report_type": report_type,
            "messages": []
        }
        
        logger.info(f"🚀 모듈 8: 팀 성과 비교 분석 시작 (팀 {team_id}, Q{period_id})")
        
        result = module8_graph.invoke(initial_state)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        success_result = {
            "status": "success",
            "execution_time_seconds": execution_time,
            "team_id": team_id,
            "period_id": period_id,
            "report_type": report_type,
            "results": {
                "cluster_id": result.get("our_team_cluster_id"),
                "similar_teams_count": len(result.get("similar_teams", [])),
                "overall_rate": result.get("our_team_overall_rate"),
                "comment_length": len(result.get("team_performance_comment", "")),
                "kpi_comparisons": len(result.get("kpi_comparison_results", [])),
                "updated_team_evaluation_id": result.get("updated_team_evaluation_id")
            },
            "messages": [msg.content for msg in result.get("messages", [])]
        }
        
        logger.info("\n✅ 모듈 8 실행 완료!")
        logger.info("📋 실행 과정:")
        for i, message in enumerate(result.get("messages", []), 1):
            logger.info(f"  {i}. {message.content}")
        
        logger.info(f"\n📊 최종 결과:")
        logger.info(f"클러스터 ID: {success_result['results']['cluster_id']}")
        logger.info(f"유사팀 수: {success_result['results']['similar_teams_count']}")
        logger.info(f"종합 달성률: {success_result['results']['overall_rate']}%")
        logger.info(f"코멘트 길이: {success_result['results']['comment_length']}자")
        logger.info(f"KPI 비교 결과: {success_result['results']['kpi_comparisons']}개")
        logger.info(f"업데이트된 team_evaluation_id: {success_result['results']['updated_team_evaluation_id']}")
        logger.info(f"⏱️  실행 시간: {execution_time:.2f}초")
        logger.info(f"{'='*60}")
        
        return success_result
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_result = {
            "status": "error",
            "execution_time_seconds": execution_time,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "team_id": team_id,
            "period_id": period_id
        }
        
        logger.error(f"\n❌ 모듈 8 실행 실패!")
        logger.error(f"⏱️  실행 시간: {execution_time:.2f}초")
        logger.error(f"💥 오류: {e}")
        logger.error(f"🔍 오류 유형: {type(e).__name__}")
        logger.error(f"{'='*60}")
        
        return error_result

# ===== 테스트 및 실행 함수 =====

def test_module8() -> Optional[Dict]:
    """모듈 8 테스트 실행"""
    logger.info("=== 모듈 8 테스트 시작 ===")
    
    # 기본 테스트
    result = execute_module8_pipeline(team_id=1, period_id=2, report_type="quarterly")
    
    if result and result.get("status") == "success":
        logger.info(f"\n📊 최종 테스트 결과:")
        logger.info(f"상태: {result['status']}")
        logger.info(f"실행 시간: {result['execution_time_seconds']:.2f}초")
        logger.info(f"클러스터 ID: {result['results']['cluster_id']}")
        logger.info(f"유사팀 수: {result['results']['similar_teams_count']}")
        logger.info(f"종합 달성률: {result['results']['overall_rate']}%")
        logger.info(f"코멘트 길이: {result['results']['comment_length']}자")
        logger.info(f"KPI 비교 결과: {result['results']['kpi_comparisons']}개")
        logger.info(f"업데이트된 team_evaluation_id: {result['results']['updated_team_evaluation_id']}")
        
        return result
    else:
        logger.error("테스트 실패!")
        if result:
            logger.error(f"오류: {result.get('error_message', 'Unknown error')}")
            logger.error(f"오류 유형: {result.get('error_type', 'Unknown')}")
        return None

# ===== 메인 실행 부분 =====

if __name__ == "__main__":
    # 테스트 케이스들
    test_cases = [
        {"team_id": 1, "period_id": 2, "report_type": "quarterly", "desc": "Q2 분기별"},
        {"team_id": 1, "period_id": 4, "report_type": "annual_manager", "desc": "Q4 연말"}
    ]

    for test_case in test_cases:
        logger.info(f"\n🧪 모듈8 테스트 실행 - {test_case['desc']}")
        logger.info(f"   테스트 팀: {test_case['team_id']}")
        logger.info(f"   테스트 기간: Q{test_case['period_id']}")
        logger.info(f"   리포트 타입: {test_case['report_type']}")
        
        try:
            result = execute_module8_pipeline(
                test_case['team_id'], 
                test_case['period_id'], 
                test_case['report_type']
            )
            
            if result.get('status') == 'success':
                logger.info(f"\n🎉 테스트 성공!")
                logger.info(f"📊 실행 결과 요약:")
                logger.info(f"   • 상태: {result['status']}")
                logger.info(f"   • 실행 시간: {result['execution_time_seconds']:.2f}초")
                logger.info(f"   • 클러스터 ID: {result['results']['cluster_id']}")
                logger.info(f"   • 유사팀 수: {result['results']['similar_teams_count']}")
                logger.info(f"   • 종합 달성률: {result['results']['overall_rate']}%")
                logger.info(f"   • 코멘트 길이: {result['results']['comment_length']}자")
                logger.info(f"   • KPI 비교 수: {result['results']['kpi_comparisons']}개")
                logger.info(f"   • 업데이트된 team_evaluation_id: {result['results']['updated_team_evaluation_id']}")
            else:
                logger.error(f"\n❌ 테스트 실패!")
                logger.error(f"   • 오류: {result.get('error_message', 'Unknown error')}")
                logger.error(f"   • 오류 유형: {result.get('error_type', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"\n💥 테스트 실행 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info("🏁 모듈8 테스트 완료")
    logger.info(f"{'='*60}")