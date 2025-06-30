# ================================================================
# comparison_utils_module8.py - 모듈 8 비교 분석 관련 유틸리티
# ================================================================

import statistics
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from agents.evaluation.modules.module_08_team_comparision.comparison_utils import *

# ================================================================
# KPI 비교 분석 함수들
# ================================================================

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