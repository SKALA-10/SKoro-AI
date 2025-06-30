# ================================================================
# calculation_utils_module2.py - 모듈 2 계산 관련 유틸리티
# ================================================================

import re
import logging
from typing import List, Dict, Optional
from agents.evaluation.modules.module_02_goal_achievement.db_utils import *

logger = logging.getLogger(__name__)

# ================================================================
# 유틸리티 함수들
# ================================================================

def extract_number_from_response(response: str) -> float:
    """응답에서 숫자 추출"""
    patterns = [
        r'^(\d+(?:\.\d+)?)$',           # "85", "85.5"
        r'(\d+(?:\.\d+)?)%',            # "85%"  
        r'(\d+(?:\.\d+)?)\s*점',         # "85점"
        r':(\d+(?:\.\d+)?)(?:[:%]|$)',  # "1:85", "1:85%"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            return float(match.group(1))
    
    raise ValueError(f"No valid number found in response: {response}")

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """안전한 나누기"""
    if denominator == 0:
        print(f"   ⚠️  나누기 오류: {numerator}/{denominator}, 기본값 {default} 사용")
        return default
    return numerator / denominator

def calculate_weighted_average(values: List[float], weights: List[float], default: float = 0) -> float:
    """가중평균 계산: Σ(값 × 가중치) / Σ(가중치)"""
    if not values or not weights or len(values) != len(weights):
        return default
    
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights)
    
    return safe_divide(weighted_sum, total_weight, default)

def calculate_individual_weighted_achievement_rate(individual_tasks: List[Dict]) -> Dict[str, float]:
    """개인별 가중평균 달성률 계산"""
    if not individual_tasks:
        return {"achievement_rate": 0, "total_weight": 0}
    
    # 달성률 가중평균 계산
    achievement_rates = []
    weights = []
    
    for task in individual_tasks:
        achievement_rate = task.get('ai_achievement_rate', 0)
        weight = task.get('weight', 0)
        
        achievement_rates.append(achievement_rate)
        weights.append(weight)
    
    # 가중평균 달성률
    weighted_achievement = calculate_weighted_average(achievement_rates, weights, 0)
    
    # 기여도는 하이브리드 3단계에서 이미 계산되어 저장되므로 제거
    # contribution_rates = [task.get('ai_contribution_score', 0) for task in individual_tasks]
    # avg_contribution = sum(contribution_rates) / len(contribution_rates) if contribution_rates else 0
    
    total_weight = sum(weights)
    
    return {
        "achievement_rate": weighted_achievement,
        # "contribution_rate": avg_contribution,  # 삭제 - 하이브리드 3단계에서 이미 계산됨
        "total_weight": total_weight
    }

# ================================================================
# 평가 기준 처리
# ================================================================

def get_evaluation_criteria(team_kpi_id: int) -> List[str]:
    """우리가 상의한 평가 기준 처리 로직"""
    from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_team_kpi_data
    
    kpi_data = fetch_team_kpi_data(team_kpi_id)
    grade_rule = kpi_data.get('grade_rule')
    
    if grade_rule and grade_rule.strip():
        criteria = parse_criteria_from_grade_rule(grade_rule)
        if criteria:
            return criteria
    
    # 기본 평가 기준
    return ["목표달성 기여도", "성과 영향력", "업무 완성도"]

def parse_criteria_from_grade_rule(grade_rule: str) -> Optional[List[str]]:
    """grade_rule에서 평가 기준 추출"""
    if not grade_rule or not grade_rule.strip():
        return None
    
    lines = grade_rule.strip().split('\n')
    criteria = []
    
    for line in lines:
        line = line.strip()
        # "- " 또는 "• " 제거하고 내용 추출
        match = re.match(r'^[-•]\s*(.+)$', line)
        if match:
            criteria.append(match.group(1).strip())
        elif line and not line.startswith(('-', '•')):
            criteria.append(line)
    
    # 너무 많은 기준은 제한
    if len(criteria) > 5:
        criteria = criteria[:5]
        
    return criteria if criteria else None

# ================================================================
# 기여도 계산 함수들
# ================================================================

def calculate_quantitative_contributions(kpi_id: int, period_id: int) -> Dict[str, float]:
    """정량 평가 기여도 계산"""
    from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_kpi_tasks
    
    tasks = fetch_kpi_tasks(kpi_id, period_id)
    
    # 개인별 성과 수집
    emp_performance = {}
    for task in tasks:
        emp_no = task['emp_no']
        performance_text = task.get('task_performance', '')
        
        # 성과에서 수치 추출 시도
        try:
            performance_value = extract_number_from_response(performance_text)
            if emp_no not in emp_performance:
                emp_performance[emp_no] = 0
            emp_performance[emp_no] += performance_value
        except:
            # 수치 추출 실패시 동등 분배
            emp_performance[emp_no] = 1.0
    
    # 기여도 계산
    total_performance = sum(emp_performance.values())
    contributions = {}
    
    for emp_no, performance in emp_performance.items():
        contribution_rate = safe_divide(performance, total_performance, 1/len(emp_performance)) * 100
        contributions[emp_no] = round(contribution_rate, 2)
    
    return contributions

# ================================================================
# 팀 분석 계산 함수들
# ================================================================

def calculate_team_average_achievement_rate(team_kpi_ids: List[int]) -> float:
    """팀 전체 평균 달성률 계산 (KPI 비중 고려)"""
    
    total_weight = 0
    weighted_sum = 0
    
    for kpi_id in team_kpi_ids:
        kpi_data = fetch_team_kpi_data(kpi_id)
        weight = kpi_data.get('weight', 0)
        rate = kpi_data.get('ai_kpi_progress_rate', 0)
        
        total_weight += weight
        weighted_sum += rate * weight
    
    return safe_divide(weighted_sum, total_weight, 80.0)