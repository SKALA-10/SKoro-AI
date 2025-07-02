
# ================================================================
# agent_1.py - 모듈 9 상태 관리 데이터클래스, 메인 에이전트 클래스
# ================================================================

import os
import logging
import statistics
from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END
from decimal import Decimal
import math
from datetime import datetime

from db_utils_1 import *
from llm_utils_1 import *

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 상태 정의
# ================================================================

class Module9AgentState(TypedDict):
    """모듈 9 (본부 단위 CL별 제로섬 조정) 상태 - 향상된 버전"""
    messages: Annotated[List[HumanMessage], operator.add]
    
    # 입력 정보
    headquarter_id: int
    period_id: int  # 연말: 4
    
    # 4단계 결과 저장 (확장)
    department_data: Dict[str, Dict]           # 1단계: 부문 데이터 수집 결과
    enhanced_analysis: Dict[str, Dict]         # 2단계: 향상된 타당성 분석 결과
    supervisor_results: Dict[str, Dict]        # 3단계: AI Supervisor 실행 결과  
    update_results: Dict                       # 4단계: 배치 업데이트 결과
    
    # 처리 상태
    total_processed: int
    total_failed: int
    error_logs: List[str]

# ================================================================
# 유틸리티 함수들
# ================================================================

def calculate_target_total(member_count: int) -> float:
    """CL별 목표 총점 계산 (인원수 × 3.5점)"""
    return member_count * 3.5

def calculate_surplus(manager_score_sum: float, target_total: float) -> float:
    """초과분 계산 (팀장 수정 점수합 - 목표 총점)"""
    return round(manager_score_sum - target_total, 2)

def get_cl_target_stdev(cl_group: str) -> float:
    """CL별 목표 표준편차 반환"""
    target_stdevs = {
        "CL3": 1.7,  # 고위직 → 큰 변별력
        "CL2": 1.5,  # 중간직 → 적당한 변별력  
        "CL1": 1.4   # 주니어 → 안정적 평가
    }
    return target_stdevs.get(cl_group, 1.5)

def calculate_task_complexity_factor(member: Dict) -> float:
    """업무 복잡도 및 난이도 고려 요소"""
    
    task_data = member.get('task_data', [])
    if not task_data:
        return 1.0  # 기본값
    
    complexity_factors = []
    
    for task in task_data:
        task_weight = task.get('task_weight', 1)
        kpi_weight = task.get('kpi_weight', 1)
        ai_grade = task.get('ai_assessed_grade', 'C')
        
        # 업무 중요도 (가중치)
        importance_factor = min(2.0, (task_weight + kpi_weight) / 10)
        
        # AI 평가 등급 고려
        grade_factors = {'S': 1.5, 'A': 1.3, 'B': 1.1, 'C': 1.0, 'D': 0.8}
        grade_factor = grade_factors.get(ai_grade, 1.0)
        
        complexity_factors.append(importance_factor * grade_factor)
    
    # 평균 복잡도 반환
    return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 1.0

def calculate_enhanced_captain_validity(member: Dict) -> Dict:
    """향상된 팀장 수정 타당성 계산 (업무+동료평가 포함) - 변경 없음은 만점 처리"""
    
    # 변경하지 않은 경우 = 타당성 만점 (팀장이 적절하다고 판단)
    if not member.get('changed_by_manager', True):
        return {
            "basic_validity": 1.0,
            "task_evidence": 1.0,
            "peer_consistency": 1.0,
            "complexity_factor": 1.0,
            "final_validity": 1.0,
            "validity_grade": "매우 타당",
            "detailed_analysis": {
                "direction_score": 1.0,
                "reason_score": 1.0,
                "magnitude_score": 1.0,
                "complexity_adjustment": 0.0
            },
            "no_change_reason": "팀장이 모듈7 점수를 적절하다고 판단하여 변경하지 않음"
        }
    
    # 변경한 경우: 기존 향상된 타당성 분석 수행
    
    # 1. 기본 타당성 (기존 로직)
    score_change = member.get('score_diff', 0)
    reason = member.get('captain_reason', '') or ''
    kpi_achievement = member.get('kpi_achievement', 100)
    
    # 기본 방향성 일치 점수
    if kpi_achievement >= 100 and score_change > 0:
        direction_score = 1.0
    elif kpi_achievement >= 90 and score_change >= 0:
        direction_score = 0.9
    elif kpi_achievement < 80 and score_change < 0:
        direction_score = 1.0
    elif kpi_achievement < 90 and score_change <= 0:
        direction_score = 0.8
    elif abs(score_change) < 0.1:
        direction_score = 1.0
    elif (kpi_achievement >= 100 and score_change < -0.3) or (kpi_achievement < 70 and score_change > 0.3):
        direction_score = 0.2
    else:
        direction_score = 0.5
    
    # 기본 사유 품질 점수
    reason_length = len(reason.strip()) if reason else 0
    performance_keywords = ['성과', '기여', '우수', '개선', '달성', '노력', '역량', '협업', '리더십']
    specific_keywords = ['프로젝트', '고객', '매출', '품질', '효율', '혁신', '멘토링']
    
    keyword_count = sum(1 for word in performance_keywords if word in reason)
    specific_count = sum(1 for word in specific_keywords if word in reason)
    
    if reason_length > 30 and keyword_count >= 2 and specific_count >= 1:
        reason_score = 1.0
    elif reason_length > 20 and keyword_count >= 1:
        reason_score = 0.8
    elif reason_length > 10:
        reason_score = 0.6
    elif reason_length > 0:
        reason_score = 0.4
    else:
        reason_score = 0.1
    
    # 조정 폭 적절성
    abs_change = abs(score_change)
    if abs_change <= 0.3:
        magnitude_score = 1.0
    elif abs_change <= 0.6:
        magnitude_score = 0.8
    elif abs_change <= 1.0:
        magnitude_score = 0.6
    elif abs_change <= 1.5:
        magnitude_score = 0.3
    else:
        magnitude_score = 0.1
    
    # 2. 업무 증거 일치성 (신규)
    task_evidence = analyze_task_evidence_consistency(member)
    
    # 3. 동료평가 일치성 (신규)
    peer_consistency = analyze_peer_evaluation_consistency(member)
    
    # 4. 업무 복잡도 고려 (신규)
    complexity_factor = calculate_task_complexity_factor(member)
    
    # 5. 종합 타당성 계산 (향상된 공식)
    basic_validity = (
        direction_score * 0.3 +      # 방향성 30%
        reason_score * 0.2 +         # 사유 품질 20%
        magnitude_score * 0.1        # 조정 폭 10%
    )
    
    enhanced_validity = (
        basic_validity * 0.4 +       # 기본 타당성 40%
        task_evidence * 0.3 +        # 업무 증거 30%
        peer_consistency * 0.3       # 동료평가 일치성 30%
    )
    
    # 복잡도 보정 (최대 ±20%)
    complexity_adjustment = min(0.2, max(-0.2, (complexity_factor - 1.0) * 0.2))
    final_validity = max(0.0, min(1.0, enhanced_validity + complexity_adjustment))
    
    return {
        "basic_validity": round(basic_validity, 3),
        "task_evidence": round(task_evidence, 3),
        "peer_consistency": round(peer_consistency, 3),
        "complexity_factor": round(complexity_factor, 3),
        "final_validity": round(final_validity, 3),
        "validity_grade": get_validity_grade(final_validity),
        "detailed_analysis": {
            "direction_score": round(direction_score, 3),
            "reason_score": round(reason_score, 3),
            "magnitude_score": round(magnitude_score, 3),
            "complexity_adjustment": round(complexity_adjustment, 3)
        }
    }

def get_validity_grade(validity_score: float) -> str:
    """타당성 점수를 등급으로 변환"""
    if validity_score >= 0.8:
        return "매우 타당"
    elif validity_score >= 0.6:
        return "타당"
    elif validity_score >= 0.4:
        return "보통"
    elif validity_score >= 0.2:
        return "의심"
    else:
        return "매우 의심"

def calculate_comprehensive_performance_score(member: Dict) -> float:
    """종합 성과 점수 계산 (KPI + 업무증거 + 동료평가 타당성)"""
    
    kpi_score = member.get('kpi_achievement', 100)
    validity_analysis = calculate_enhanced_captain_validity(member)
    final_validity = validity_analysis['final_validity']
    
    # KPI 달성률 70% + 타당성 30%로 종합 성과 계산
    comprehensive_score = kpi_score * 0.7 + final_validity * 100 * 0.3
    
    return round(comprehensive_score, 1)

def check_performance_reversal(adjustments: List[Dict], kpi_data: Dict) -> Dict:
    """성과 역전 방지 검증 - KPI 차이 20%p 이상 시 점수 역전 불가"""
    
    reversals = []
    warnings = []
    
    # 모든 조합을 체크
    for i in range(len(adjustments)):
        for j in range(i + 1, len(adjustments)):
            emp1 = adjustments[i]
            emp2 = adjustments[j]
            
            # KPI 달성률 가져오기
            kpi1 = kpi_data.get(emp1["emp_no"], 100)
            kpi2 = kpi_data.get(emp2["emp_no"], 100)
            
            # KPI 차이 계산
            kpi_diff = abs(kpi1 - kpi2)
            
            # 20%p 이상 차이나는 경우만 체크
            if kpi_diff >= 20:
                # 누가 더 높은 KPI인지 확인
                if kpi1 > kpi2:
                    high_kpi_emp, low_kpi_emp = emp1, emp2
                    high_kpi, low_kpi = kpi1, kpi2
                else:
                    high_kpi_emp, low_kpi_emp = emp2, emp1
                    high_kpi, low_kpi = kpi2, kpi1
                
                # 점수 역전 체크
                if high_kpi_emp["final_score"] < low_kpi_emp["final_score"]:
                    reversals.append({
                        "high_kpi_emp": high_kpi_emp["emp_no"],
                        "low_kpi_emp": low_kpi_emp["emp_no"],
                        "high_kpi": high_kpi,
                        "low_kpi": low_kpi,
                        "kpi_diff": kpi_diff,
                        "high_kpi_score": high_kpi_emp["final_score"],
                        "low_kpi_score": low_kpi_emp["final_score"],
                        "score_diff": low_kpi_emp["final_score"] - high_kpi_emp["final_score"]
                    })
                elif high_kpi_emp["final_score"] == low_kpi_emp["final_score"]:
                    warnings.append({
                        "high_kpi_emp": high_kpi_emp["emp_no"],
                        "low_kpi_emp": low_kpi_emp["emp_no"],
                        "kpi_diff": kpi_diff,
                        "message": "KPI 차이 큰데 동점"
                    })
    
    return {
        "has_reversal": len(reversals) > 0,
        "reversal_count": len(reversals),
        "reversals": reversals,
        "warnings": warnings
    }

def validate_zero_sum_result(adjustments: List[Dict], target_reduction: float, 
                           target_stdev: float, cl_group: str) -> Dict:
    """제로섬 조정 결과 검증"""
    
    if not adjustments:
        return {"valid": False, "errors": ["조정 결과가 없습니다"]}
    
    # 기본 검증
    final_scores = [adj["final_score"] for adj in adjustments]
    actual_total = sum(final_scores)
    actual_mean = actual_total / len(final_scores)
    actual_stdev = statistics.stdev(final_scores) if len(final_scores) > 1 else 0
    
    # 실제 차감량 계산
    actual_reduction = sum(adj["original_score"] - adj["final_score"] for adj in adjustments)
    
    errors = []
    warnings = []
    
    # 1. 제로섬 검증
    reduction_error = abs(actual_reduction - target_reduction)
    if reduction_error > 0.02:
        errors.append(f"제로섬 실패: 목표 {target_reduction:.2f}, 실제 {actual_reduction:.2f}")
    
    # 2. 평균 검증
    mean_error = abs(actual_mean - 3.5)
    if mean_error > 0.02:
        errors.append(f"평균 실패: 목표 3.5, 실제 {actual_mean:.2f}")
    
    # 3. 표준편차 검증 (더 관대하게)
    stdev_error = abs(actual_stdev - target_stdev)
    if stdev_error > 0.4:
        errors.append(f"표준편차 실패: 목표 {target_stdev:.1f}, 실제 {actual_stdev:.1f}")
    elif stdev_error > 0.2:
        warnings.append(f"표준편차 경고: 목표 {target_stdev:.1f}, 실제 {actual_stdev:.1f}")
    
    # 4. 점수 범위 검증
    invalid_scores = [adj for adj in adjustments if adj["final_score"] < 0.0 or adj["final_score"] > 5.0]
    if invalid_scores:
        errors.append(f"점수 범위 초과: {len(invalid_scores)}명")
    
    # 5. 성과 역전 검증
    kpi_data = {adj["emp_no"]: adj.get("kpi_achievement", 100) for adj in adjustments}
    performance_reversal = check_performance_reversal(adjustments, kpi_data)
    
    if performance_reversal["has_reversal"]:
        errors.append(f"성과 역전 발생: {performance_reversal['reversal_count']}건")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "performance_reversal": performance_reversal,
        "metrics": {
            "actual_reduction": round(actual_reduction, 2),
            "target_reduction": round(target_reduction, 2),
            "actual_mean": round(actual_mean, 2),
            "target_mean": 3.5,
            "actual_stdev": round(actual_stdev, 2),
            "target_stdev": round(target_stdev, 2),
            "reduction_error": round(reduction_error, 2),
            "mean_error": round(mean_error, 2),
            "stdev_error": round(stdev_error, 2)
        }
    }

# ================================================================
# Fallback 알고리즘
# ================================================================

def execute_proper_zero_sum_adjustment(supervisor_input: Dict) -> Dict:
    """올바른 제로섬 조정: 성과 기반 차등 차감으로 표준편차 달성 + 변경 없음 우선순위 반영"""
    
    members = supervisor_input["members"]
    target_total = supervisor_input["current_situation"]["target_total"]
    surplus = supervisor_input["total_surplus"]
    target_mean = 3.5
    target_stdev = supervisor_input.get("distribution_targets", {}).get("target_stdev", 1.5)
    member_count = len(members)
    cl_group = supervisor_input.get("cl_group", "CL")
    
    print(f"🔧 {cl_group} 올바른 제로섬 조정 실행:")
    print(f"  목표: 평균 {target_mean}, 표준편차 {target_stdev}, 차감 {surplus:.2f}점")
    
    if member_count == 0:
        return {"analysis_summary": "처리할 멤버 없음", "adjustments": [], "validation_check": {"all_conditions_met": False}}
    
    # 1. 변경 우선순위별 분류
    high_priority = [m for m in members if m.get('change_priority') == 'high']      # 팀장이 변경함
    maintain_priority = [m for m in members if m.get('change_priority') == 'maintain']  # 팀장이 유지함 (변경 없음)
    
    print(f"  우선순위 분류:")
    print(f"    주요 조정 대상: {len(high_priority)}명")
    print(f"    유지 우선 대상: {len(maintain_priority)}명 (타당성 만점)")
    
    # 2. 향상된 성과 점수 계산 및 정렬
    members_with_performance = []
    for member in members:
        validity_analysis = calculate_enhanced_captain_validity(member)
        comprehensive_score = calculate_comprehensive_performance_score(member)
        
        member_copy = member.copy()
        member_copy["validity_analysis"] = validity_analysis
        member_copy["comprehensive_performance"] = comprehensive_score
        members_with_performance.append(member_copy)
    
    # 성과 순위로 정렬 (높은 순)
    sorted_members = sorted(members_with_performance, 
                          key=lambda x: x["comprehensive_performance"], 
                          reverse=True)
    
    print(f"  성과 순위 (상위 3명):")
    for i, member in enumerate(sorted_members[:3]):
        validity_grade = member["validity_analysis"]["validity_grade"]
        change_status = "변경함" if member.get('changed_by_manager', True) else "유지함"
        print(f"    {i+1}위: {member['emp_no']} | 종합성과 {member['comprehensive_performance']:.1f}, 현재점수 {member['current_score']:.1f}, 타당성 {validity_grade}, {change_status}")
    
    # 3. 현재 표준편차 계산
    current_scores = [m["current_score"] for m in sorted_members]
    current_stdev = statistics.stdev(current_scores) if len(current_scores) > 1 else 0
    current_mean = sum(current_scores) / len(current_scores)
    
    print(f"  현재 상태: 평균 {current_mean:.2f}, 표준편차 {current_stdev:.2f}")
    
    # 4. 차등 조정 가중치 생성 (변경 우선순위 반영)
    adjustment_weights = generate_performance_based_weights_with_priority(
        sorted_members, target_stdev, current_stdev, surplus
    )
    
    # 5. 제로섬 차감 분배
    adjustments = []
    total_reduction = 0.0
    
    for i, member in enumerate(sorted_members):
        # 개별 차감량 계산
        individual_reduction = surplus * adjustment_weights[i] / sum(adjustment_weights)
        final_score = member["current_score"] - individual_reduction
        
        # 0-5 범위 제한
        final_score = max(0.0, min(5.0, final_score))
        actual_reduction = member["current_score"] - final_score
        total_reduction += actual_reduction
        
        change_type = "maintain" if abs(actual_reduction) < 0.01 else "decrease"
        
        # 조정 사유 생성
        rank = i + 1
        performance_tier = "상위" if i < member_count * 0.3 else "중위" if i < member_count * 0.7 else "하위"
        validity_grade = member["validity_analysis"]["validity_grade"]
        change_status = "변경함" if member.get('changed_by_manager', True) else "유지함"
        
        if member.get('change_priority') == 'maintain':
            adjustment_reason = f"성과기반 조정: {rank}위/{member_count} ({performance_tier}군, 타당성 {validity_grade}, 팀장 {change_status} - 최소조정)"
        else:
            adjustment_reason = f"성과기반 조정: {rank}위/{member_count} ({performance_tier}군, 타당성 {validity_grade}, 팀장 {change_status})"
        
        adjustments.append({
            "emp_no": member["emp_no"],
            "original_score": member["current_score"],
            "final_score": round(final_score, 2),
            "change_amount": round(-actual_reduction, 2),  # 음수 (차감)
            "change_type": change_type,
            "reason": adjustment_reason,
            "final_evaluation_report_id": member.get("final_evaluation_report_id"),
            "performance_rank": rank,
            "performance_tier": performance_tier,
            "validity_analysis": member["validity_analysis"],
            "kpi_achievement": member.get("kpi_achievement", 100),  # 성과 역전 검증용
            "change_priority": member.get("change_priority", "high")
        })
    
    # 6. 최종 검증
    validation_result = validate_zero_sum_result(adjustments, surplus, target_stdev, cl_group)
    
    print(f"  결과: 총차감 {validation_result['metrics']['actual_reduction']:.2f}점")
    print(f"  검증: {'✅ 통과' if validation_result['valid'] else '❌ 실패'}")
    
    if not validation_result["valid"]:
        for error in validation_result["errors"][:2]:  # 상위 2개 에러만 출력
            print(f"    - {error}")
    
    # 성과 역전 체크
    if validation_result["performance_reversal"]["has_reversal"]:
        print(f"  ⚠️ 성과 역전 발생: {validation_result['performance_reversal']['reversal_count']}건")
    
    return {
        "analysis_summary": f"{cl_group} 성과기반 조정 (우선순위 반영): {member_count}명 → 평균 {validation_result['metrics']['actual_mean']:.2f}, 표준편차 {validation_result['metrics']['actual_stdev']:.2f}",
        "adjustments": adjustments,
        "validation_check": {
            "target_total": target_total,
            "actual_total": validation_result["metrics"]["actual_mean"] * member_count,
            "target_mean": target_mean,
            "actual_mean": validation_result["metrics"]["actual_mean"],
            "target_stdev": target_stdev,
            "actual_stdev": validation_result["metrics"]["actual_stdev"],
            "target_reduction": surplus,
            "actual_reduction": validation_result["metrics"]["actual_reduction"],
            "zero_sum_achieved": validation_result["metrics"]["reduction_error"] < 0.02,
            "stdev_achieved": validation_result["metrics"]["stdev_error"] < 0.3,
            "mean_achieved": validation_result["metrics"]["mean_error"] < 0.02,
            "all_conditions_met": validation_result["valid"],
            "performance_reversal": validation_result["performance_reversal"],
            "priority_analysis": {
                "high_priority_count": len(high_priority),
                "maintain_priority_count": len(maintain_priority),
                "high_priority_members": [m["emp_no"] for m in high_priority],
                "maintain_priority_members": [m["emp_no"] for m in maintain_priority]
            }
        }
    }

def generate_performance_based_weights_with_priority(sorted_members: List[Dict], target_stdev: float, 
                                                   current_stdev: float, surplus: float) -> List[float]:
    """성과 기반 차등 차감 가중치 생성 - 변경 우선순위 반영"""
    
    member_count = len(sorted_members)
    
    # 표준편차 조정 필요성 계산
    stdev_factor = calculate_stdev_adjustment_factor(current_stdev, target_stdev)
    
    # 기본 전략 결정
    if stdev_factor <= 0.3:
        strategy = "uniform_with_priority"
        print(f"  표준편차 전략: 우선순위 반영 균등차감 (factor: {stdev_factor:.2f})")
    else:
        strategy = "differential_with_priority"
        print(f"  표준편차 전략: 우선순위 반영 차등차감 (factor: {stdev_factor:.2f})")
    
    base_weights = []
    
    for i in range(member_count):
        member = sorted_members[i]
        rank_ratio = i / max(1, member_count - 1)  # 0(1위) ~ 1(꼴찌)
        
        # 변경 우선순위 반영
        if member.get('change_priority') == 'maintain':
            # 변경하지 않은 직원: 최소 가중치 (거의 조정 안함)
            base_weight = 0.1
        else:
            # 변경한 직원: 성과와 stdev_factor 기반 가중치
            if strategy == "uniform_with_priority":
                base_weight = 1.0  # 기본 가중치
            else:
                # 차등 가중치
                if rank_ratio <= 0.3:  # 상위 30%
                    base_weight = 1.0 - (0.6 * stdev_factor)  # 최소 0.4까지
                elif rank_ratio <= 0.7:  # 중위 40%
                    base_weight = 1.0  # 기준점
                else:  # 하위 30%
                    base_weight = 1.0 + (0.6 * stdev_factor)  # 최대 1.6까지
        
        base_weights.append(base_weight)
    
    # 타당성 보정
    adjusted_weights = []
    for i, member in enumerate(sorted_members):
        base_weight = base_weights[i]
        validity_score = member["validity_analysis"]["final_validity"]
        
        # 변경하지 않은 직원은 타당성 보정 제외 (이미 최소 가중치)
        if member.get('change_priority') == 'maintain':
            validity_modifier = 1.0  # 보정 없음
        else:
            # 타당성이 높으면 차감 보호, 낮으면 더 차감
            if validity_score >= 0.7:  # 고타당성
                validity_modifier = 0.8
            elif validity_score >= 0.4:  # 중타당성
                validity_modifier = 1.0
            else:  # 저타당성
                validity_modifier = 1.2
        
        adjusted_weight = base_weight * validity_modifier
        adjusted_weights.append(adjusted_weight)
    
    return adjusted_weights

def calculate_stdev_adjustment_factor(current_stdev: float, target_stdev: float) -> float:
    """표준편차 조정 필요성 계산"""
    
    if target_stdev == 0:
        return 1.0
    
    stdev_ratio = current_stdev / target_stdev
    
    if stdev_ratio >= 1.2:
        # 현재 변별력이 과도함 → 균등 차감
        return 0.0  # 차등 없음
    elif stdev_ratio >= 0.8:
        # 적정 변별력 → 약간 차등
        return 0.3
    else:
        # 변별력 부족 → 강한 차등
        return 1.0

# ================================================================
# 필터링 함수
# ================================================================

def fetch_headquarter_cl_members_enhanced_filtered(headquarter_id: int, cl_group: str, period_id: int) -> List[Dict]:
    """본부 내 특정 CL 그룹의 직원 데이터 조회 - 팀장 변경분만 필터링"""
    
    # 모든 직원 조회
    all_members = fetch_headquarter_cl_members_enhanced(headquarter_id, cl_group, period_id)
    
    # 변경 여부 분석 (제외하지 않고 분류만)
    changed_count = 0
    unchanged_count = 0
    
    for member in all_members:
        baseline_score = safe_decimal_to_float(member.get('baseline_score', 0))
        manager_score = safe_decimal_to_float(member.get('manager_score', 0))
        score_diff = abs(manager_score - baseline_score)
        
        if score_diff > 0.01:
            member['changed_by_manager'] = True
            member['score_diff'] = round(manager_score - baseline_score, 2)
            member['change_priority'] = 'high'    # 주요 조정 대상
            changed_count += 1
        else:
            member['changed_by_manager'] = False
            member['score_diff'] = 0.0
            member['change_priority'] = 'maintain'  # 유지 우선
            unchanged_count += 1
    
    # 모든 멤버에 전체 정보 추가
    total_members_count = len(all_members)
    
    for member in all_members:
        member['total_cl_members'] = total_members_count
        member['unchanged_cl_members'] = unchanged_count
    
    print(f"   📊 {cl_group} 팀장 변경 분석:")
    print(f"     - 전체 직원: {len(all_members)}명")
    print(f"     - 변경된 직원: {changed_count}명")
    print(f"     - 변경 없는 직원: {unchanged_count}명")
    
    if unchanged_count > 0:
        unchanged_list = [m['emp_no'] for m in all_members if not m['changed_by_manager']]
        print(f"     - 변경 없음: {', '.join(unchanged_list)} (타당성 만점)")
    
    # 모든 직원 반환 (제외하지 않음)
    return all_members

# ================================================================
# 서브모듈 함수 정의
# ================================================================

def department_data_collection(state: Module9AgentState) -> Module9AgentState:
    """1단계: 부문 데이터 수집 - 팀장 변경분만 처리"""
    
    headquarter_id = state["headquarter_id"]
    period_id = state["period_id"]
    
    try:
        print(f"🔍 1단계: 본부 {headquarter_id} 팀장 변경분 데이터 수집 시작")
        
        # 1. 본부 내 모든 CL 그룹 조회
        cl_groups = get_all_cl_groups_in_headquarter(headquarter_id, period_id)
        print(f"   발견된 CL 그룹: {cl_groups}")
        
        department_data = {}
        total_members = 0
        total_changed_members = 0
        adjustment_needed_cls = []
        
        # 2. CL별 필터링된 데이터 수집
        for cl_group in cl_groups:
            print(f"\n📊 {cl_group} 팀장 변경분 분석 중...")
            
            # 팀장이 변경한 직원만 조회
            changed_members = fetch_headquarter_cl_members_enhanced_filtered(headquarter_id, cl_group, period_id)
            
            if len(changed_members) == 0:
                print(f"   ✅ {cl_group}: 팀장 변경 없음 - 조정 불필요")
                department_data[cl_group] = {
                    "surplus": 0.0,
                    "needs_adjustment": False,
                    "member_count": 0,
                    "total_cl_members": 0,
                    "unchanged_members": 0,
                    "members_with_requests": [],
                    "validity_summary": {"매우 타당": 0, "타당": 0, "보통": 0, "의심": 0, "매우 의심": 0},
                    "target_total": 0,
                    "manager_score_sum": 0,
                    "target_stdev": get_cl_target_stdev(cl_group),
                    "members_data": []
                }
                continue
            
            # 변경된 직원들의 초과분 계산
            manager_scores = [m['manager_score'] for m in changed_members]
            module7_scores = [m['module7_score'] for m in changed_members]
            
            manager_score_sum = sum(manager_scores)
            module7_score_sum = sum(module7_scores)
            
            # 초과분 = 팀장 수정 후 총점 - 팀장 수정 전 총점
            surplus = round(manager_score_sum - module7_score_sum, 2)
            
            # 전체 CL 인원 정보
            total_cl_members = changed_members[0]['total_cl_members'] if changed_members else 0
            unchanged_members = changed_members[0]['unchanged_cl_members'] if changed_members else 0
            
            # 상승 요청한 사람들 식별 및 타당성 분석
            members_with_requests = []
            validity_summary = {"매우 타당": 0, "타당": 0, "보통": 0, "의심": 0, "매우 의심": 0}
            
            for m in changed_members:
                if m['score_diff'] > 0:  # 상승 요청한 경우만
                    validity_analysis = calculate_enhanced_captain_validity(m)
                    validity_grade = validity_analysis['validity_grade']
                    validity_summary[validity_grade] += 1
                    
                    members_with_requests.append({
                        "emp_no": m['emp_no'],
                        "score_diff": m['score_diff'],
                        "validity_grade": validity_grade,
                        "validity_score": validity_analysis['final_validity']
                    })
            
            # 조정 필요성 판단 (초과분이 0.05점 이상이면 조정 필요)
            needs_adjustment = abs(surplus) > 0.05
            
            # 통계 출력
            print(f"   변경된 인원: {len(changed_members)}명 (전체 {total_cl_members}명 중)")
            print(f"   변경 없는 인원: {unchanged_members}명")
            print(f"   점수 변화 총합: {surplus:+.2f}점")
            print(f"   상승 요청자: {len(members_with_requests)}명")
            print(f"   타당성 분포: 매우타당 {validity_summary['매우 타당']}명, 타당 {validity_summary['타당']}명, 의심 {validity_summary['의심']}명")
            print(f"   조정 필요: {'✅ Yes' if needs_adjustment else '❌ No'}")
            
            # department_data에 저장
            department_data[cl_group] = {
                "surplus": surplus,
                "needs_adjustment": needs_adjustment,
                "member_count": len(changed_members),
                "total_cl_members": total_cl_members,
                "unchanged_members": unchanged_members,
                "members_with_requests": members_with_requests,
                "validity_summary": validity_summary,
                "target_total": len(changed_members) * 3.5,  # 변경된 직원 기준
                "manager_score_sum": manager_score_sum,
                "target_stdev": get_cl_target_stdev(cl_group),
                "members_data": changed_members  # 변경된 직원만
            }
            
            total_members += total_cl_members
            total_changed_members += len(changed_members)
            
            if needs_adjustment:
                adjustment_needed_cls.append(cl_group)
        
        # 전체 요약
        print(f"\n📈 본부 {headquarter_id} 팀장 변경분 분석 요약:")
        print(f"   전체 인원: {total_members}명")
        print(f"   팀장 변경 인원: {total_changed_members}명")
        print(f"   변경 없는 인원: {total_members - total_changed_members}명")
        print(f"   전체 CL 그룹: {len(cl_groups)}개")
        print(f"   조정 필요 CL: {len(adjustment_needed_cls)}개 {adjustment_needed_cls}")
        
        # State 업데이트
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"1단계 완료: {total_changed_members}/{total_members}명 변경, {len(adjustment_needed_cls)}개 CL 조정 필요")],
            "department_data": department_data
        })
        
        return updated_state
        
    except Exception as e:
        print(f"❌ 1단계 실패: {str(e)}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"1단계 실패: {str(e)}")],
            "error_logs": state.get("error_logs", []) + [f"department_data_collection: {str(e)}"]
        })
        return updated_state

def enhanced_analysis_submodule(state: Module9AgentState) -> Module9AgentState:
    """2단계: 향상된 타당성 분석 - 변경된 직원만 분석"""
    
    try:
        department_data = state["department_data"]
        headquarter_id = state["headquarter_id"]
        period_id = state["period_id"]
        
        print(f"🧠 2단계: 변경된 직원 타당성 분석 시작")
        
        enhanced_analysis = {}
        total_analyzed = 0
        
        # 조정이 필요한 CL들만 처리
        for cl_group, cl_data in department_data.items():
            if not cl_data["needs_adjustment"] or cl_data["member_count"] == 0:
                print(f"⏭️ {cl_group}: 분석 불필요 (변경된 직원 {cl_data['member_count']}명)")
                enhanced_analysis[cl_group] = {
                    "analysis_completed": False,
                    "skip_reason": "변경된 직원 없음" if cl_data["member_count"] == 0 else "조정 불필요",
                    "members_analyzed": 0
                }
                continue
            
            print(f"\n🔍 {cl_group} 변경된 직원 타당성 분석 중...")
            
            # 안전한 통계 정보 가져오기
            total_cl_members = cl_data.get('total_cl_members', cl_data['member_count'])
            print(f"   대상: {cl_data['member_count']}명 (전체 {total_cl_members}명 중)")
            
            members = cl_data["members_data"]
            analyzed_members = []
            validity_distribution = {"매우 타당": [], "타당": [], "보통": [], "의심": [], "매우 의심": []}
            
            # 변경된 멤버별 상세 분석
            for member in members:
                score_diff = member.get('score_diff', 0)
                print(f"   📋 {member['emp_no']} 분석 중... (변경량: {score_diff:+.2f}점)")
                
                # 향상된 타당성 분석 실행
                validity_analysis = calculate_enhanced_captain_validity(member)
                
                # 종합 성과 점수 계산
                comprehensive_score = calculate_comprehensive_performance_score(member)
                
                # 분석 결과 저장
                analyzed_member = {
                    "emp_no": member["emp_no"],
                    "emp_name": member["emp_name"],
                    "original_manager_score": member["manager_score"],
                    "module7_score": member["module7_score"],
                    "score_diff": score_diff,
                    "captain_reason": member.get("captain_reason", ""),
                    "kpi_achievement": member.get("kpi_achievement", 100),
                    "validity_analysis": validity_analysis,
                    "comprehensive_performance": comprehensive_score,
                    "task_count": len(member.get("task_data", [])),
                    "peer_summary_available": bool(member.get("peer_evaluation_data", {}).get("peer_summary")),
                    "changed_by_manager": member.get("changed_by_manager", True)
                }
                
                analyzed_members.append(analyzed_member)
                validity_distribution[validity_analysis["validity_grade"]].append(member["emp_no"])
                total_analyzed += 1
                
                # 상세 출력
                print(f"     타당성: {validity_analysis['final_validity']:.3f} ({validity_analysis['validity_grade']})")
                print(f"     업무증거: {validity_analysis['task_evidence']:.3f}, 동료일치: {validity_analysis['peer_consistency']:.3f}")
                print(f"     종합성과: {comprehensive_score:.1f}점")
            
            # CL별 분석 요약
            if analyzed_members:
                avg_validity = sum(m["validity_analysis"]["final_validity"] for m in analyzed_members) / len(analyzed_members)
                high_validity_count = len([m for m in analyzed_members if m["validity_analysis"]["final_validity"] >= 0.7])
                low_validity_count = len([m for m in analyzed_members if m["validity_analysis"]["final_validity"] < 0.4])
                
                enhanced_analysis[cl_group] = {
                    "analysis_completed": True,
                    "members_analyzed": len(analyzed_members),
                    "analyzed_members": analyzed_members,
                    "validity_distribution": validity_distribution,
                    "avg_validity": round(avg_validity, 3),
                    "high_validity_count": high_validity_count,
                    "low_validity_count": low_validity_count,
                    "analysis_summary": {
                        "total_members": len(analyzed_members),
                        "avg_validity": round(avg_validity, 3),
                        "validity_range": f"{min(m['validity_analysis']['final_validity'] for m in analyzed_members):.3f} - {max(m['validity_analysis']['final_validity'] for m in analyzed_members):.3f}",
                        "high_validity_ratio": round(high_validity_count / len(analyzed_members) * 100, 1),
                        "recommendation": "고타당성" if avg_validity >= 0.7 else "보통타당성" if avg_validity >= 0.4 else "저타당성"
                    }
                }
                
                print(f"   ✅ {cl_group} 분석 완료: 평균 타당성 {avg_validity:.3f}, 고타당성 {high_validity_count}명, 저타당성 {low_validity_count}명")
            else:
                # 분석할 멤버가 없는 경우
                enhanced_analysis[cl_group] = {
                    "analysis_completed": False,
                    "skip_reason": "분석 대상 없음",
                    "members_analyzed": 0
                }
        
        # 전체 분석 요약
        print(f"\n📊 2단계 변경된 직원 분석 완료:")
        print(f"   총 분석 인원: {total_analyzed}명 (팀장이 점수 변경한 직원만)")
        
        completed_analyses = [analysis for analysis in enhanced_analysis.values() if analysis.get("analysis_completed")]
        if completed_analyses:
            overall_avg_validity = sum(analysis["avg_validity"] for analysis in completed_analyses) / len(completed_analyses)
            print(f"   전체 평균 타당성: {overall_avg_validity:.3f}")
        
        # State 업데이트
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"2단계 완료: {total_analyzed}명 변경된 직원 분석")],
            "enhanced_analysis": enhanced_analysis
        })
        
        return updated_state
        
    except Exception as e:
        print(f"❌ 2단계 실패: {str(e)}")
        print(f"   디버그 정보: {type(e).__name__}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"2단계 실패: {str(e)}")],
            "error_logs": state.get("error_logs", []) + [f"enhanced_analysis: {str(e)}"],
            "enhanced_analysis": {}  # 빈 분석 결과로 초기화
        })
        return updated_state

# ================================================================
# 3단계 서브모듈: cl_supervisor_execution
# ================================================================

def build_enhanced_supervisor_input_data(cl_group: str, cl_data: Dict, enhanced_analysis: Dict, headquarter_id: int) -> Dict:
    """향상된 AI Supervisor 입력 데이터 구성"""
    
    surplus = cl_data["surplus"]
    target_total = cl_data["target_total"]
    manager_score_sum = cl_data["manager_score_sum"]
    target_stdev = cl_data["target_stdev"]
    
    # 향상된 분석 결과 가져오기
    analysis_data = enhanced_analysis.get(cl_group, {})
    analyzed_members = analysis_data.get("analyzed_members", [])
    
    # 본부명 조회 (실제로는 DB에서, 지금은 더미)
    headquarter_name = f"본부{headquarter_id}"
    
    supervisor_input = {
        "cl_group": cl_group,
        "headquarter_name": headquarter_name,
        "total_surplus": surplus,
        "current_situation": {
            "target_total": target_total,
            "manager_score_sum": manager_score_sum,
            "required_reduction": surplus
        },
        "distribution_targets": {
            "target_mean": 3.5,
            "target_stdev": target_stdev,
            "member_count": len(analyzed_members)
        },
        "enhanced_analysis_summary": analysis_data.get("analysis_summary", {}),
        "members": []
    }
    
    # 멤버별 향상된 데이터 구성
    for analysis in analyzed_members:
        member_data = {
            "emp_no": analysis["emp_no"],
            "emp_name": analysis["emp_name"],
            "current_score": analysis["original_manager_score"],
            "baseline_score": analysis["module7_score"],
            "score_change": analysis["score_diff"],
            "captain_reason": analysis["captain_reason"],
            "kpi_achievement": analysis["kpi_achievement"],
            "validity_analysis": analysis["validity_analysis"],
            "comprehensive_performance": analysis["comprehensive_performance"],
            "task_count": analysis["task_count"],
            "peer_summary_available": analysis["peer_summary_available"],
            "final_evaluation_report_id": None  # 원본 데이터에서 가져와야 함
        }
        
        # original member data에서 final_evaluation_report_id 찾기
        for original_member in cl_data["members_data"]:
            if original_member["emp_no"] == analysis["emp_no"]:
                member_data["final_evaluation_report_id"] = original_member.get("final_evaluation_report_id")
                break
        
        supervisor_input["members"].append(member_data)
    
    return supervisor_input

def apply_standard_deviation_algorithm(llm_result: Dict, target_stdev: float, cl_group: str) -> Dict:
    """표준편차만 수학적으로 조정하는 알고리즘 - 구조 안전성 강화"""
    
    # 1. 안전한 adjustments 추출
    try:
        if "adjustments" in llm_result:
            # 직접 구조 (Fallback 결과)
            adjustments = llm_result["adjustments"]
            validation_check = llm_result.get("validation_check", {})
            is_nested_structure = False
        elif "result" in llm_result and "adjustments" in llm_result["result"]:
            # 중첩 구조 (LLM 결과)
            adjustments = llm_result["result"]["adjustments"]
            validation_check = llm_result["result"].get("validation_check", {})
            is_nested_structure = True
        else:
            raise KeyError(f"adjustments를 찾을 수 없습니다. 사용 가능한 키: {list(llm_result.keys())}")
    except Exception as e:
        print(f"❌ {cl_group}: adjustments 추출 실패 - {str(e)}")
        return llm_result
    
    member_count = len(adjustments)
    if member_count == 0:
        print(f"⚠️ {cl_group}: 조정할 멤버가 없습니다")
        return llm_result
    
    print(f"📊 {cl_group} 표준편차 수학적 조정:")
    print(f"   대상: {member_count}명")
    
    # 2. 현재 상태 분석
    try:
        current_scores = [(adj["final_score"], adj["emp_no"]) for adj in adjustments]
        current_scores.sort(key=lambda x: x[0], reverse=True)  # 점수 순 정렬 (높은 순)
        
        score_values = [score for score, _ in current_scores]
        current_stdev = statistics.stdev(score_values) if member_count > 1 else 0
        current_mean = sum(score_values) / member_count
        
        print(f"   현재: 평균 {current_mean:.3f}, 표준편차 {current_stdev:.2f}")
        print(f"   목표: 평균 3.500, 표준편차 {target_stdev:.1f}")
        
    except Exception as e:
        print(f"❌ {cl_group}: 현재 상태 분석 실패 - {str(e)}")
        return llm_result
    
    # 3. 표준편차 달성 여부 확인
    stdev_diff = abs(current_stdev - target_stdev)
    if stdev_diff <= 0.2:
        print(f"   ✅ 표준편차 이미 달성 (차이: {stdev_diff:.2f})")
        return llm_result
    
    # 4. 목표 표준편차를 위한 새로운 점수 계산
    target_mean = 3.5
    
    try:
        if member_count == 1:
            # 1명인 경우: 평균값만 설정
            new_scores = [target_mean]
            
        elif member_count == 2:
            # 2명인 경우: 평균 기준 대칭 분포
            spread = target_stdev * 1.0  # 2명일 때 적절한 분산
            new_scores = [
                target_mean + spread,
                target_mean - spread
            ]
            
        else:
            # 3명 이상인 경우: 등차수열 기반 분포
            # 표준편차 공식: σ = sqrt(Σ(x-μ)²/n)
            # 등차수열에서 표준편차를 역산하여 범위 결정
            
            # 안전한 범위 설정 (0.0 ~ 5.0 내에서)
            max_spread = min(2.0, target_stdev * 2.0)  # 최대 분산 제한
            max_score = min(5.0, target_mean + max_spread)
            min_score = max(0.0, target_mean - max_spread)
            
            # 등차수열 생성
            if member_count > 1:
                step = (max_score - min_score) / (member_count - 1)
                new_scores = [max_score - i * step for i in range(member_count)]
            else:
                new_scores = [target_mean]
            
            # 평균 보정
            current_new_mean = sum(new_scores) / member_count
            mean_adjustment = target_mean - current_new_mean
            new_scores = [score + mean_adjustment for score in new_scores]
            
            # 범위 제한 재적용
            new_scores = [max(0.0, min(5.0, score)) for score in new_scores]
            
            # 평균 재보정 (범위 제한 후)
            actual_mean = sum(new_scores) / member_count
            if abs(actual_mean - target_mean) > 0.01:
                final_adjustment = target_mean - actual_mean
                new_scores = [max(0.0, min(5.0, score + final_adjustment)) for score in new_scores]
        
        print(f"   계산된 새 점수: {[round(s, 2) for s in new_scores]}")
        
    except Exception as e:
        print(f"❌ {cl_group}: 새 점수 계산 실패 - {str(e)}")
        return llm_result
    
    # 5. 새로운 점수 적용 (성과 순위 유지)
    try:
        for i, (old_score, emp_no) in enumerate(current_scores):
            for adj in adjustments:
                if adj["emp_no"] == emp_no:
                    adj["final_score"] = round(new_scores[i], 2)
                    adj["change_amount"] = round(adj["final_score"] - adj["original_score"], 2)
                    
                    # change_type 업데이트
                    if adj["change_amount"] < -0.01:
                        adj["change_type"] = "decrease"
                    elif adj["change_amount"] > 0.01:
                        adj["change_type"] = "increase"
                    else:
                        adj["change_type"] = "maintain"
                    
                    # 조정 사유 업데이트
                    rank = i + 1
                    adj["reason"] = f"표준편차 조정: {rank}위/{member_count}명 (목표 σ={target_stdev:.1f})"
                    break
                    
    except Exception as e:
        print(f"❌ {cl_group}: 점수 적용 실패 - {str(e)}")
        return llm_result
    
    # 6. 최종 검증
    try:
        final_scores = [adj["final_score"] for adj in adjustments]
        final_mean = sum(final_scores) / member_count
        final_stdev = statistics.stdev(final_scores) if member_count > 1 else 0
        
        # 목표 달성 여부
        mean_achieved = abs(final_mean - 3.5) <= 0.02
        stdev_achieved = abs(final_stdev - target_stdev) <= 0.3
        
        print(f"   결과: 평균 {final_mean:.3f}, 표준편차 {final_stdev:.2f}")
        print(f"   평균 달성: {'✅' if mean_achieved else '⚠️'}")
        print(f"   표준편차 달성: {'✅' if stdev_achieved else '⚠️'}")
        
        # validation_check 업데이트
        validation_updates = {
            "actual_mean": round(final_mean, 3),
            "actual_stdev": round(final_stdev, 2),
            "target_stdev": target_stdev,
            "mean_achieved": mean_achieved,
            "stdev_achieved": stdev_achieved,
            "stdev_adjustment_applied": True
        }
        
        validation_check.update(validation_updates)
        
    except Exception as e:
        print(f"❌ {cl_group}: 최종 검증 실패 - {str(e)}")
        return llm_result
    
    # 7. 결과 구조 복원
    try:
        if is_nested_structure:
            # LLM 결과 구조: result 키 안에 저장
            result_data = {
                "analysis_summary": llm_result.get("result", {}).get("analysis_summary", f"{cl_group} 표준편차 조정 완료"),
                "adjustments": adjustments,
                "validation_check": validation_check
            }
            return result_data
        else:
            # Fallback 결과 구조: 직접 저장
            return {
                "analysis_summary": llm_result.get("analysis_summary", f"{cl_group} 표준편차 조정 완료"),
                "adjustments": adjustments,
                "validation_check": validation_check
            }
            
    except Exception as e:
        print(f"❌ {cl_group}: 결과 구조 복원 실패 - {str(e)}")
        return llm_result

def cl_supervisor_execution_submodule(state: Module9AgentState) -> Module9AgentState:
    """3단계: CL별 향상된 Supervisor 실행 서브모듈"""
    
    try:
        department_data = state["department_data"]
        enhanced_analysis = state["enhanced_analysis"]
        headquarter_id = state["headquarter_id"]
        period_id = state["period_id"]
        
        print(f"🎯 3단계: LLM 제로섬 + 수학 표준편차 분리 실행 시작")
        
        supervisor_results = {}
        total_adjustments = 0
        
        # 조정이 필요한 CL들만 처리
        for cl_group, cl_data in department_data.items():
            if not cl_data["needs_adjustment"]:
                print(f"⏭️ {cl_group}: 조정 불필요 (surplus: {cl_data['surplus']:.2f})")
                supervisor_results[cl_group] = {
                    "success": True,
                    "adjustments_made": 0,
                    "distribution_achieved": True,
                    "processing_time_ms": 0,
                    "fallback_used": False,
                    "skip_reason": "조정 불필요"
                }
                continue
            
            print(f"\n🎯 {cl_group} 2단계 처리 중... (surplus: {cl_data['surplus']:+.2f}점)")
            
            # 1. Supervisor 입력 데이터 구성
            supervisor_input = build_enhanced_supervisor_input_data(cl_group, cl_data, enhanced_analysis, headquarter_id)
            
            # 2. LLM 제로섬 조정 실행 (표준편차 제외)
            import time
            start_time = time.time()
            
            print(f"🧠 {cl_group}: LLM 성과 기반 제로섬 조정")
            llm_result = call_enhanced_supervisor_llm(supervisor_input)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # 3. LLM 결과 처리
            if llm_result["success"]:
                print(f"✅ {cl_group}: LLM 제로섬 조정 성공")
                
                # 4. 표준편차 수학적 조정 실행
                target_stdev = cl_data.get("target_stdev", get_cl_target_stdev(cl_group))
                print(f"📊 {cl_group}: 표준편차 수학적 조정 ({target_stdev:.1f}점 목표)")
                
                # 구조에 맞게 전달
                final_result = apply_standard_deviation_algorithm(llm_result["result"], target_stdev, cl_group)
                supervisor_output = final_result
                fallback_used = False
                
            else:
                print(f"🔧 {cl_group}: LLM 실패, Fallback 알고리즘 실행")
                
                # Fallback 실행 전에 supervisor_input 확인
                if not supervisor_input.get("members"):
                    print(f"❌ {cl_group}: supervisor_input에 members 데이터 없음")
                    supervisor_results[cl_group] = {
                        "success": False,
                        "error": "supervisor_input에 members 데이터 없음",
                        "adjustments_made": 0,
                        "fallback_used": True
                    }
                    continue
                
                try:
                    supervisor_output = execute_proper_zero_sum_adjustment(supervisor_input)
                    if not supervisor_output.get("adjustments"):
                        print(f"❌ {cl_group}: Fallback에서 adjustments 생성 실패")
                        supervisor_results[cl_group] = {
                            "success": False,
                            "error": "Fallback에서 adjustments 생성 실패",
                            "adjustments_made": 0,
                            "fallback_used": True
                        }
                        continue
                except Exception as fallback_error:
                    print(f"❌ {cl_group}: Fallback 실행 실패 - {str(fallback_error)}")
                    supervisor_results[cl_group] = {
                        "success": False,
                        "error": f"Fallback 실행 실패: {str(fallback_error)}",
                        "adjustments_made": 0,
                        "fallback_used": True
                    }
                    continue
                
                # Fallback 결과에도 표준편차 조정 적용
                target_stdev = cl_data.get("target_stdev", get_cl_target_stdev(cl_group))
                fake_llm_result = {"result": supervisor_output, "success": True}
                try:
                    final_result = apply_standard_deviation_algorithm(fake_llm_result, target_stdev, cl_group)
                    supervisor_output = final_result["result"]
                except Exception as stdev_error:
                    print(f"⚠️ {cl_group}: 표준편차 조정 실패 - {str(stdev_error)}")
                    # 표준편차 조정 실패해도 원본 결과 사용
                    pass
                    
                fallback_used = True
            
            # 5. 최종 검증
            adjustments = supervisor_output["adjustments"]
            target_reduction = cl_data["surplus"]
            target_stdev = cl_data.get("target_stdev", get_cl_target_stdev(cl_group))
            
            validation_result = validate_zero_sum_result(adjustments, target_reduction, target_stdev, cl_group)
            
            # 검증 결과 출력
            if validation_result["valid"]:
                print(f"✅ {cl_group} 2단계 처리 완료")
                print(f"   📊 결과: 평균 {validation_result['metrics']['actual_mean']:.3f}, 표준편차 {validation_result['metrics']['actual_stdev']:.2f}")
                print(f"   💰 차감: {validation_result['metrics']['actual_reduction']:.3f}/{validation_result['metrics']['target_reduction']:.3f}")
            else:
                print(f"⚠️ {cl_group} 검증 경고:")
                for warning in validation_result["warnings"][:2]:
                    print(f"     - {warning}")
            
            # 성과 역전 체크
            if validation_result["performance_reversal"]["has_reversal"]:
                print(f"   ⚠️ 성과 역전: {validation_result['performance_reversal']['reversal_count']}건")
            
            # 6. DB 업데이트
            update_result = batch_update_final_evaluation_reports(adjustments, period_id)
            
            # 7. 결과 저장
            supervisor_results[cl_group] = {
                "success": True,
                "adjustments_made": update_result["success_count"],
                "distribution_achieved": validation_result["valid"],
                "processing_time_ms": processing_time,
                "fallback_used": fallback_used,
                "update_success_count": update_result["success_count"],
                "update_failed_count": update_result["failed_count"],
                "validation_result": validation_result,
                "supervisor_output": supervisor_output,
                "enhanced_features": {
                    "llm_zero_sum_used": not fallback_used,
                    "math_stdev_applied": True,
                    "two_stage_processing": True
                }
            }
            
            total_adjustments += update_result["success_count"]
            
            print(f"✅ {cl_group} 2단계 처리 완료: {update_result['success_count']}명 조정")
        
        # State 업데이트
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"3단계 완료: {total_adjustments}명 2단계 조정 (LLM 제로섬 + 수학 표준편차)")],
            "supervisor_results": supervisor_results
        })
        
        return updated_state
        
    except Exception as e:
        print(f"❌ 3단계 실패: {str(e)}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"3단계 실패: {str(e)}")],
            "error_logs": state.get("error_logs", []) + [f"cl_supervisor_execution: {str(e)}"]
        })
        return updated_state

# ================================================================
# 4단계 서브모듈: batch_update
# ================================================================

def batch_update_submodule(state: Module9AgentState) -> Module9AgentState:
    """4단계: 배치 업데이트 서브모듈 - 향상된 분석 결과 통합 및 최종 집계"""
    
    try:
        supervisor_results = state["supervisor_results"]
        enhanced_analysis = state["enhanced_analysis"]
        
        print(f"📊 4단계: 향상된 배치 업데이트 및 결과 집계 시작")
        
        # 전체 결과 집계
        total_cls_processed = len(supervisor_results)
        successful_cls = 0
        failed_cls = 0
        total_adjustments = 0
        total_distribution_achieved = 0
        total_processing_time = 0
        fallback_used_count = 0
        enhanced_features_used = 0
        
        successful_employees = []
        failed_employees = []
        cl_summaries = []
        
        # CL별 결과 분석
        for cl_group, result in supervisor_results.items():
            cl_summary = {
                "cl_group": cl_group,
                "success": result["success"],
                "adjustments_made": result["adjustments_made"],
                "distribution_achieved": result.get("distribution_achieved", False),
                "processing_time_ms": result["processing_time_ms"],
                "fallback_used": result["fallback_used"],
                "enhanced_features": result.get("enhanced_features", {})
            }
            
            # 향상된 분석 정보 추가
            if cl_group in enhanced_analysis:
                analysis_info = enhanced_analysis[cl_group]
                if analysis_info.get("analysis_completed"):
                    cl_summary["enhanced_analysis"] = {
                        "avg_validity": analysis_info["avg_validity"],
                        "high_validity_count": analysis_info["high_validity_count"],
                        "low_validity_count": analysis_info["low_validity_count"],
                        "recommendation": analysis_info["analysis_summary"]["recommendation"]
                    }
            
            # 집계
            if result["success"]:
                successful_cls += 1
                total_adjustments += result["adjustments_made"]
                
                if result.get("distribution_achieved"):
                    total_distribution_achieved += 1
                
                if result["fallback_used"]:
                    fallback_used_count += 1
                
                if result.get("enhanced_features", {}).get("validity_analysis_used"):
                    enhanced_features_used += 1
                
                total_processing_time += result["processing_time_ms"]
                
                # 성공한 직원들 수집
                if result.get("supervisor_output") and result["supervisor_output"].get("adjustments"):
                    for adj in result["supervisor_output"]["adjustments"]:
                        employee_info = {
                            "emp_no": adj["emp_no"],
                            "cl_group": cl_group,
                            "original_score": adj["original_score"],
                            "final_score": adj["final_score"],
                            "change_amount": adj["change_amount"],
                            "reason": adj["reason"]
                        }
                        
                        # 향상된 분석 정보 추가
                        if "validity_analysis" in adj:
                            employee_info["validity_grade"] = adj["validity_analysis"]["validity_grade"]
                            employee_info["final_validity"] = adj["validity_analysis"]["final_validity"]
                        
                        successful_employees.append(employee_info)
                
                cl_summary["status"] = "완료"
                if result.get("skip_reason"):
                    cl_summary["note"] = result["skip_reason"]
                elif result["fallback_used"]:
                    cl_summary["note"] = "향상된 Fallback 알고리즘 사용"
                else:
                    cl_summary["note"] = "향상된 AI Supervisor 성공"
                    
                # 검증 결과 추가
                if "validation_result" in result:
                    validation = result["validation_result"]
                    cl_summary["validation_summary"] = {
                        "valid": validation["valid"],
                        "zero_sum_achieved": validation["metrics"]["reduction_error"] < 0.02,
                        "stdev_achieved": validation["metrics"]["stdev_error"] < 0.3,
                        "performance_reversal": validation["performance_reversal"]["has_reversal"]
                    }
            else:
                failed_cls += 1
                cl_summary["status"] = "실패"
                cl_summary["error"] = result.get("error", "알 수 없는 오류")
                
                # 실패한 직원들 추정
                department_data = state.get("department_data", {})
                if cl_group in department_data:
                    for member in department_data[cl_group].get("members_data", []):
                        failed_employees.append({
                            "emp_no": member["emp_no"],
                            "cl_group": cl_group,
                            "error": result.get("error", "처리 실패")
                        })
            
            cl_summaries.append(cl_summary)
            
            # 진행 상황 출력
            status_icon = "✅" if result["success"] else "❌"
            enhanced_note = " (향상됨)" if result.get("enhanced_features", {}).get("validity_analysis_used") else ""
            validation_note = ""
            
            if result["success"] and "validation_result" in result:
                validation = result["validation_result"]
                if validation["valid"]:
                    validation_note = " ✓검증통과"
                else:
                    validation_note = f" ⚠️검증실패({len(validation['errors'])}건)"
            
            print(f"   {status_icon} {cl_group}: {cl_summary['status']} ({result['adjustments_made']}명){enhanced_note}{validation_note}")
        
        # 전체 성공률 계산
        success_rate = (successful_cls / total_cls_processed * 100) if total_cls_processed > 0 else 0
        distribution_rate = (total_distribution_achieved / successful_cls * 100) if successful_cls > 0 else 0
        enhanced_rate = (enhanced_features_used / successful_cls * 100) if successful_cls > 0 else 0
        avg_processing_time = (total_processing_time / successful_cls) if successful_cls > 0 else 0
        
        # 최종 결과 구성
        update_results = {
            "total_cls_processed": total_cls_processed,
            "successful_cls": successful_cls,
            "failed_cls": failed_cls,
            "success_rate": round(success_rate, 1),
            "total_adjustments": total_adjustments,
            "distribution_achieved_count": total_distribution_achieved,
            "distribution_rate": round(distribution_rate, 1),
            "fallback_used_count": fallback_used_count,
            "enhanced_features_used_count": enhanced_features_used,
            "enhanced_rate": round(enhanced_rate, 1),
            "total_processing_time_ms": total_processing_time,
            "avg_processing_time_ms": round(avg_processing_time, 0),
            "successful_employees": successful_employees,
            "failed_employees": failed_employees,
            "cl_summaries": cl_summaries
        }
        
        # 상세 결과 출력
        print(f"\n📈 향상된 최종 집계 결과:")
        print(f"   처리된 CL 그룹: {total_cls_processed}개")
        print(f"   성공한 CL: {successful_cls}개 ({success_rate:.1f}%)")
        print(f"   실패한 CL: {failed_cls}개")
        print(f"   총 조정 인원: {total_adjustments}명")
        print(f"   분포 달성: {total_distribution_achieved}/{successful_cls}개 CL ({distribution_rate:.1f}%)")
        print(f"   Fallback 사용: {fallback_used_count}개 CL")
        print(f"   향상된 기능 사용: {enhanced_features_used}/{successful_cls}개 CL ({enhanced_rate:.1f}%)")
        print(f"   평균 처리시간: {avg_processing_time:.0f}ms")
        
        if failed_employees:
            print(f"   ❌ 실패 직원: {len(failed_employees)}명")
            for failed in failed_employees[:3]:  # 처음 3명만
                print(f"     - {failed['emp_no']} ({failed['cl_group']}): {failed['error']}")
        
        # State 업데이트
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"4단계 완료: {total_adjustments}명 향상된 조정, {successful_cls}/{total_cls_processed}개 CL 성공")],
            "update_results": update_results,
            "total_processed": len(successful_employees),
            "total_failed": len(failed_employees)
        })
        
        return updated_state
        
    except Exception as e:
        print(f"❌ 4단계 실패: {str(e)}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"4단계 실패: {str(e)}")],
            "error_logs": state.get("error_logs", []) + [f"batch_update: {str(e)}"],
            "update_results": {"error": str(e)},
            "total_processed": 0,
            "total_failed": 0
        })
        return updated_state


# ================================================================
# LangGraph 워크플로우 구성
# ================================================================

def create_enhanced_module9_graph():
    """향상된 모듈9 LangGraph 워크플로우 생성"""
    
    # StateGraph 생성
    enhanced_module9_workflow = StateGraph(Module9AgentState)
    
    # 노드 추가 (4개 서브모듈)
    enhanced_module9_workflow.add_node("department_data_collection", department_data_collection)
    enhanced_module9_workflow.add_node("validity_analysis", enhanced_analysis_submodule)
    enhanced_module9_workflow.add_node("cl_supervisor_execution", cl_supervisor_execution_submodule)
    enhanced_module9_workflow.add_node("batch_update", batch_update_submodule)
    
    # 엣지 정의 (순차 실행)
    enhanced_module9_workflow.add_edge(START, "department_data_collection")
    enhanced_module9_workflow.add_edge("department_data_collection", "validity_analysis")
    enhanced_module9_workflow.add_edge("validity_analysis", "cl_supervisor_execution")
    enhanced_module9_workflow.add_edge("cl_supervisor_execution", "batch_update")
    enhanced_module9_workflow.add_edge("batch_update", END)
    
    return enhanced_module9_workflow.compile()