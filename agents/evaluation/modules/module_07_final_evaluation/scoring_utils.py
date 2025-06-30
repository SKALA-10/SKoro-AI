# ================================================================
# scoring_utils_module7.py - 모듈 7 점수 계산 관련 유틸리티
# ================================================================

import statistics
from typing import Dict, List

# ================================================================
# 가중치 계산 함수
# ================================================================

def get_evaluation_weights_by_cl(cl) -> Dict[str, float]:
    """CL별 디폴트 가중치 반환"""
    # CL 값을 문자열로 변환 (숫자든 문자열이든 처리)
    if isinstance(cl, (int, float)):
        cl_key = f"CL{int(cl)}"
    else:
        cl_key = str(cl).upper()
        if not cl_key.startswith("CL"):
            cl_key = f"CL{cl_key}"
    
    cl_weights = {
        "CL3": {"achievement": 0.6, "fourp": 0.4},
        "CL2": {"achievement": 0.5, "fourp": 0.5}, 
        "CL1": {"achievement": 0.4, "fourp": 0.6}
    }
    return cl_weights.get(cl_key, {"achievement": 0.5, "fourp": 0.5})  # 기본값

# ================================================================
# SK 등급 기반 절대평가 달성률 점수 계산 함수
# ================================================================

def calculate_achievement_score_by_grade(achievement_rate: float) -> tuple[float, str]:
    """SK 등급 체계 기반 절대평가 점수 계산"""
    
    if achievement_rate >= 120:
        # S등급 상한 (120% 이상) → 5.0점
        score = 5.0
        grade = "S+"
        reason = f"달성률 {achievement_rate:.1f}% (S+등급, 탁월한 성과)"
        
    elif achievement_rate >= 110:
        # S등급 (110-120%) → 4.0-5.0점 선형 배치
        progress = (achievement_rate - 110) / 10  # 0~1 사이 값
        score = 4.0 + (progress * 1.0)  # 4.0 ~ 5.0
        grade = "S"
        reason = f"달성률 {achievement_rate:.1f}% (S등급, 매우 우수한 성과)"
        
    elif achievement_rate >= 100:
        # A등급 (100-110%) → 3.5-4.0점 선형 배치  
        progress = (achievement_rate - 100) / 10  # 0~1 사이 값
        score = 3.5 + (progress * 0.5)  # 3.5 ~ 4.0
        grade = "A"
        reason = f"달성률 {achievement_rate:.1f}% (A등급, 목표 달성)"
        
    elif achievement_rate >= 80:
        # B등급 (80-100%) → 2.5-3.5점 선형 배치
        progress = (achievement_rate - 80) / 20  # 0~1 사이 값  
        score = 2.5 + (progress * 1.0)  # 2.5 ~ 3.5
        grade = "B"
        reason = f"달성률 {achievement_rate:.1f}% (B등급, 목표 근접)"
        
    elif achievement_rate >= 60:
        # C등급 (60-80%) → 1.5-2.5점 선형 배치
        progress = (achievement_rate - 60) / 20  # 0~1 사이 값
        score = 1.5 + (progress * 1.0)  # 1.5 ~ 2.5
        grade = "C"
        reason = f"달성률 {achievement_rate:.1f}% (C등급, 목표 미달)"
        
    else:
        # D등급 (60% 미만) → 1.0-1.5점 선형 배치
        if achievement_rate <= 0:
            score = 1.0
        else:
            progress = achievement_rate / 60  # 0~1 사이 값
            score = 1.0 + (progress * 0.5)  # 1.0 ~ 1.5
        grade = "D"
        reason = f"달성률 {achievement_rate:.1f}% (D등급, 크게 미달)"
    
    return round(score, 2), reason

# ================================================================
# CL별 정규화 관련 함수들
# ================================================================

def get_cl_normalization_params(cl) -> Dict[str, float]:
    """CL별 정규화 파라미터 반환 (SK 표준)"""
    # CL 값을 문자열로 변환 (숫자든 문자열이든 처리)
    if isinstance(cl, (int, float)):
        cl_key = f"CL{int(cl)}"
    else:
        cl_key = str(cl).upper()
        if not cl_key.startswith("CL"):
            cl_key = f"CL{cl_key}"
    
    params = {
        "CL3": {"target_mean": 3.5, "target_stdev": 1.7},
        "CL2": {"target_mean": 3.5, "target_stdev": 1.5}, 
        "CL1": {"target_mean": 3.5, "target_stdev": 1.4}
    }
    return params.get(cl_key, {"target_mean": 3.5, "target_stdev": 1.5})

def normalize_cl_group(members: List[Dict], cl: str) -> List[Dict]:
    """CL 그룹 내 정규화 실행 (4명 이상일 때만 적용)"""
    
    if len(members) == 0:
        return members
    
    print(f"   {cl} 그룹 ({len(members)}명) 정규화 처리:")
    
    # 3명 이하인 경우 원시점수 유지
    if len(members) <= 3:
        for member in members:
            member["normalized_score"] = member["hybrid_score"]  # 원시점수 그대로
            member["normalization_reason"] = f"팀 내 {cl} {len(members)}명 (원시점수 유지)"
            member["raw_hybrid_score"] = member["hybrid_score"]  # 원시점수 보관
            print(f"     {member['emp_no']}: {member['hybrid_score']:.2f}점 (원시점수 유지)")
        return members
    
    # 4명 이상인 경우 정규화 적용
    # CL별 목표 파라미터
    params = get_cl_normalization_params(cl)
    target_mean = params["target_mean"]
    target_stdev = params["target_stdev"]
    
    # 원시점수 수집 (하이브리드 점수)
    raw_scores = [m["hybrid_score"] for m in members]
    
    # 현재 통계
    current_mean = statistics.mean(raw_scores)
    current_stdev = statistics.stdev(raw_scores)
    
    print(f"     정규화 적용: 평균 {current_mean:.2f} → {target_mean}, 표준편차 {current_stdev:.2f} → {target_stdev}")
    
    # 정규화 적용
    for member in members:
        raw_score = member["hybrid_score"]
        
        if current_stdev == 0:
            # 모든 점수가 동일한 경우
            normalized_score = target_mean
            reason = f"{cl} 동일점수 → 평균 {target_mean}점"
        else:
            # Z-score 계산 후 목표 분포로 변환
            z_score = (raw_score - current_mean) / current_stdev
            normalized_score = target_mean + (z_score * target_stdev)
            
            # 0.0-5.0 범위 제한 (SK 기준)
            normalized_score = max(0.0, min(5.0, normalized_score))
            
            reason = f"{cl} 정규화 (Z-Score: {z_score:.2f})"
        
        member["normalized_score"] = round(normalized_score, 2)
        member["normalization_reason"] = reason
        member["raw_hybrid_score"] = raw_score  # 원시점수 보관
        
        print(f"     {member['emp_no']}: {raw_score:.2f} → {normalized_score:.2f} ({reason})")
    
    return members

# ================================================================
# 점수 체계 미리보기 함수
# ================================================================

def preview_achievement_scoring():
    """SK 등급 기반 절대평가 점수 미리보기"""
    print("📋 SK 등급 기반 절대평가 점수 체계:")
    print("=" * 60)
    
    test_rates = [0, 30, 50, 70, 85, 95, 100, 105, 115, 125, 150, 200]
    
    for rate in test_rates:
        score, reason = calculate_achievement_score_by_grade(rate)
        print(f"달성률 {rate:3d}% → {score:4.2f}점 ({reason.split('(')[1].split(',')[0]})")
    
    print("=" * 60)