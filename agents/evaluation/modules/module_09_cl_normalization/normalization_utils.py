# ================================================================
# normalization_utils_module9.py - 모듈 9 정규화 관련 유틸리티
# ================================================================

import statistics
from typing import Dict, List

# ================================================================
# CL별 정규화 관련 함수들 (모듈 7에서 재사용)
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
    """CL 그룹 내 정규화 실행 (무조건 정규화 적용)"""
    
    if len(members) == 0:
        return members
    
    print(f"   {cl} 그룹 ({len(members)}명) 정규화 처리:")
    
    # CL별 목표 파라미터
    params = get_cl_normalization_params(cl)
    target_mean = params["target_mean"]
    target_stdev = params["target_stdev"]
    
    # 원시점수 수집 (manager_score) - Decimal을 float로 변환
    raw_scores = [float(m["manager_score"]) for m in members]
    
    # 현재 통계
    current_mean = statistics.mean(raw_scores)
    current_stdev = statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0
    
    print(f"     정규화 적용: 평균 {current_mean:.2f} → {target_mean}, 표준편차 {current_stdev:.2f} → {target_stdev}")
    
    # 정규화 적용
    for member in members:
        raw_score = float(member["manager_score"])  # Decimal을 float로 변환
        
        if current_stdev == 0 or len(members) == 1:
            # 모든 점수가 동일하거나 1명인 경우
            final_score = target_mean
            reason = f"본부 내 {cl} 정규화 → 평균 {target_mean}점"
        else:
            # Z-score 계산 후 목표 분포로 변환
            z_score = (raw_score - current_mean) / current_stdev
            final_score = target_mean + (z_score * target_stdev)
            
            # 0.0-5.0 범위 제한 (SK 기준)
            final_score = max(0.0, min(5.0, final_score))
            
            reason = f"본부 내 {cl} 정규화 (Z-Score: {z_score:.2f})"
        
        member["final_score"] = round(final_score, 2)
        member["cl_reason"] = reason
        
        print(f"     {member['emp_no']}: {raw_score:.2f} → {final_score:.2f} ({reason})")
    
    return members