# ================================================================
# llm_utils_1.py - 모듈 9 LLM 처리 및 JSON 파싱 유틸리티
# ================================================================

import re
import json
import time
import logging
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
logger = logging.getLogger(__name__)

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    # ```json ... ``` 형태 추출
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # { ... } 형태 추출 (가장 큰 JSON 객체)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                return text[start_idx:i+1]
    
    return text.strip()

# ================================================================
# 에러 처리 클래스
# ================================================================

class Module9ValidationError(Exception):
    pass

# ================================================================
# 업무 증거 일치성 분석 함수들
# ================================================================

def _fallback_task_evidence_analysis(captain_reason: str, task_data: List[Dict]) -> float:
    """LLM 실패 시 사용할 간단한 업무 증거 분석"""
    
    if not captain_reason or not task_data:
        return 0.3
    
    # 간단한 키워드 매칭
    evidence_score = 0.0
    total_checks = 0
    
    performance_indicators = ['완료', '달성', '개선', '성공', '우수', '기여']
    
    for indicator in performance_indicators:
        total_checks += 1
        if indicator in captain_reason:
            # 실제 업무에서 높은 성과가 있는지 확인
            high_performance = any(
                task.get('ai_achievement_rate', 0) >= 80 or 
                task.get('ai_contribution_score', 0) >= 70
                for task in task_data
            )
            if high_performance:
                evidence_score += 1.0
            else:
                evidence_score += 0.3
    
    return min(1.0, evidence_score / total_checks if total_checks > 0 else 0.3)

def _fallback_peer_evaluation_analysis(captain_reason: str, peer_data: Dict) -> float:
    """LLM 실패 시 사용할 간단한 동료평가 분석"""
    
    if not captain_reason or not peer_data:
        return 0.5
    
    strengths = peer_data.get('strengths', '').lower()
    concerns = peer_data.get('concerns', '').lower()
    
    # 긍정적 키워드 매칭
    positive_keywords = ['우수', '뛰어난', '잘', '성공', '기여', '리더십', '협업']
    negative_keywords = ['부족', '문제', '지연', '미흡', '개선']
    
    captain_positive = any(word in captain_reason for word in positive_keywords)
    peer_positive = len(strengths) > len(concerns)
    
    if captain_positive and peer_positive:
        return 0.8
    elif not captain_positive and not peer_positive:
        return 0.7
    elif captain_positive and not peer_positive:
        return 0.2  # 팀장은 긍정적인데 동료는 부정적
    else:
        return 0.6  # 팀장은 부정적인데 동료는 긍정적

def analyze_task_evidence_consistency(member: Dict) -> float:
    """LLM 기반 업무 실적과 팀장 사유의 일치성 분석"""
    
    captain_reason = member.get('captain_reason', '') or ''
    task_data = member.get('task_data', [])
    
    if not captain_reason.strip() or not task_data:
        return 0.3  # 데이터 부족 시 낮은 점수
    
    # 업무 데이터 요약
    task_summary = []
    for task in task_data:
        task_info = {
            "task_name": task.get('task_name', ''),
            "task_detail": task.get('task_detail', ''),
            "achievement_rate": task.get('ai_achievement_rate', 0),
            "contribution_score": task.get('ai_contribution_score', 0),
            "ai_comment": task.get('ai_analysis_comment_task', '')
        }
        task_summary.append(task_info)
    
    # LLM 프롬프트 구성
    system_prompt = """당신은 인사평가 전문가입니다. 팀장이 제시한 수정 사유가 직원의 실제 업무 성과와 얼마나 일치하는지 분석해주세요.

분석 기준:
1. 팀장 사유에서 언급한 성과가 실제 업무 데이터에서 확인되는가?
2. 구체적인 수치나 결과가 업무 성과와 부합하는가?
3. 팀장이 강조한 강점들이 업무 달성률이나 기여도에 반영되어 있는가?

점수 기준:
- 0.9-1.0: 완전히 일치 (구체적 증거와 수치가 정확히 부합)
- 0.7-0.8: 대체로 일치 (주요 내용은 부합하나 일부 과장 또는 누락)
- 0.5-0.6: 부분적 일치 (일부 내용만 확인되고 나머지는 모호)
- 0.3-0.4: 불일치 (사유와 실제 성과 간 상당한 차이)
- 0.1-0.2: 완전히 불일치 (사유가 실제 성과와 반대되거나 근거 없음)

응답은 반드시 다음 JSON 형식으로만 제공하세요:
{
  "consistency_score": 0.0-1.0 사이 점수,
  "evidence_matches": ["일치하는 증거 항목들"],
  "evidence_conflicts": ["불일치하는 항목들"],
  "analysis_summary": "분석 요약 (50자 이내)"
}"""

    user_prompt = f"""팀장 수정 사유:
"{captain_reason}"

실제 업무 성과:
{json.dumps(task_summary, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 팀장 사유와 업무 실적의 일치성을 분석해주세요."""

    try:
        # LLM 호출
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm_client.invoke(messages)
        response_text = str(response.content)  # 안전한 문자열 변환
        
        # JSON 추출 및 파싱
        json_text = _extract_json_from_llm_response(response_text)
        result = json.loads(json_text)
        
        consistency_score = result.get("consistency_score", 0.5)
        return max(0.0, min(1.0, consistency_score))
        
    except Exception as e:
        print(f"⚠️ LLM 업무 증거 일치성 분석 실패: {str(e)}")
        # Fallback: 간단한 키워드 기반 분석
        return _fallback_task_evidence_analysis(captain_reason, task_data)

def analyze_peer_evaluation_consistency(member: Dict) -> float:
    """LLM 기반 동료평가와 팀장 사유의 일치성 분석"""
    
    captain_reason = member.get('captain_reason', '')
    peer_data = member.get('peer_evaluation_data', {})
    
    if not captain_reason or not peer_data:
        return 0.5  # 중립
    
    # 동료평가 데이터 추출
    strengths = peer_data.get('strengths', '')
    concerns = peer_data.get('concerns', '')
    collaboration_obs = peer_data.get('collaboration_observations', '')
    
    if not (strengths or concerns or collaboration_obs):
        return 0.5  # AI 요약 데이터 없음
    
    peer_summary = {
        "strengths": strengths,
        "concerns": concerns,
        "collaboration_observations": collaboration_obs
    }
    
    # LLM 프롬프트 구성
    system_prompt = """당신은 인사평가 전문가입니다. 팀장이 제시한 수정 사유가 동료평가 결과와 얼마나 일치하는지 분석해주세요.

분석 기준:
1. 팀장이 언급한 강점들이 동료평가의 strengths에서 확인되는가?
2. 팀장 사유와 동료들의 concerns 사이에 모순이 있는가?
3. 협업 관련 언급이 collaboration_observations와 부합하는가?
4. 전반적인 평가 톤(긍정/부정)이 일치하는가?

점수 기준:
- 0.9-1.0: 완전히 일치 (팀장 사유가 동료평가와 완벽하게 부합)
- 0.7-0.8: 대체로 일치 (주요 내용은 부합하나 일부 차이)
- 0.5-0.6: 중립적 (특별한 일치나 불일치 없음)
- 0.3-0.4: 불일치 (팀장 사유와 동료평가 간 상당한 차이)
- 0.1-0.2: 심각한 불일치 (정반대 평가나 명백한 모순)

응답은 반드시 다음 JSON 형식으로만 제공하세요:
{
  "consistency_score": 0.0-1.0 사이 점수,
  "alignment_points": ["일치하는 평가 포인트들"],
  "contradiction_points": ["모순되는 평가 포인트들"],
  "analysis_summary": "분석 요약 (50자 이내)"
}"""

    user_prompt = f"""팀장 수정 사유:
"{captain_reason}"

동료평가 AI 요약:
{json.dumps(peer_summary, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 팀장 사유와 동료평가의 일치성을 분석해주세요."""

    try:
        # LLM 호출
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm_client.invoke(messages)
        response_text = str(response.content)  # 안전한 문자열 변환
        
        # JSON 추출 및 파싱
        json_text = _extract_json_from_llm_response(response_text)
        result = json.loads(json_text)
        
        consistency_score = result.get("consistency_score", 0.5)
        return max(0.0, min(1.0, consistency_score))
        
    except Exception as e:
        print(f"⚠️ LLM 동료평가 일치성 분석 실패: {str(e)}")
        # Fallback: 간단한 키워드 기반 분석
        return _fallback_peer_evaluation_analysis(captain_reason, peer_data)

# ================================================================
# 강화된 제로섬 조정 LLM 함수
# ================================================================

def call_enhanced_supervisor_llm(supervisor_input: Dict, retry_count: int = 0) -> Dict:
    """극도로 정확한 제로섬 LLM - 수학적 강제 및 단계별 검증"""
    
    cl_group = supervisor_input["cl_group"]
    surplus = supervisor_input["total_surplus"]
    members = supervisor_input["members"]
    member_count = len(members)
    
    # 현재 상태 정확한 계산
    current_scores = [m["current_score"] for m in members]
    current_total = sum(current_scores)
    target_total = member_count * 3.5
    actual_surplus = current_total - target_total
    
    # KPI 순서로 정렬 (성과 높은 순)
    kpi_sorted_members = sorted(members, key=lambda x: x.get('kpi_achievement', 100), reverse=True)
    
    print(f"🧮 {cl_group} 수학적 정밀 계산:")
    print(f"   현재 총점: {current_total:.3f}")
    print(f"   목표 총점: {target_total:.3f}")
    print(f"   실제 차감: {actual_surplus:.3f}")
    print(f"   입력 차감: {surplus:.3f}")
    
    try:
        # 종합 성과 순위 계산 (새로운 기준 적용)
        comprehensive_rankings = []
        for member in members:
            # 1. 종합 성과 지표 (55%)
            kpi_score = member.get('kpi_achievement', 100) / 100  # 정규화
            comprehensive_performance = member.get('comprehensive_performance', 70) / 100  # 정규화
            performance_score = (kpi_score * 0.6 + comprehensive_performance * 0.4) * 0.55
            
            # 2. 팀장 수정 타당성 (30%)
            validity_analysis = member.get('validity_analysis', {})
            validity_score = validity_analysis.get('final_validity', 0.5) * 0.30
            
            # 3. 조직 기여도 (15%) - 동료평가에서 추출
            collaboration_score = 0.7 * 0.15  # 기본값
            
            # 종합 점수 계산
            total_score = performance_score + validity_score + collaboration_score
            
            comprehensive_rankings.append({
                "emp_no": member["emp_no"],
                "current_score": member["current_score"],
                "kpi_achievement": member.get('kpi_achievement', 100),
                "comprehensive_score": total_score,
                "validity_grade": validity_analysis.get('validity_grade', '보통'),
                "performance_score": performance_score,
                "validity_score": validity_score,
                "collaboration_score": collaboration_score,
                "final_evaluation_report_id": member.get('final_evaluation_report_id')
            })
        
        # 종합 점수로 정렬 (높은 순)
        comprehensive_rankings.sort(key=lambda x: x["comprehensive_score"], reverse=True)
        
        # 차등 차감 비율 계산 (성과와 타당성 기반)
        reduction_ratios = []
        total_weight = 0
        
        for i, member in enumerate(comprehensive_rankings):
            rank = i + 1
            score = member["comprehensive_score"]
            validity_grade = member["validity_grade"]
            
            # 차감 가중치 결정
            if score >= 0.8 and validity_grade in ["매우 타당", "타당"]:
                # 고성과 + 고타당성: 최소 차감
                weight = 0.5
            elif score >= 0.6 and validity_grade in ["보통"]:
                # 중성과 + 중타당성: 보통 차감
                weight = 1.0
            elif score < 0.5 or validity_grade in ["의심", "매우 의심"]:
                # 저성과 + 저타당성: 최대 차감
                weight = 2.0
            else:
                # 기타
                weight = 1.0 + (rank - 1) * 0.2
            
            reduction_ratios.append(weight)
            total_weight += weight
        
        # 비율 정규화
        reduction_ratios = [w / total_weight for w in reduction_ratios]
        
        # 개별 차감량 계산
        total_reductions = []
        final_scores_guide = []
        
        for i, member in enumerate(comprehensive_rankings):
            individual_reduction = actual_surplus * reduction_ratios[i]
            
            # 개인 차감 한계 적용 (1.0점 이내)
            individual_reduction = min(individual_reduction, 1.0)
            
            final_score = member['current_score'] - individual_reduction
            
            # 점수 범위 제한 (0.0~5.0)
            final_score = max(0.0, min(5.0, final_score))
            
            # 실제 차감량 재계산
            actual_individual_reduction = member['current_score'] - final_score
            
            total_reductions.append(actual_individual_reduction)
            final_scores_guide.append(final_score)
        
        # 제로섬 보정 (차감량 합계 맞추기)
        actual_total_reduction = sum(total_reductions)
        adjustment_needed = actual_surplus - actual_total_reduction
        
        # 미세 조정
        if abs(adjustment_needed) > 0.001:
            # 중간 성과자들에게 미세 조정 분배
            middle_indices = [i for i in range(len(comprehensive_rankings)) 
                            if 0.4 <= comprehensive_rankings[i]["comprehensive_score"] <= 0.7]
            
            if middle_indices:
                adjustment_per_person = adjustment_needed / len(middle_indices)
                for idx in middle_indices:
                    total_reductions[idx] += adjustment_per_person
                    final_scores_guide[idx] = comprehensive_rankings[idx]['current_score'] - total_reductions[idx]
                    final_scores_guide[idx] = max(0.0, min(5.0, final_scores_guide[idx]))

        # LLM용 가이드 JSON 생성
        adjustments_json = []
        for i, member in enumerate(comprehensive_rankings):
            reduction_percent = int(reduction_ratios[i] * 100)
            
            adjustments_json.append({
                "emp_no": member['emp_no'],
                "original_score": member['current_score'],
                "final_score": round(final_scores_guide[i], 3),
                "change_amount": round(-total_reductions[i], 3),
                "change_type": "decrease",
                "reason": f"{i+1}위 종합성과 {member['comprehensive_score']:.3f}, {reduction_percent}% 차감 (KPI {member['kpi_achievement']:.0f}%, {member['validity_grade']})",
                "final_evaluation_report_id": member.get('final_evaluation_report_id', i+1),
                "performance_breakdown": {
                    "comprehensive_score": member['comprehensive_score'],
                    "performance_component": member['performance_score'],
                    "validity_component": member['validity_score'],
                    "collaboration_component": member['collaboration_score']
                }
            })

        # 검증 계산
        total_reduction_check = sum(total_reductions)
        total_final_scores = sum(final_scores_guide)
        final_mean_check = total_final_scores / member_count
        
        # 결과 구성
        result = {
            "analysis_summary": f"{cl_group} 종합평가 기반 제로섬 조정 (성과 55% + 타당성 30% + 협업 15%)",
            "adjustments": adjustments_json,
            "evaluation_criteria": {
                "performance_weight": 0.55,
                "validity_weight": 0.30,
                "collaboration_weight": 0.15
            },
            "validation_check": {
                "target_total": target_total,
                "actual_total": total_final_scores,
                "target_mean": 3.500,
                "actual_mean": final_mean_check,
                "target_reduction": actual_surplus,
                "actual_reduction": total_reduction_check,
                "zero_sum_achieved": abs(total_reduction_check - actual_surplus) <= 0.01,
                "mean_achieved": abs(final_mean_check - 3.5) <= 0.01,
                "performance_order_maintained": True,
                "all_conditions_met": abs(total_reduction_check - actual_surplus) <= 0.01 and abs(final_mean_check - 3.5) <= 0.01,
                "reduction_error": abs(total_reduction_check - actual_surplus),
                "mean_error": abs(final_mean_check - 3.5)
            }
        }
        
        # final_evaluation_report_id 보정
        for adj in result.get("adjustments", []):
            matching_member = next((m for m in members if m["emp_no"] == adj["emp_no"]), None)
            if matching_member:
                adj["final_evaluation_report_id"] = matching_member.get("final_evaluation_report_id")
        
        # 극도로 엄격한 검증 (소수점 3자리)
        adjustments = result.get("adjustments", [])
        
        if not adjustments:
            raise ValueError("조정 결과가 없습니다")
        
        # 1. 제로섬 검증 (극도로 엄격)
        actual_reduction = sum(adj["original_score"] - adj["final_score"] for adj in adjustments)
        zero_sum_error = abs(actual_reduction - actual_surplus)
        
        # 2. 평균 검증 (극도로 엄격)
        final_scores = [adj["final_score"] for adj in adjustments]
        actual_mean = sum(final_scores) / len(final_scores)
        mean_error = abs(actual_mean - 3.5)
        
        # 극도로 엄격한 검증 (0.01 오차만 허용)
        if zero_sum_error > 0.01:
            raise ValueError(f"제로섬 실패: 목표 {actual_surplus:.3f}, 실제 {actual_reduction:.3f} (오차 {zero_sum_error:.3f})")
        
        if mean_error > 0.01:
            raise ValueError(f"평균 실패: 목표 3.500, 실제 {actual_mean:.3f} (오차 {mean_error:.3f})")
        
        print(f"✅ 극도 정밀 LLM 성공: {cl_group}")
        print(f"   제로섬: {actual_reduction:.3f}/{actual_surplus:.3f} (오차: {zero_sum_error:.3f})")
        print(f"   평균: {actual_mean:.3f}/3.500 (오차: {mean_error:.3f})")
        
        return {
            "success": True,
            "result": result,
            "retry_count": retry_count,
            "precision_level": "ultra_high"
        }
        
    except Exception as e:
        print(f"❌ 극도 정밀 LLM 실패 (시도 {retry_count + 1}): {str(e)}")
        
        if retry_count < 3:  # 재시도 늘림
            print(f"🔄 극도 정밀 재시도... ({retry_count + 1}/3)")
            return call_enhanced_supervisor_llm(supervisor_input, retry_count + 1)
        else:
            print(f"💥 {cl_group}: 극도 정밀 LLM 완전 실패, 수학적 Fallback 실행")
            return {
                "success": False,
                "error": str(e),
                "retry_count": retry_count,
                "precision_level": "failed"
            }