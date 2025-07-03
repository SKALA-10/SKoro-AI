# ================================================================
# run_module_09_1.py - 모듈 9 메인 실행 함수 (함수 순서 수정)
# ================================================================

import logging
from typing import Dict, List
from datetime import datetime
from langchain_core.messages import HumanMessage

from agents.evaluation.modules.module_09_cl_normalization.agent import *
from agents.evaluation.modules.module_09_cl_normalization.db_utils import *
from agents.evaluation.modules.module_09_cl_normalization.llm_utils import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx 및 관련 라이브러리 로그 레벨 조정 (HTTP 요청 로그 숨김)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# ================================================================
# 유틸리티 함수들 (먼저 정의)
# ================================================================

def print_enhanced_workflow_summary(state: Module9AgentState):
    """향상된 워크플로우 실행 결과 요약 출력"""
    
    print(f"📋 향상된 워크플로우 실행 결과 요약:")
    
    # 메시지 히스토리
    messages = state.get("messages", [])
    if messages:
        print(f"\n📝 실행 단계:")
        for i, msg in enumerate(messages, 1):
            print(f"   {i}. {msg.content}")
    
    # 1단계 결과
    department_data = state.get("department_data", {})
    if department_data:
        print(f"\n🔍 1단계 - 향상된 데이터 수집 결과:")
        total_members = sum(cl_data.get("member_count", 0) for cl_data in department_data.values())
        adjustment_needed = len([cl for cl, data in department_data.items() if data.get("needs_adjustment", False)])
        total_surplus = sum(cl_data.get("surplus", 0) for cl_data in department_data.values())
        
        print(f"   - 총 인원: {total_members}명")
        print(f"   - 조정 필요 CL: {adjustment_needed}개")
        print(f"   - 총 초과분: {total_surplus:+.2f}점")
        
        for cl_group, cl_data in department_data.items():
            status = "🔧 조정 필요" if cl_data.get("needs_adjustment") else "✅ 조정 불필요"
            validity_summary = cl_data.get("validity_summary", {})
            high_validity = validity_summary.get("매우 타당", 0) + validity_summary.get("타당", 0)
            print(f"   - {cl_group}: {cl_data.get('member_count', 0)}명, {cl_data.get('surplus', 0):+.2f}점 {status} (고타당성 {high_validity}명)")
    
    # 2단계 결과
    enhanced_analysis = state.get("enhanced_analysis", {})
    if enhanced_analysis:
        print(f"\n🧠 2단계 - 향상된 타당성 분석 결과:")
        analyzed_cls = [analysis for analysis in enhanced_analysis.values() if analysis.get("analysis_completed")]
        total_analyzed = sum(analysis.get("members_analyzed", 0) for analysis in analyzed_cls)
        
        if analyzed_cls:
            avg_validity = sum(analysis["avg_validity"] for analysis in analyzed_cls) / len(analyzed_cls)
            total_high_validity = sum(analysis["high_validity_count"] for analysis in analyzed_cls)
            total_low_validity = sum(analysis["low_validity_count"] for analysis in analyzed_cls)
            
            print(f"   - 분석 완료 CL: {len(analyzed_cls)}개")
            print(f"   - 총 분석 인원: {total_analyzed}명")
            print(f"   - 평균 타당성: {avg_validity:.3f}")
            print(f"   - 고타당성 인원: {total_high_validity}명")
            print(f"   - 저타당성 인원: {total_low_validity}명")
            
            for cl_group, analysis in enhanced_analysis.items():
                if analysis.get("analysis_completed"):
                    recommendation = analysis["analysis_summary"]["recommendation"]
                    print(f"   - {cl_group}: 평균 {analysis['avg_validity']:.3f} ({recommendation})")
    
    # 3단계 결과
    supervisor_results = state.get("supervisor_results", {})
    if supervisor_results:
        print(f"\n🤖 3단계 - 향상된 AI Supervisor 결과:")
        successful_cls = len([cl for cl, result in supervisor_results.items() if result.get("success", False)])
        total_adjustments = sum(result.get("adjustments_made", 0) for result in supervisor_results.values())
        fallback_used = len([cl for cl, result in supervisor_results.items() if result.get("fallback_used", False)])
        enhanced_used = len([cl for cl, result in supervisor_results.items() if result.get("enhanced_features", {}).get("validity_analysis_used", False)])
        
        print(f"   - 성공한 CL: {successful_cls}/{len(supervisor_results)}개")
        print(f"   - 총 조정 인원: {total_adjustments}명")
        print(f"   - Fallback 사용: {fallback_used}개 CL")
        print(f"   - 향상된 기능 사용: {enhanced_used}개 CL")
        
        for cl_group, result in supervisor_results.items():
            status_icon = "✅" if result.get("success") else "❌"
            enhanced_note = " (향상됨)" if result.get("enhanced_features", {}).get("validity_analysis_used") else ""
            print(f"   - {cl_group}: {status_icon} {result.get('adjustments_made', 0)}명 조정{enhanced_note}")
    
    # 4단계 결과
    update_results = state.get("update_results", {})
    if update_results:
        print(f"\n📊 4단계 - 향상된 최종 집계 결과:")
        print(f"   - 성공률: {update_results.get('success_rate', 0):.1f}%")
        print(f"   - 분포 달성률: {update_results.get('distribution_rate', 0):.1f}%")
        print(f"   - 향상된 기능 적용률: {update_results.get('enhanced_rate', 0):.1f}%")
        print(f"   - 평균 처리시간: {update_results.get('avg_processing_time_ms', 0):.0f}ms")
        print(f"   - 총 처리시간: {update_results.get('total_processing_time_ms', 0):.0f}ms")
    
    # 에러 로그
    error_logs = state.get("error_logs", [])
    if error_logs:
        print(f"\n⚠️ 에러 로그:")
        for error in error_logs:
            print(f"   - {error}")
    
    # 최종 요약
    total_processed = state.get("total_processed", 0)
    total_failed = state.get("total_failed", 0)
    print(f"\n🏁 향상된 최종 결과:")
    print(f"   ✅ 성공: {total_processed}명")
    print(f"   ❌ 실패: {total_failed}명")
    print(f"   📈 성공률: {(total_processed / (total_processed + total_failed) * 100) if (total_processed + total_failed) > 0 else 0:.1f}%")
    print(f"   🚀 향상된 기능: 업무증거분석, 동료평가통합, 다면검증 적용")

def generate_enhanced_summary_report(state: Module9AgentState) -> Dict:
    """향상된 최종 요약 보고서 생성"""
    
    headquarter_id = state["headquarter_id"]
    period_id = state["period_id"]
    department_data = state.get("department_data", {})
    enhanced_analysis = state.get("enhanced_analysis", {})
    supervisor_results = state.get("supervisor_results", {})
    update_results = state.get("update_results", {})
    
    # 처리 전 현황
    initial_stats = {
        "total_cls": len(department_data),
        "total_members": sum(cl_data.get("member_count", 0) for cl_data in department_data.values()),
        "adjustment_needed_cls": len([cl for cl, data in department_data.items() if data.get("needs_adjustment", False)]),
        "total_surplus": sum(cl_data.get("surplus", 0) for cl_data in department_data.values())
    }
    
    # 향상된 분석 결과
    enhanced_stats = {
        "analyzed_cls": len([analysis for analysis in enhanced_analysis.values() if analysis.get("analysis_completed")]),
        "avg_validity_score": 0,
        "high_validity_members": 0,
        "low_validity_members": 0,
        "validity_distribution": {"매우 타당": 0, "타당": 0, "보통": 0, "의심": 0, "매우 의심": 0}
    }
    
    # 타당성 통계 계산
    all_validities = []
    for analysis in enhanced_analysis.values():
        if analysis.get("analysis_completed"):
            all_validities.append(analysis["avg_validity"])
            enhanced_stats["high_validity_members"] += analysis["high_validity_count"]
            enhanced_stats["low_validity_members"] += analysis["low_validity_count"]
            
            # 분포 집계
            for grade, members in analysis.get("validity_distribution", {}).items():
                enhanced_stats["validity_distribution"][grade] += len(members)
    
    if all_validities:
        enhanced_stats["avg_validity_score"] = round(sum(all_validities) / len(all_validities), 3)
    
    # 처리 후 결과
    final_stats = {
        "processed_cls": update_results.get("successful_cls", 0),
        "adjusted_members": update_results.get("total_adjustments", 0),
        "distribution_achieved_cls": update_results.get("distribution_achieved_count", 0),
        "fallback_used_cls": update_results.get("fallback_used_count", 0),
        "enhanced_features_used_cls": update_results.get("enhanced_features_used_count", 0)
    }
    
    # 성과 지표
    performance_metrics = {
        "success_rate": update_results.get("success_rate", 0),
        "distribution_rate": update_results.get("distribution_rate", 0),
        "enhanced_rate": update_results.get("enhanced_rate", 0),
        "avg_processing_time_ms": update_results.get("avg_processing_time_ms", 0),
        "total_processing_time_ms": update_results.get("total_processing_time_ms", 0)
    }
    
    enhanced_summary_report = {
        "headquarter_id": headquarter_id,
        "period_id": period_id,
        "execution_timestamp": datetime.now().isoformat(),
        "version": "enhanced_v2.0",
        "features": ["업무증거분석", "동료평가통합", "향상된타당성판단"],
        "initial_stats": initial_stats,
        "enhanced_analysis_stats": enhanced_stats,
        "final_stats": final_stats,
        "performance_metrics": performance_metrics,
        "cl_details": update_results.get("cl_summaries", []),
        "successful_employees": update_results.get("successful_employees", []),
        "failed_employees": update_results.get("failed_employees", []),
        "messages": [msg.content for msg in state.get("messages", [])],
        "error_logs": state.get("error_logs", [])
    }
    
    return enhanced_summary_report

def run_enhanced_module9_workflow_fixed(headquarter_id: int, period_id: int = 4):
    """향상된 모듈9 완전한 워크플로우 실행"""
    
    print(f"🚀 향상된 모듈9 워크플로우 실행 시작")
    print(f"   본부: {headquarter_id}, 기간: {period_id}")
    print(f"   특징: 업무증거분석 + 동료평가통합 + 향상된타당성판단")
    print(f"   {'='*70}")
    
    # 초기 State 생성
    initial_state = Module9AgentState(
        messages=[HumanMessage(content=f"향상된 모듈9 시작: 본부 {headquarter_id}")],
        headquarter_id=headquarter_id,
        period_id=period_id,
        department_data={},
        enhanced_analysis={},
        supervisor_results={},
        update_results={},
        total_processed=0,
        total_failed=0,
        error_logs=[]
    )
    
    try:
        # LangGraph 워크플로우 생성 및 실행
        enhanced_module9_graph = create_enhanced_module9_graph()
        
        print(f"🔄 향상된 워크플로우 실행 중...")
        result_state = enhanced_module9_graph.invoke(initial_state)
        
        # 실행 완료 메시지
        print(f"\n✅ 향상된 모듈9 워크플로우 실행 완료!")
        print(f"   {'='*70}")
        
        # 최종 결과 요약
        print_enhanced_workflow_summary(result_state)
        
        # 향상된 최종 보고서 생성
        enhanced_final_report = generate_enhanced_summary_report(result_state)
        
        return {
            "success": True,
            "final_state": result_state,
            "enhanced_summary_report": enhanced_final_report,
            "total_processed": result_state.get("total_processed", 0),
            "total_failed": result_state.get("total_failed", 0)
        }
        
    except Exception as e:
        print(f"❌ 향상된 모듈9 워크플로우 실행 실패: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "total_processed": 0,
            "total_failed": 0
        }

# ================================================================
# 메인 파이프라인 함수 (유틸리티 함수들 다음에 정의)
# ================================================================

def execute_module9_pipeline(headquarter_id: int, period_id: int = 4) -> Dict:
    """모듈 9 향상된 제로섬 조정 평가 실행"""
    
    start_time = datetime.now()
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 모듈 9: 향상된 제로섬 조정 분석 시작")
    logger.info(f"{'='*60}")
    logger.info(f"📍 설정 정보:")
    logger.info(f"   본부 ID: {headquarter_id}")
    logger.info(f"   기간 ID: {period_id}")
    logger.info(f"   특징: 업무증거분석 + 동료평가통합 + 향상된타당성판단")
    
    try:
        # 초기 State 생성
        initial_state = Module9AgentState(
            messages=[HumanMessage(content=f"향상된 모듈9 시작: 본부 {headquarter_id}")],
            headquarter_id=headquarter_id,
            period_id=period_id,
            department_data={},
            enhanced_analysis={},
            supervisor_results={},
            update_results={},
            total_processed=0,
            total_failed=0,
            error_logs=[]
        )
        
        logger.info(f"🚀 모듈 9: 향상된 제로섬 조정 분석 시작 (본부 {headquarter_id}, Q{period_id})")
        
        # 모듈 9 그래프 생성 및 실행
        module9_graph = create_enhanced_module9_graph()
        result_state = module9_graph.invoke(initial_state)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        success_result = {
            "status": "success",
            "execution_time_seconds": execution_time,
            "headquarter_id": headquarter_id,
            "period_id": period_id,
            "results": {
                "total_processed": result_state.get("total_processed", 0),
                "total_failed": result_state.get("total_failed", 0),
                "successful_cls": result_state.get("update_results", {}).get("successful_cls", 0),
                "total_adjustments": result_state.get("update_results", {}).get("total_adjustments", 0),
                "success_rate": result_state.get("update_results", {}).get("success_rate", 0),
                "enhanced_features_used": result_state.get("update_results", {}).get("enhanced_features_used_count", 0)
            },
            "messages": [msg.content for msg in result_state.get("messages", [])],
            "final_state": result_state
        }
        
        logger.info("\n✅ 모듈 9 실행 완료!")
        logger.info("📋 실행 과정:")
        for i, message in enumerate(result_state.get("messages", []), 1):
            logger.info(f"  {i}. {message.content}")
        
        logger.info(f"\n📊 최종 결과:")
        logger.info(f"처리된 직원: {success_result['results']['total_processed']}명")
        logger.info(f"실패한 직원: {success_result['results']['total_failed']}명")
        logger.info(f"성공한 CL: {success_result['results']['successful_cls']}개")
        logger.info(f"총 조정 인원: {success_result['results']['total_adjustments']}명")
        logger.info(f"성공률: {success_result['results']['success_rate']:.1f}%")
        logger.info(f"향상된 기능 사용: {success_result['results']['enhanced_features_used']}개 CL")
        logger.info(f"⏱️  실행 시간: {execution_time:.2f}초")
        logger.info(f"{'='*60}")
        
        # 상세 요약 출력 - 이제 함수가 정의되었으므로 호출 가능
        print_enhanced_workflow_summary(result_state)
        
        return success_result
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_result = {
            "status": "error",
            "execution_time_seconds": execution_time,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "headquarter_id": headquarter_id,
            "period_id": period_id
        }
        
        logger.error(f"\n❌ 모듈 9 실행 실패!")
        logger.error(f"⏱️  실행 시간: {execution_time:.2f}초")
        logger.error(f"💥 오류: {e}")
        logger.error(f"🔍 오류 유형: {type(e).__name__}")
        logger.error(f"{'='*60}")
        
        return error_result

# ================================================================
# 테스트 및 실행 함수
# ================================================================

def test_module9() -> Dict:
    """모듈 9 테스트 실행"""
    logger.info("=== 모듈 9 향상된 테스트 시작 ===")
    
    # 기본 테스트
    result = execute_module9_pipeline(headquarter_id=1, period_id=4)
    
    if result and result.get("status") == "success":
        logger.info(f"\n📊 최종 테스트 결과:")
        logger.info(f"상태: {result['status']}")
        logger.info(f"실행 시간: {result['execution_time_seconds']:.2f}초")
        logger.info(f"처리된 직원: {result['results']['total_processed']}명")
        logger.info(f"실패한 직원: {result['results']['total_failed']}명")
        logger.info(f"성공한 CL: {result['results']['successful_cls']}개")
        logger.info(f"총 조정 인원: {result['results']['total_adjustments']}명")
        logger.info(f"성공률: {result['results']['success_rate']:.1f}%")
        logger.info(f"향상된 기능 사용: {result['results']['enhanced_features_used']}개 CL")
        
        return result
    else:
        logger.error("테스트 실패!")
        if result:
            logger.error(f"오류: {result.get('error_message', 'Unknown error')}")
            logger.error(f"오류 유형: {result.get('error_type', 'Unknown')}")
        return result

def test_enhanced_module9_fixed():
    """향상된 모듈9 테스트 실행"""
    
    print("🧪 향상된 모듈9 테스트 시작")
    print("="*60)
    
    # 테스트 실행
    try:
        # 단일 본부 테스트
        result = run_enhanced_module9_workflow_fixed(1, 4)
        
        if result["success"]:
            print("✅ 향상된 모듈9 테스트 성공!")
            print(f"   처리된 직원: {result['total_processed']}명")
            print(f"   실패한 직원: {result['total_failed']}명")
        else:
            print("❌ 향상된 모듈9 테스트 실패!")
            print(f"   오류: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"💥 테스트 실행 중 오류: {str(e)}")

def run_multiple_headquarters_module9(headquarter_ids: List[int], period_id: int = 4):
    """여러 본부 향상된 일괄 실행"""
    
    logger.info(f"🏢 여러 본부 향상된 모듈9 일괄 실행: {len(headquarter_ids)}개 본부")
    logger.info(f"   대상 본부: {headquarter_ids}")
    logger.info(f"   향상된 기능: 업무증거분석 + 동료평가통합 + 다면검증")
    logger.info(f"   {'='*80}")
    
    results = {}
    total_success = 0
    total_failed = 0
    total_processed_employees = 0
    total_failed_employees = 0
    
    for i, hq_id in enumerate(headquarter_ids, 1):
        logger.info(f"\n🏢 본부 {hq_id} 향상된 처리 중... ({i}/{len(headquarter_ids)})")
        logger.info(f"   {'-'*60}")
        
        try:
            result = execute_module9_pipeline(hq_id, period_id)
            results[hq_id] = result
            
            if result["status"] == "success":
                total_success += 1
                total_processed_employees += result["results"]["total_processed"]
                total_failed_employees += result["results"]["total_failed"]
            else:
                total_failed += 1
                
        except Exception as e:
            logger.error(f"❌ 본부 {hq_id} 향상된 처리 실패: {str(e)}")
            results[hq_id] = {"status": "error", "error_message": str(e)}
            total_failed += 1
    
    # 전체 결과 요약
    logger.info(f"\n🎉 여러 본부 향상된 처리 완료!")
    logger.info(f"   {'='*80}")
    logger.info(f"🏁 전체 향상된 결과 요약:")
    logger.info(f"   - 성공한 본부: {total_success}/{len(headquarter_ids)}개")
    logger.info(f"   - 실패한 본부: {total_failed}/{len(headquarter_ids)}개")
    logger.info(f"   - 본부 성공률: {(total_success / len(headquarter_ids) * 100):.1f}%")
    logger.info(f"   - 총 처리 직원: {total_processed_employees}명")
    logger.info(f"   - 총 실패 직원: {total_failed_employees}명")
    logger.info(f"   - 직원 성공률: {(total_processed_employees / (total_processed_employees + total_failed_employees) * 100) if (total_processed_employees + total_failed_employees) > 0 else 0:.1f}%")
    logger.info(f"   🚀 적용된 향상 기능: 업무실적검증, 동료평가일치성분석, 종합타당성판단")
    
    return results

def run_multiple_headquarters_enhanced_module9_fixed(headquarter_ids: List[int], period_id: int = 4):
    """여러 본부 향상된 일괄 실행"""
    
    print(f"🏢 여러 본부 향상된 모듈9 일괄 실행: {len(headquarter_ids)}개 본부")
    print(f"   대상 본부: {headquarter_ids}")
    print(f"   향상된 기능: 업무증거분석 + 동료평가통합 + 다면검증")
    print(f"   {'='*80}")
    
    results = {}
    total_success = 0
    total_failed = 0
    total_processed_employees = 0
    total_failed_employees = 0
    
    for i, hq_id in enumerate(headquarter_ids, 1):
        print(f"\n🏢 본부 {hq_id} 향상된 처리 중... ({i}/{len(headquarter_ids)})")
        print(f"   {'-'*60}")
        
        try:
            result = run_enhanced_module9_workflow_fixed(hq_id, period_id)
            results[hq_id] = result
            
            if result["success"]:
                total_success += 1
                total_processed_employees += result["total_processed"]
                total_failed_employees += result["total_failed"]
            else:
                total_failed += 1
                
        except Exception as e:
            print(f"❌ 본부 {hq_id} 향상된 처리 실패: {str(e)}")
            results[hq_id] = {"success": False, "error": str(e)}
            total_failed += 1
    
    # 전체 결과 요약
    print(f"\n🎉 여러 본부 향상된 처리 완료!")
    print(f"   {'='*80}")
    print(f"🏁 전체 향상된 결과 요약:")
    print(f"   - 성공한 본부: {total_success}/{len(headquarter_ids)}개")
    print(f"   - 실패한 본부: {total_failed}/{len(headquarter_ids)}개")
    print(f"   - 본부 성공률: {(total_success / len(headquarter_ids) * 100):.1f}%")
    print(f"   - 총 처리 직원: {total_processed_employees}명")
    print(f"   - 총 실패 직원: {total_failed_employees}명")
    print(f"   - 직원 성공률: {(total_processed_employees / (total_processed_employees + total_failed_employees) * 100) if (total_processed_employees + total_failed_employees) > 0 else 0:.1f}%")
    print(f"   🚀 적용된 향상 기능: 업무실적검증, 동료평가일치성분석, 종합타당성판단")
    
    return results

# ================================================================
# 메인 실행 부분
# ================================================================

if __name__ == "__main__":
    # 테스트 케이스들
    test_cases = [
        {"headquarter_id": 1, "period_id": 4, "desc": "본부1 Q4 연말평가"},
        {"headquarter_id": 2, "period_id": 4, "desc": "본부2 Q4 연말평가"}
    ]

    for test_case in test_cases:
        logger.info(f"\n🧪 모듈9 향상된 테스트 실행 - {test_case['desc']}")
        logger.info(f"   테스트 본부: {test_case['headquarter_id']}")
        logger.info(f"   테스트 기간: Q{test_case['period_id']}")
        
        try:
            result = run_enhanced_module9_workflow_fixed(
                test_case['headquarter_id'], 
                test_case['period_id']
            )
            
            if result.get('success'):
                logger.info(f"\n🎉 테스트 성공!")
                logger.info(f"📊 실행 결과 요약:")
                logger.info(f"   • 상태: {result.get('success', False)}")
                logger.info(f"   • 처리된 직원: {result.get('total_processed', 0)}명")
                logger.info(f"   • 실패한 직원: {result.get('total_failed', 0)}명")
                logger.info(f"   • 성공률: {(result.get('total_processed', 0) / (result.get('total_processed', 0) + result.get('total_failed', 0)) * 100) if (result.get('total_processed', 0) + result.get('total_failed', 0)) > 0 else 0:.1f}%")
            else:
                logger.error(f"\n❌ 테스트 실패!")
                logger.error(f"   • 오류: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"\n💥 테스트 실행 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info("🏁 모듈9 향상된 테스트 완료")
    logger.info(f"{'='*60}")

# ================================================================
# 유틸리티 및 안내 메시지
# ================================================================

print("📋 모듈9 실행 함수:")
print("  - execute_module9_pipeline(headquarter_id, period_id)")
print("  - test_module9()")
print("  - run_multiple_headquarters_module9(headquarter_ids, period_id)")
print("  - run_enhanced_module9_workflow_fixed(headquarter_id, period_id)")
print("  - test_enhanced_module9_fixed()")
print("  - run_multiple_headquarters_enhanced_module9_fixed(headquarter_ids, period_id)")
print()
print("🚀 이제 다음과 같이 실행하세요:")
print("   result = execute_module9_pipeline(1, 4)")
print("   또는")
print("   result = run_enhanced_module9_workflow_fixed(1, 4)")
print("   또는")
print("   test_enhanced_module9_fixed()")