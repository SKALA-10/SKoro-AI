# ================================================================
# run_module8.py - 모듈 8 실행 파일
# ================================================================

import logging
from typing import Dict, Any
from datetime import datetime

from agents.evaluation.modules.module_08_team_comparision.agent import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx 및 관련 라이브러리 로그 레벨 조정 (HTTP 요청 로그 숨김)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# ================================================================
# 메인 파이프라인 함수
# ================================================================

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
        
        # 모듈 8 그래프 생성 및 실행
        module8_graph = create_module8_graph()
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

# ================================================================
# 테스트 및 실행 함수
# ================================================================

def test_module8() -> Dict:
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
        return result

# ================================================================
# 메인 실행 부분
# ================================================================

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