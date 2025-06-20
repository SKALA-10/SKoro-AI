# ================================================================
# run_module_06.py - 모듈 6 실행 
# ================================================================

from agent import create_module6_graph_efficient


def run_module6_evaluation(emp_no: str, report_type: str = "quarterly", 
                          team_id: int = 1, period_id: int = None,
                          feedback_report_id: int = None, 
                          final_evaluation_report_id: int = None):
    """모듈 6 4P 평가 실행 (프로덕션용)"""
    
    print(f"\n{'='*60}")
    print(f"모듈 6 4P 평가 실행 - {emp_no} ({report_type})")
    print(f"{'='*60}")
    
    # period_id 기본값 설정
    if period_id is None:
        period_id = 4 if report_type == "annual" else 2
    
    # State 초기화
    state = {
        "messages": [f"모듈 6 {report_type} 평가 시작"],
        "report_type": report_type,
        "team_id": team_id,
        "period_id": period_id,
        "emp_no": emp_no,
        "feedback_report_id": feedback_report_id,
        "final_evaluation_report_id": final_evaluation_report_id,
        "raw_evaluation_criteria": "",  # DB에서 채워짐
        "evaluation_criteria": {},  # 캐시에서 채워짐
        "evaluation_results": {},
        "integrated_data": {},
    }
    
    print(f"📍 설정 정보:")
    print(f"  - 직원번호: {emp_no}")
    print(f"  - 평가유형: {report_type}")
    print(f"  - 팀ID: {team_id}")
    print(f"  - 기간ID: {period_id}")
    print(f"  - 분기 리포트 ID: {feedback_report_id}")
    print(f"  - 연말 리포트 ID: {final_evaluation_report_id}")
    
    # 그래프 실행
    module6_graph = create_module6_graph_efficient()
    
    try:
        result = module6_graph.invoke(state)
        
        print(f"\n📊 평가 완료:")
        for message in result.get('messages', []):
            print(f"  {message}")
        
        # 결과 반환
        integrated_result = result.get('integrated_data', {}).get('integrated_4p_result', {})
        evaluation_results = result.get('evaluation_results', {})
        
        return {
            "success": True,
            "integrated_result": integrated_result,
            "evaluation_results": evaluation_results,
            "messages": result.get('messages', [])
        }
        
    except Exception as e:
        print(f"❌ 모듈 6 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "integrated_result": {},
            "evaluation_results": {},
            "messages": [f"평가 실행 실패: {str(e)}"]
        }

