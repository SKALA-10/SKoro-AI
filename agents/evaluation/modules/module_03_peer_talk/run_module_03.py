# 실행 스크립트 (모듈2 패턴 따라함)

from typing import List
from agents.evaluation.modules.module_03_peer_talk.agent import *
from agents.evaluation.modules.module_03_peer_talk.db_utils import *
from langchain_core.messages import HumanMessage

def run_module3_single_employee(period_id: int, target_emp_no: str):
    """모듈 4 단일 직원 분석 실행"""
    # State 정의
    state = Module3AgentState(
        messages=[HumanMessage(content=f"모듈 4: {target_emp_no} 동료평가 분석 시작")],
        period_id=period_id,
        target_emp_no=target_emp_no,
        peer_evaluation_ids=[],
        evaluator_emp_nos=[],
        evaluation_weights=[],
        keyword_collections=[],
        task_summaries=[],
        peer_evaluation_summary_sentences=[],
        strengths=[],
        concerns=[],
        collaboration_observations=[],
        weighted_analysis_result={},
        feedback_report_id=None,
        final_evaluation_report_id=None
    )

    # 그래프 생성 및 실행
    print(f"🎯 모듈 4 실행 시작: {target_emp_no} ({period_id}분기)...")
    module3_graph = create_module3_graph()
    result = module3_graph.invoke(state)
    print(f"✅ 모듈 4 실행 완료: {target_emp_no}")
    print(f"최종 메시지: {result['messages'][-1].content}")
    
    # 결과 요약 출력
    if result.get('strengths'):
        print(f"강점: {result['strengths'][0][:50]}...")
    if result.get('concerns'):
        print(f"우려: {result['concerns'][0][:50]}...")
    if result.get('collaboration_observations'):
        print(f"협업관찰: {result['collaboration_observations'][0][:50]}...")
    
    return result

def run_module3_multiple_employees(period_id: int, emp_list: List[str]):
    """모듈 4 여러 직원 일괄 분석 실행"""
    print(f"🚀 모듈 4 일괄 분석 시작: {len(emp_list)}명 ({period_id}분기)")
    
    # 저장될 테이블 정보 표시
    table_name = "final_evaluation_reports" if period_id == 4 else "feedback_reports"
    print(f"💾 저장 테이블: {table_name}.ai_peer_talk_summary")
    
    results = {}
    success_count = 0
    skipped_count = 0
    
    for i, emp_no in enumerate(emp_list, 1):
        print(f"\n[{i}/{len(emp_list)}] 📊 {emp_no} 분석 시작...")
        print("-" * 60)
        
        # 기존 분석 결과 확인
        if check_existing_peer_evaluation_by_period(period_id, emp_no):
            print(f"⏭️ {emp_no}: 이미 분석 완료 ({table_name}), 스킵")
            skipped_count += 1
            results[emp_no] = "already_completed"
            continue
        
        try:
            result = run_module3_single_employee(period_id, emp_no)
            
            # 결과 검증
            if (result and result.get('strengths') and result.get('concerns') and result.get('collaboration_observations')):
                print(f"✅ {emp_no} 분석 완료!")
                success_count += 1
            else:
                print(f"⚠️ {emp_no} 분석 완료되었으나 결과가 불완전")
            
            results[emp_no] = result
            
        except Exception as e:
            print(f"❌ {emp_no} 분석 실패: {str(e)}")
            results[emp_no] = None
    
    # 요약 출력
    print(f"\n" + "="*80)
    print(f"📊 분석 결과 요약 ({table_name} 테이블)")
    print(f"="*80)
    print(f"총 대상자: {len(emp_list)}명")
    print(f"성공: {success_count}명")
    print(f"스킵 (기존 완료): {skipped_count}명")
    print(f"실패: {len(emp_list) - success_count - skipped_count}명")
    
    return results

def run_module3_period_analysis(period_id: int):
    """모듈 4 분기 전체 분석 실행"""
    print(f"="*80)
    print(f"🚀 모듈 4: {period_id}분기 전체 직원 동료평가 분석 시작")
    print(f"="*80)
    
    # 해당 분기의 모든 평가 대상자 조회
    emp_list = get_all_employees_in_period(period_id)
    
    if not emp_list:
        print(f"❌ {period_id}분기에 분석할 직원이 없습니다.")
        return {}
    
    print(f"📊 {period_id}분기 동료평가 대상자: {len(emp_list)}명")
    for i, emp_no in enumerate(emp_list, 1):
        print(f"  {i}. {emp_no}")
    
    # 일괄 분석 실행
    results = run_module3_multiple_employees(period_id, emp_list)
    
    print(f"\n✅ 모듈 4: {period_id}분기 전체 분석 완료!")
    return results

def quick_module3_analysis(emp_no: str, period_id: int):
    """모듈 4 빠른 단일 분석"""
    print(f"🎯 모듈 4 빠른 분석: {emp_no} ({period_id}분기)")
    
    # 기존 분석 결과 확인
    if check_existing_peer_evaluation_by_period(period_id, emp_no):
        table_name = "final_evaluation_reports" if period_id == 4 else "feedback_reports"
        print(f"⏭️ {emp_no}: 이미 분석 완료 ({table_name}.ai_peer_talk_summary)")
        return "already_completed"
    
    result = run_module3_single_employee(period_id, emp_no)
    
    if result and result.get('strengths'):
        print("\n✅ 분석 결과:")
        print(f"강점: {result['strengths'][0]}")
        print(f"우려: {result['concerns'][0]}")  
        print(f"협업관찰: {result['collaboration_observations'][0]}")
    else:
        print("❌ 분석 실패 또는 결과 없음")
    
    return result


if __name__ == "__main__":
    # 예시 실행
    # 단일 직원 분석
    # quick_module3_analysis("E002", 3)
    
    # # 분기 전체 분석
    run_module3_period_analysis(2)
    run_module3_period_analysis(4)
    
    print("\n🎉 모듈 4 실행 스크립트 로드 완료!")
    # print("사용 가능한 함수들:")
    # print("- run_module3_single_employee(period_id, target_emp_no)")
    # print("- run_module3_multiple_employees(period_id, emp_list)")
    # print("- run_module3_period_analysis(period_id)")
    # print("- quick_module3_analysis(emp_no, period_id)")