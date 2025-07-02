# ================================================================
# run_module9.py - 모듈 9 실행 파일
# ================================================================

from typing import List, Optional
from langchain_core.messages import HumanMessage
from agents.evaluation.modules.module_09_cl_normalization.agent import *
from agents.evaluation.modules.module_09_cl_normalization.db_utils import *

# ================================================================
# 실행 함수들
# ================================================================

def run_headquarter_module9_evaluation(headquarter_id: str, period_id: int = 4):
    """본부 단위 모듈 9 CL별 정규화 실행"""
    
    print(f"🚀 본부 단위 모듈 9 CL별 정규화 실행 시작: {headquarter_id} (period_id: {period_id})")
    
    # State 정의
    state = HeadquarterModule9AgentState(
        messages=[HumanMessage(content=f"본부 {headquarter_id}: CL별 정규화 시작")],
        headquarter_id=headquarter_id,
        period_id=period_id,
        headquarter_members=[],
        cl_groups={},
        normalized_scores=[],
        processed_count=0,
        failed_members=[]
    )
    
    # 그래프 생성 및 실행
    headquarter_module9_graph = create_headquarter_module9_graph()
    
    try:
        result = headquarter_module9_graph.invoke(state)
        
        print("✅ 본부 단위 모듈 9 CL별 정규화 실행 완료!")
        print(f"📊 결과:")
        for message in result['messages']:
            print(f"  - {message.content}")
        
        if result.get('processed_count'):
            print(f"🎯 처리 완료: {result['processed_count']}명")
            if result.get('failed_members'):
                print(f"❌ 실패한 직원: {result['failed_members']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 본부 단위 모듈 9 CL별 정규화 실행 실패: {e}")
        return None

def run_all_headquarters_module9(period_id: int = 4):
    """전체 본부 일괄 실행"""
    
    # 모든 본부 ID 조회
    headquarters = get_all_headquarters_info()
    
    print(f"🚀 전체 본부 모듈 9 CL별 정규화 실행: {len(headquarters)}개 본부")
    
    results = {}
    total_processed = 0
    total_failed = 0
    
    for hq in headquarters:
        headquarter_id = hq["headquarter_id"]
        headquarter_name = hq["headquarter_name"]
        
        print(f"\n{'='*50}")
        print(f"본부 {headquarter_id} ({headquarter_name}) 처리 중...")
        
        result = run_headquarter_module9_evaluation(headquarter_id, period_id)
        results[headquarter_id] = result
        
        if result:
            total_processed += result.get('processed_count', 0)
            total_failed += len(result.get('failed_members', []))
    
    print(f"\n🎯 전체 결과:")
    print(f"   처리된 본부: {len([r for r in results.values() if r is not None])}/{len(headquarters)}")
    print(f"   처리된 인원: {total_processed}명")
    print(f"   실패한 인원: {total_failed}명")
    
    return results

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def test_headquarter_module9(headquarter_id: Optional[str] = None, period_id: int = 4):
    """본부 모듈 9 CL별 정규화 테스트"""
    if not headquarter_id:
        headquarters = get_all_headquarters_with_data(period_id)
        if headquarters:
            headquarter_id = headquarters[0]
            print(f"🧪 테스트 본부 자동 선택: {headquarter_id}")
        else:
            print("❌ 테스트할 본부가 없습니다")
            return
    
    return run_headquarter_module9_evaluation(headquarter_id, period_id)

# ================================================================
# 실행 예시
# ================================================================

if __name__ == "__main__":
    print("🚀 본부 단위 모듈 9 CL별 정규화 준비 완료!")
    print("\n🔥 주요 특징:")
    print("✅ 본부 내 CL별 정규화 (무조건 정규화 적용)")
    print("✅ CL별 목표: 평균 3.5점, CL별 표준편차 차등")
    print("✅ final_evaluation_reports.score, cl_reason 업데이트")
    print("✅ 본부 단위 배치 처리")
    
    print("\n실행 명령어:")
    print("1. run_headquarter_module9_evaluation('HQ001', 4)     # 단일 본부 실행")
    print("2. run_all_headquarters_module9(4)                   # 전체 본부 일괄 실행")
    print("3. test_headquarter_module9()                        # 테스트 실행")
    
    # 자동 테스트 (필요시 주석 해제)
    test_headquarter_module9()