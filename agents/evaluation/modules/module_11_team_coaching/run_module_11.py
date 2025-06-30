# run_module_11.py
# ▶️ 실행 함수 및 테스트 코드

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from db_utils import init_database, DatabaseError
from agent import Module11TeamRiskManagementAgent, Module11AgentState
from llm_utils import parse_json_field

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ====================================
# 메인 실행 함수
# ====================================

async def run_module_11(team_id: int, period_id: int, team_evaluation_id: int) -> Module11AgentState:
    """Module 11 실행 함수"""
    
    try:
        # 데이터베이스 연결 초기화
        data_access = init_database()
        
        # Agent 초기화
        agent = Module11TeamRiskManagementAgent(data_access)
        
        # 실행
        result = await agent.execute(team_id, period_id, team_evaluation_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Module 11 실행 실패: {str(e)}")
        raise


#######################3
# 여기 밑으로는 test코드에요. test코드 필요할 것 같아서 추가

# ====================================
# 전체 통합 테스트
# ====================================

async def test_improved_module11(team_id, period_id, team_evaluation_id):
    """개선된 Module 11 전체 실행 테스트 (파라미터 직접 전달)"""
    print("🚀 개선된 Module 11 테스트 시작")
    print("=" * 80)
    print(f"📋 테스트 파라미터:")
    print(f"   팀 ID: {team_id}")
    print(f"   기간 ID: {period_id}")
    print(f"   평가 ID: {team_evaluation_id}")
    print("=" * 80)
    try:
        print("\n🔍 실행 전 DB 상태 확인...")
        before_data = check_db_before_test(team_evaluation_id)
        print("\n⚡ 개선된 Module 11 실행 중...")
        start_time = datetime.now()
        result = await run_module_11(team_id, period_id, team_evaluation_id)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"✅ 실행 완료! (소요시간: {execution_time:.1f}초)")
        print("\n🔍 실행 후 DB 상태 확인...")
        after_data = check_db_after_test(team_evaluation_id)
        print("\n📊 결과 분석...")
        analyze_test_results(result, before_data, after_data, execution_time)
        print("\n🔬 JSON 구조 검증...")
        validate_json_structures(after_data)
        print("\n📝 내용 품질 검증...")
        validate_content_quality(after_data)
        print("\n📋 최종 테스트 결과 요약...")
        summarize_test_results(result, before_data, after_data)
        return result
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ====================================
# 개별 구성요소 테스트
# ====================================

async def test_individual_components():
    """개별 구성요소별 테스트"""
    
    print("\n🧪 개별 구성요소 테스트")
    print("-" * 50)
    
    try:
        # 데이터베이스 연결 초기화
        data_access = init_database()
        
        # Agent 초기화
        agent = Module11TeamRiskManagementAgent(data_access)
        
        # 실제 데이터로 team_evaluation_id 찾기
        found_team_evaluation_id, found_period_id = find_team_evaluation_id_for_team_1()
        if not found_team_evaluation_id or not found_period_id:
            print("❌ 테스트용 데이터를 찾을 수 없습니다.")
            return None
        
        # 실제 Module11AgentState 사용
        from agent import Module11AgentState
        state = Module11AgentState(
            team_id=1,
            period_id=found_period_id,
            team_evaluation_id=found_team_evaluation_id,
            is_final=True,
            key_risks=None,
            collaboration_bias_level=None,
            performance_trend=None,
            ai_risk_result=None,
            ai_plan_result=None,
            overall_comment_result=None
        )
        
        # 실제 데이터 수집
        print("📊 실제 데이터 수집 중...")
        data = agent._collect_all_data_sequential(state)
        print(f"   ✅ 데이터 수집 완료: {len(data)}개 데이터셋")
        
        # 각 구성요소별 테스트
        print("\n🔍 협업 리스크 분석 테스트...")
        collaboration_result = await agent._analyze_collaboration_risks_with_llm_async(
            data.get('collaboration_matrix'), 
            data.get('team_members', [])
        )
        print(f"   결과: {len(collaboration_result.get('risks', []))}개 협업 리스크 식별")
        
        print("\n🔍 개인 리스크 패턴 분석 테스트...")
        individual_result = await agent._analyze_individual_risk_patterns_with_llm_async(
            data.get('individual_risks', []), 
            data.get('team_members', [])
        )
        print(f"   결과: {len(individual_result.get('risks', []))}개 개인 리스크 식별")
        
        print("\n🔍 성과 트렌드 분석 테스트...")
        performance_result = await agent._analyze_performance_trends_with_llm_async(
            data.get('team_performance', {}), 
            data.get('team_kpis', [])
        )
        print(f"   결과: {len(performance_result.get('risks', []))}개 성과 리스크 식별")
        
        return {
            'collaboration': collaboration_result,
            'individual': individual_result,
            'performance': performance_result
        }
        
    except Exception as e:
        print(f"   ❌ 개별 구성요소 테스트 실패: {str(e)}")
        return None

# ====================================
# DB 상태 확인 함수들
# ====================================

def check_db_before_test(team_evaluation_id: int) -> Dict[str, Any]:
    """테스트 실행 전 DB 상태 확인"""
    
    try:
        data_access = init_database()
        
        query = """
        SELECT ai_risk, ai_plan, overall_comment,
               ai_collaboration_matrix, ai_team_coaching, ai_team_comparison,
               updated_at
        FROM team_evaluations 
        WHERE team_evaluation_id = :team_evaluation_id
        """
        
        result = data_access.db.fetch_one(query, {'team_evaluation_id': team_evaluation_id})
        
        if result:
            print("📊 실행 전 상태:")
            print(f"   ai_risk: {'있음' if result['ai_risk'] else '없음'} ({len(str(result['ai_risk'])) if result['ai_risk'] else 0}자)")
            print(f"   ai_plan: {'있음' if result['ai_plan'] else '없음'} ({len(str(result['ai_plan'])) if result['ai_plan'] else 0}자)")
            print(f"   overall_comment: {'있음' if result['overall_comment'] else '없음'} ({len(str(result['overall_comment'])) if result['overall_comment'] else 0}자)")
            print(f"   마지막 업데이트: {result.get('updated_at', 'Unknown')}")
            
            return dict(result)
        else:
            print("❌ 테스트 대상 데이터를 찾을 수 없음")
            return {}
            
    except Exception as e:
        logger.error(f"실행 전 DB 상태 확인 실패: {str(e)}")
        return {}

def check_db_after_test(team_evaluation_id: int) -> Dict[str, Any]:
    """테스트 실행 후 DB 상태 확인"""
    
    try:
        data_access = init_database()
        
        query = """
        SELECT ai_risk, ai_plan, overall_comment,
               ai_collaboration_matrix, ai_team_coaching, ai_team_comparison,
               updated_at
        FROM team_evaluations 
        WHERE team_evaluation_id = :team_evaluation_id
        """
        
        result = data_access.db.fetch_one(query, {'team_evaluation_id': team_evaluation_id})
        
        if result:
            print("📊 실행 후 상태:")
            print(f"   ai_risk: {'생성됨' if result['ai_risk'] else '생성 실패'} ({len(str(result['ai_risk'])) if result['ai_risk'] else 0}자)")
            print(f"   ai_plan: {'생성됨' if result['ai_plan'] else '생성 실패'} ({len(str(result['ai_plan'])) if result['ai_plan'] else 0}자)")
            print(f"   overall_comment: {'생성됨' if result['overall_comment'] else '생성 실패'} ({len(str(result['overall_comment'])) if result['overall_comment'] else 0}자)")
            print(f"   마지막 업데이트: {result.get('updated_at', 'Unknown')}")
            
            return dict(result)
        else:
            print("❌ 테스트 결과 데이터를 찾을 수 없음")
            return {}
            
    except Exception as e:
        logger.error(f"실행 후 DB 상태 확인 실패: {str(e)}")
        return {}

# ====================================
# 결과 분석 함수들
# ====================================

def analyze_test_results(result: Any, before_data: Dict, after_data: Dict, execution_time: float):
    """테스트 결과 종합 분석"""
    
    print("🎯 테스트 결과 분석:")
    print(f"   실행 시간: {execution_time:.1f}초")
    print(f"   연말 평가 여부: {result.is_final if result else 'Unknown'}")
    print(f"   식별된 리스크 수: {len(result.key_risks) if result and result.key_risks else 0}개")
    print(f"   협업 편향 수준: {result.collaboration_bias_level if result else 'Unknown'}")
    
    # 데이터 생성 여부 확인
    module11_fields = ['ai_risk', 'ai_plan', 'overall_comment']
    
    for field in module11_fields:
        before_len = len(str(before_data.get(field, ''))) if before_data.get(field) else 0
        after_len = len(str(after_data.get(field, ''))) if after_data.get(field) else 0
        
        if after_len > before_len:
            print(f"   ✅ {field}: 성공 ({before_len} → {after_len}자, +{after_len - before_len})")
        elif after_len == before_len and after_len > 0:
            print(f"   ⚠️ {field}: 변경 없음 ({after_len}자)")
        else:
            print(f"   ❌ {field}: 생성 실패 ({before_len} → {after_len}자)")

def validate_json_structures(after_data: Dict[str, Any]):
    """JSON 구조 검증"""
    
    print("🔬 JSON 구조 검증:")
    
    # ai_risk JSON 구조 검증
    if after_data.get('ai_risk'):
        try:
            risk_data = json.loads(after_data['ai_risk'])
            
            # 필수 구조 확인
            if 'risk_analysis' in risk_data:
                major_risks = risk_data['risk_analysis'].get('major_risks', [])
                print(f"   ✅ ai_risk: 올바른 JSON 구조 (주요 리스크 {len(major_risks)}개)")
                
                # 첫 번째 리스크 구조 확인
                if major_risks:
                    first_risk = major_risks[0]
                    required_fields = ['risk_name', 'severity', 'description', 'causes', 'impacts', 'strategies']
                    missing_fields = [f for f in required_fields if f not in first_risk]
                    
                    if not missing_fields:
                        print(f"      ✅ 리스크 필드 완전성: 모든 필수 필드 존재")
                    else:
                        print(f"      ⚠️ 리스크 필드 누락: {missing_fields}")
                else:
                    print(f"      ⚠️ 주요 리스크가 비어있음")
            else:
                print(f"   ❌ ai_risk: 잘못된 JSON 구조 (risk_analysis 키 없음)")
                
        except json.JSONDecodeError as e:
            print(f"   ❌ ai_risk: JSON 파싱 실패 - {str(e)}")
    else:
        print(f"   ❌ ai_risk: 데이터 없음")
    
    # ai_plan JSON 구조 검증
    if after_data.get('ai_plan'):
        try:
            plan_data = json.loads(after_data['ai_plan'])
            
            if 'annual_plans' in plan_data:
                annual_plans = plan_data['annual_plans']
                print(f"   ✅ ai_plan: 올바른 JSON 구조 (연간 계획 {len(annual_plans)}개)")
                
                # 계획 구조 확인
                if annual_plans:
                    first_plan = annual_plans[0]
                    if 'personnel_strategies' in first_plan and 'collaboration_improvements' in first_plan:
                        personnel_count = len(first_plan['personnel_strategies'])
                        collaboration_count = len(first_plan['collaboration_improvements'])
                        print(f"      ✅ 계획 구성: 인사전략 {personnel_count}개, 협업개선 {collaboration_count}개")
                    else:
                        print(f"      ⚠️ 계획 구조 불완전")
                else:
                    print(f"      ⚠️ 연간 계획이 비어있음")
            else:
                print(f"   ❌ ai_plan: 잘못된 JSON 구조 (annual_plans 키 없음)")
                
        except json.JSONDecodeError as e:
            print(f"   ❌ ai_plan: JSON 파싱 실패 - {str(e)}")
    else:
        print(f"   ❌ ai_plan: 데이터 없음")

def validate_content_quality(after_data: Dict[str, Any]):
    """내용 품질 검증"""
    
    print("📝 내용 품질 검증:")
    
    # ai_risk 내용 품질 확인
    if after_data.get('ai_risk'):
        risk_content = after_data['ai_risk']
        
        # 기본 품질 지표
        quality_indicators = {
            '구체적 수치 포함': any(char.isdigit() for char in risk_content),
            '한국어 분석': '팀' in risk_content or '성과' in risk_content,
            '실행 가능한 전략': '전략' in risk_content or '개선' in risk_content,
            '충분한 내용량': len(risk_content) > 500
        }
        
        print("   📋 ai_risk 품질:")
        for indicator, passed in quality_indicators.items():
            status = "✅" if passed else "❌"
            print(f"      {status} {indicator}")
    
    # overall_comment 내용 품질 확인
    if after_data.get('overall_comment'):
        comment_content = after_data['overall_comment']
        
        quality_indicators = {
            '구조화된 형식': '[' in comment_content and ']' in comment_content,
            '구체적 수치 포함': any(char.isdigit() for char in comment_content),
            '실행 가능한 내용': '전략' in comment_content or '계획' in comment_content,
            '충분한 분량': len(comment_content) > 300
        }
        
        print("   💬 overall_comment 품질:")
        for indicator, passed in quality_indicators.items():
            status = "✅" if passed else "❌"
            print(f"      {status} {indicator}")

def summarize_test_results(result: Any, before_data: Dict, after_data: Dict):
    """최종 테스트 결과 요약"""
    
    print("=" * 80)
    print("📋 최종 테스트 결과 요약")
    print("=" * 80)
    
    # 전체 성공률 계산
    total_fields = 3  # ai_risk, ai_plan, overall_comment
    successful_fields = 0
    
    if after_data.get('ai_risk'):
        successful_fields += 1
    if after_data.get('ai_plan'):
        successful_fields += 1
    if after_data.get('overall_comment'):
        successful_fields += 1
    
    success_rate = (successful_fields / total_fields) * 100
    
    print(f"🎯 전체 성공률: {success_rate:.1f}% ({successful_fields}/{total_fields})")
    
    # 개선 효과 분석
    improvements = []
    
    # 내용 길이 개선
    for field in ['ai_risk', 'ai_plan', 'overall_comment']:
        before_len = len(str(before_data.get(field, ''))) if before_data.get(field) else 0
        after_len = len(str(after_data.get(field, ''))) if after_data.get(field) else 0
        
        if after_len > before_len:
            improvement = after_len - before_len
            improvements.append(f"{field}: +{improvement}자")
    
    if improvements:
        print(f"📈 개선 효과: {', '.join(improvements)}")
    
    # JSON 구조 개선 여부
    json_improvements = []
    
    if after_data.get('ai_risk'):
        try:
            risk_data = json.loads(after_data['ai_risk'])
            if 'risk_analysis' in risk_data:
                json_improvements.append("ai_risk 구조화")
        except:
            pass
    
    if after_data.get('ai_plan'):
        try:
            plan_data = json.loads(after_data['ai_plan'])
            if 'annual_plans' in plan_data:
                json_improvements.append("ai_plan 구조화")
        except:
            pass
    
    if json_improvements:
        print(f"🔧 구조 개선: {', '.join(json_improvements)}")
    
    # 권장 사항
    print("\n💡 권장 사항:")
    if success_rate < 100:
        print("   - 실패한 필드에 대한 오류 로그 확인 필요")
        print("   - LLM 프롬프트 추가 최적화 고려")
    
    if success_rate >= 80:
        print("   ✅ 전반적으로 우수한 성능")
        print("   - 프로덕션 환경 적용 고려 가능")
    elif success_rate >= 60:
        print("   ⚠️ 부분적 성공 - 추가 개선 필요")
        print("   - 실패 원인 분석 후 재테스트 권장")
    else:
        print("   ❌ 성능 개선 필요")
        print("   - 기본 설정 및 환경 재점검 필요")
    
    print("=" * 80)

# ====================================
# 전체 테스트 실행
# ====================================

async def run_all_tests(team_id, period_id, team_evaluation_id):
    """모든 테스트 실행 (파라미터 직접 전달)"""
    print("🚀 Module 11 개선 버전 종합 테스트")
    print("=" * 80)
    # 1. 전체 통합 테스트
    print("\n1️⃣ 전체 통합 테스트")
    integration_result = await test_improved_module11(team_id, period_id, team_evaluation_id)
    # 2. 개별 구성요소 테스트
    print("\n2️⃣ 개별 구성요소 테스트")
    component_results = await test_individual_components()
    # 3. 성능 비교 테스트 (옵션)
    print("\n3️⃣ 성능 비교 테스트 (기존 vs 개선)")
    # 여러 번 실행하여 평균 성능 측정 가능
    print("\n🎉 모든 테스트 완료!")
    return {
        'integration': integration_result,
        'components': component_results
    }

# ====================================
# 실행 테스트 파라미터 (team_id=1로 수정)
# ====================================

# team_id=1로 테스트 (상수 정의 제거)
# TEST_TEAM_ID = 1
# TEST_PERIOD_ID = 4
# TEST_TEAM_EVALUATION_ID = None

# ====================================
# team_id=1용 데이터 검증 함수
# ====================================

def find_team_evaluation_id_for_team_1():
    """team_id=1에 해당하는 team_evaluation_id 찾기"""
    try:
        data_access = init_database()
        query = """
        SELECT team_evaluation_id, te.period_id, 
               p.period_name, p.year,
               te.average_achievement_rate
        FROM team_evaluations te
        JOIN periods p ON te.period_id = p.period_id
        WHERE te.team_id = :team_id
        ORDER BY p.year DESC, p.order_in_year DESC
        LIMIT 5
        """
        results = data_access.db.fetch_all(query, {'team_id': 1})
        if results:
            print("🔍 team_id=1의 평가 데이터:")
            for i, row in enumerate(results, 1):
                print(f"   {i}. team_evaluation_id: {row['team_evaluation_id']}")
                print(f"      period_id: {row['period_id']} ({row['period_name']})")
                print(f"      year: {row['year']}")
                print(f"      달성률: {row['average_achievement_rate'] or 'N/A'}")
                print()
            # 가장 최근 데이터 반환
            return results[0]['team_evaluation_id'], results[0]['period_id']
        else:
            print("❌ team_id=1의 평가 데이터가 없습니다.")
            return None, None
    except Exception as e:
        print(f"❌ 데이터 조회 실패: {str(e)}")
        return None, None

# ====================================
# 메인 실행 부분 (수정)
# ====================================

if __name__ == "__main__":
    # team_id=1용 데이터 찾기
    print("🔍 team_id=1용 데이터 검색 중...")
    found_team_evaluation_id, found_period_id = find_team_evaluation_id_for_team_1()
    if found_team_evaluation_id is not None and found_period_id is not None:
        # 찾은 데이터로 실행
        print(f"✅ 데이터 발견: team_evaluation_id={found_team_evaluation_id}, period_id={found_period_id}")
        print("🚀 Module 11 실행 시작")
        result = asyncio.run(run_module_11(1, found_period_id, found_team_evaluation_id))
        print(f"✅ 실행 완료: {result}")
        # 통합 테스트도 실행하고 싶으면 아래 주석 해제
        # test_result = asyncio.run(test_improved_module11(1, found_period_id, found_team_evaluation_id))
        # 전체 테스트 실행하고 싶으면 아래 주석 해제
        # all_test_results = asyncio.run(run_all_tests(1, found_period_id, found_team_evaluation_id))
    else:
        print("❌ team_id=1에 해당하는 데이터를 찾을 수 없습니다.")
        print("다른 team_id를 사용하거나 데이터를 확인해주세요.")