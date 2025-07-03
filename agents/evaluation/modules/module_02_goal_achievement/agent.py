# ================================================================
# agent_module2.py - 모듈 2 LangGraph 에이전트 및 상태 관리
# ================================================================

import logging
from typing import Dict, List, Optional, Any, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

from agents.evaluation.modules.module_02_goal_achievement.db_utils import *
from agents.evaluation.modules.module_02_goal_achievement.calculation_utils import *
from agents.evaluation.modules.module_02_goal_achievement.llm_utils import *
from agents.evaluation.modules.module_02_goal_achievement.comment_generator import *

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 상태 정의
# ================================================================

class Module2State(TypedDict):
    """경량 State - 우리가 상의한 DB 기반 전달 방식"""
    # 기본 정보
    report_type: Literal["quarterly", "annual"]
    team_id: int
    period_id: int
    
    # 타겟 ID들
    target_task_ids: List[int]
    target_team_kpi_ids: List[int]
    
    # 처리 결과 추적용 (DB ID만 저장)
    updated_task_ids: Optional[List[int]]
    updated_team_kpi_ids: Optional[List[int]]
    feedback_report_ids: Optional[List[int]]
    team_evaluation_id: Optional[int]
    final_evaluation_report_ids: Optional[List[int]]
    
    # 특별 전달 데이터 (서브모듈 간 필요시만)
    team_context_guide: Optional[Dict]

# ================================================================
# 에러 처리 클래스
# ================================================================

class DataIntegrityError(Exception):
    pass

# ================================================================
# 서브모듈 함수들
# ================================================================

def data_collection_submodule(state: Module2State) -> Module2State:
    """데이터 수집 서브모듈"""
    print(f"   📋 데이터 수집 중...")
    
    # team_evaluation_id 확인/생성
    team_evaluation_id = fetch_team_evaluation_id(state['team_id'], state['period_id'])
    if not team_evaluation_id:
        raise DataIntegrityError(f"team_evaluation_id not found for team {state['team_id']}, period {state['period_id']}")
    
    state['team_evaluation_id'] = team_evaluation_id
    
    # evaluation_type 확인/설정
    for kpi_id in state['target_team_kpi_ids']:
        evaluation_type = check_evaluation_type(kpi_id)
        print(f"      • KPI {kpi_id}: {evaluation_type} 평가")
    
    print(f"   ✅ 데이터 수집 완료")
    return state

def achievement_and_grade_calculation_submodule(state: Module2State) -> Module2State:
    """달성률+등급 계산 서브모듈 (통합) - 우리가 상의한 배치 처리"""
    print(f"   🎯 달성률 및 등급 계산 중...")
    
    updated_task_ids = []
    batch_data = []
    
    # 배치용 데이터 준비
    for task_id in state['target_task_ids']:
        task_data = fetch_cumulative_task_data(task_id, state['period_id'])
        if not task_data:
            continue
            
        batch_data.append({
            "task_id": task_id,
            "task_summary_id": task_data.get('task_summary_id'),
            "target_level": task_data.get('target_level', ''),
            "cumulative_performance": task_data.get('cumulative_performance', ''),
            "cumulative_summary": task_data.get('cumulative_task_summary', ''),
            "kpi_data": fetch_team_kpi_data(task_data.get('team_kpi_id') or 0)
        })
    
    # 배치 처리 (15개씩)
    batch_size = 15
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        results = batch_calculate_achievement_and_grades(batch, state['report_type'] == "annual")
        
        # 결과 저장
        for task_data, result in zip(batch, results):
            task_summary_id = task_data['task_summary_id']
            if not task_summary_id:
                continue
                
            update_data = {
                "ai_achievement_rate": int(result['achievement_rate'])
            }
            
            # 연말인 경우 등급도 저장
            if state['report_type'] == "annual" and result.get('grade'):
                update_data["ai_assessed_grade"] = result['grade']
            
            if update_task_summary(task_summary_id, update_data):
                updated_task_ids.append(task_data['task_id'])
    
    state['updated_task_ids'] = updated_task_ids
    print(f"   ✅ 달성률 계산 완료: {len(updated_task_ids)}개 Task 업데이트")
    return state

def contribution_calculation_submodule(state: Module2State) -> Module2State:
    """기여도 계산 서브모듈 - 우리가 상의한 하이브리드 방식"""
    print(f"   ⚖️ 기여도 계산 중...")
    
    updated_task_ids = []
    kpi_contributions_by_emp = {}  # {emp_no: total_score} - 하이브리드 3단계 결과
    
    # KPI별로 처리
    for kpi_id in state['target_team_kpi_ids']:
        evaluation_type = check_evaluation_type(kpi_id)
        kpi_data = fetch_team_kpi_data(kpi_id)
        
        if evaluation_type == "quantitative":
            # 정량 평가: 개인성과/팀전체성과 × 100
            contributions = calculate_quantitative_contributions(kpi_id, state['period_id'])
        else:
            # 정성 평가: LLM 기반 상대 평가
            contributions = calculate_qualitative_contributions(kpi_id, state['period_id'], kpi_data)
        
        # 🔧 추가: KPI별 기여도 합계 검증
        total_contribution = sum(contributions.values())
        if abs(total_contribution - 100) > 10:
            raise DataIntegrityError(f"KPI {kpi_id}: 기여도 합계 {total_contribution:.1f}%가 100%에서 벗어남")
        print(f"      ✅ KPI {kpi_id} 기여도 합계: {total_contribution:.1f}%")
        
        # 하이브리드 1단계: 참여자 수 보정
        kpi_tasks = fetch_kpi_tasks(kpi_id, state['period_id'])
        participants_count = len(set(task['emp_no'] for task in kpi_tasks))
        
        print(f"      • KPI {kpi_id}: {evaluation_type} 평가, 참여자 {participants_count}명")
        
        for emp_no, contribution_rate in contributions.items():
            # 1단계: 참여자 수 보정
            adjusted_score = contribution_rate * participants_count
            
            # 2단계: KPI 비중 적용
            kpi_weight = kpi_data.get('weight', 0) / 100.0
            weighted_score = adjusted_score * kpi_weight
            
            if emp_no not in kpi_contributions_by_emp:
                kpi_contributions_by_emp[emp_no] = 0
            kpi_contributions_by_emp[emp_no] += weighted_score
            
            print(f"        - {emp_no}: 원래 {contribution_rate:.1f}% → 보정 {adjusted_score:.1f} → 가중 {weighted_score:.1f}")
        
        # Task별 기여도 업데이트 (원래 KPI별 기여도 저장)
        for task in kpi_tasks:
            task_data = fetch_cumulative_task_data(task['task_id'], state['period_id'])
            if not task_data:
                continue
                
            emp_contribution = contributions.get(task['emp_no'], 0)
            
            update_data = {
                "ai_contribution_score": int(emp_contribution)  # KPI별 원래 기여도
            }
            
            if update_task_summary(task_data['task_summary_id'], update_data):
                updated_task_ids.append(task['task_id'])
    
    # 하이브리드 3단계: 팀 내 % 기여도 변환
    total_team_score = sum(kpi_contributions_by_emp.values())
    final_contributions = {}
    
    if total_team_score > 0:
        for emp_no in kpi_contributions_by_emp:
            percentage = (kpi_contributions_by_emp[emp_no] / total_team_score) * 100
            final_contributions[emp_no] = round(percentage, 2)
            print(f"      • {emp_no} 최종 기여도: {percentage:.1f}%")
    else:
        # 팀 점수가 0인 경우 동등 분배
        emp_count = len(kpi_contributions_by_emp)
        if emp_count > 0:
            equal_share = 100.0 / emp_count
            for emp_no in kpi_contributions_by_emp:
                final_contributions[emp_no] = round(equal_share, 2)
    
    # 최종 기여도를 feedback_reports 또는 final_evaluation_reports에 저장
    save_final_contributions_to_db(state, final_contributions)
    
    # 디버깅: 하이브리드 계산 과정 시각화
    debug_contribution_calculation(state)
    
    state['updated_task_ids'] = list(set((state['updated_task_ids'] or []) + updated_task_ids))
    print(f"   ✅ 기여도 계산 완료: {len(updated_task_ids)}개 Task 업데이트, {len(final_contributions)}명 최종 기여도 저장")
    return state

def save_final_contributions_to_db(state: Module2State, final_contributions: Dict[str, float]):
    """최종 기여도를 DB에 저장"""
    team_members = fetch_team_members(state['team_id'])
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
            
        emp_no = member['emp_no']
        final_contribution = final_contributions.get(emp_no, 0)
        
        if state['report_type'] == "quarterly":
            # 분기별: feedback_reports에 저장
            save_feedback_report(
                emp_no, 
                state['team_evaluation_id'] or 0,
                {"contribution_rate": int(final_contribution)}  # 기존 컬럼명 사용
            )
        else:
            # 연말: final_evaluation_reports에 저장
            save_final_evaluation_report(
                emp_no,
                state['team_evaluation_id'] or 0,
                {"contribution_rate": int(final_contribution)}  # 기존 컬럼명 사용
            )

def debug_contribution_calculation(state: Module2State):
    """기여도 계산 과정 디버깅 - 하이브리드 방식 검증"""
    print(f"\n🔍 기여도 계산 과정 디버깅")
    print(f"{'='*50}")
    
    # 1단계: KPI별 원래 기여도 수집
    kpi_contributions = {}
    for kpi_id in state['target_team_kpi_ids']:
        evaluation_type = check_evaluation_type(kpi_id)
        kpi_data = fetch_team_kpi_data(kpi_id)
        kpi_tasks = fetch_kpi_tasks(kpi_id, state['period_id'])
        participants_count = len(set(task['emp_no'] for task in kpi_tasks))
        
        if evaluation_type == "quantitative":
            contributions = calculate_quantitative_contributions(kpi_id, state['period_id'])
        else:
            contributions = calculate_qualitative_contributions(kpi_id, state['period_id'], kpi_data)
        
        kpi_contributions[kpi_id] = {
            'kpi_name': kpi_data.get('kpi_name', f'KPI{kpi_id}'),
            'weight': kpi_data.get('weight', 0),
            'participants_count': participants_count,
            'contributions': contributions
        }
    
    # 2단계: 하이브리드 계산 과정 시각화
    print(f"📊 KPI별 기여도 분석:")
    for kpi_id, kpi_info in kpi_contributions.items():
        print(f"\n🎯 {kpi_info['kpi_name']} (비중: {kpi_info['weight']}%, 참여자: {kpi_info['participants_count']}명)")
        print(f"   원래 기여도 → 참여자수 보정 → KPI 비중 적용")
        print(f"   {'─' * 50}")
        
        for emp_no, original_rate in kpi_info['contributions'].items():
            # 1단계: 참여자 수 보정
            adjusted = original_rate * kpi_info['participants_count']
            # 2단계: KPI 비중 적용
            weighted = adjusted * (kpi_info['weight'] / 100.0)
            
            print(f"   {emp_no}: {original_rate:5.1f}% → {adjusted:6.1f} → {weighted:6.1f}")
    
    # 3단계: 개인별 종합 점수 계산
    print(f"\n📈 개인별 종합 점수 (하이브리드 1-2단계 결과):")
    emp_total_scores = {}
    
    for kpi_id, kpi_info in kpi_contributions.items():
        for emp_no, original_rate in kpi_info['contributions'].items():
            if emp_no not in emp_total_scores:
                emp_total_scores[emp_no] = 0
            
            adjusted = original_rate * kpi_info['participants_count']
            weighted = adjusted * (kpi_info['weight'] / 100.0)
            emp_total_scores[emp_no] += weighted
    
    for emp_no, total_score in sorted(emp_total_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {emp_no}: {total_score:.1f}점")
    
    # 4단계: 팀 내 % 기여도 변환
    total_team_score = sum(emp_total_scores.values())
    print(f"\n🏆 최종 기여도 (하이브리드 3단계 결과):")
    print(f"   팀 전체 점수: {total_team_score:.1f}")
    print(f"   {'─' * 30}")
    
    for emp_no, total_score in sorted(emp_total_scores.items(), key=lambda x: x[1], reverse=True):
        final_percentage = (total_score / total_team_score) * 100 if total_team_score > 0 else 0
        print(f"   {emp_no}: {final_percentage:.1f}% ({total_score:.1f}점)")
    
    print(f"{'='*50}")

def team_analysis_submodule(state: Module2State) -> Module2State:
    """팀 목표 분석 서브모듈 - 우리가 상의한 LLM 기반"""
    print(f"   🏢 팀 목표 분석 중...")
    
    updated_kpi_ids = []
    kpi_rates = []
    
    # 정량 평가 KPI들 처리 (LLM으로 팀 KPI 달성률 계산)
    for kpi_id in state['target_team_kpi_ids']:
        evaluation_type = check_evaluation_type(kpi_id)
        
        if evaluation_type == "quantitative":
            # 정량 KPI도 LLM이 종합 판단
            kpi_data = fetch_team_kpi_data(kpi_id)
            kpi_rate = calculate_team_kpi_achievement_rate(kpi_id, state['period_id'], kpi_data)
            
            update_data = {
                "ai_kpi_progress_rate": int(kpi_rate['rate']),
                "ai_kpi_analysis_comment": kpi_rate['comment']
            }
            
            if update_team_kpi(kpi_id, update_data):
                updated_kpi_ids.append(kpi_id)
                kpi_rates.append(kpi_rate['rate'])
        else:
            # 정성 KPI는 이미 서브모듈 3에서 처리됨
            kpi_data = fetch_team_kpi_data(kpi_id)
            if kpi_data.get('ai_kpi_progress_rate') is not None:
                kpi_rates.append(kpi_data['ai_kpi_progress_rate'])
    
    # 팀 전체 평균 달성률 계산 (KPI 비중 고려)
    team_average_rate = calculate_team_average_achievement_rate(state['target_team_kpi_ids'])
    
    # team_evaluations 업데이트
    team_eval_data = {
        "average_achievement_rate": int(team_average_rate)
    }
    
    # 연말인 경우 전년 대비 성장률 계산 시도
    if state['report_type'] == "annual":
        yoy_growth = calculate_year_over_year_growth(state['team_id'], state['period_id'], team_average_rate)
        if yoy_growth is not None:
            team_eval_data["year_over_year_growth"] = int(yoy_growth)
    
    if update_team_evaluations(state['team_evaluation_id'] or 0, team_eval_data):
        print(f"      • 팀 평균 달성률: {team_average_rate:.1f}%")
    
    state['updated_team_kpi_ids'] = updated_kpi_ids
    print(f"   ✅ 팀 분석 완료: {len(updated_kpi_ids)}개 KPI 업데이트")
    return state

def comment_generation_submodule(state: Module2State) -> Module2State:
    """코멘트 생성 서브모듈 - 통합 시스템 사용"""
    print(f"   📝 코멘트 생성 중...")
    
    # 팀 일관성 가이드 생성
    team_context_guide = generate_team_consistency_guide(state['team_id'], state['period_id'])
    state['team_context_guide'] = team_context_guide
    
    # 통합 코멘트 생성기 사용
    generate_task_comments_unified(state)
    generate_individual_summary_comments_unified(state)
    generate_team_overall_comment_unified(state)
    
    print(f"   ✅ 코멘트 생성 완료")
    return state

def generate_task_comments_unified(state: Module2State):
    """Task별 코멘트 생성 (통합 시스템)"""
    period_type = "annual" if state['report_type'] == "annual" else "quarterly"
    
    for task_id in state['target_task_ids']:
        task_data = fetch_cumulative_task_data(task_id, state['period_id'])
        if not task_data:
            continue
        
        # 통합 코멘트 생성기 사용
        generator = CommentGenerator("task", period_type, state['team_context_guide'])
        comment = generator.generate(task_data)
        
        if task_data.get('task_summary_id'):
            update_task_summary(task_data['task_summary_id'], {
                "ai_analysis_comment_task": comment
            })

def generate_individual_summary_comments_unified(state: Module2State):
    """개인 종합 코멘트 생성 (통합 시스템)"""
    if 'feedback_report_ids' not in state or state['feedback_report_ids'] is None:
        state['feedback_report_ids'] = []
    team_members = fetch_team_members(state['team_id'])
    period_type = "annual" if state['report_type'] == "annual" else "quarterly"
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
        
        # 개인 Task 데이터 수집
        individual_tasks = []
        for task_id in state['target_task_ids']:
            task_data = fetch_cumulative_task_data(task_id, state['period_id'])
            if task_data and task_data.get('emp_no') == member['emp_no']:
                individual_tasks.append(task_data)
        
        if not individual_tasks:
            continue
        
        # 통합 코멘트 생성기 사용
        generator = CommentGenerator("individual", period_type, state['team_context_guide'])
        comment = generator.generate({
            **member,
            "tasks": individual_tasks
        })
        
        # 분기별/연말별 저장
        if state['report_type'] == "quarterly":
            feedback_report_id = save_feedback_report(
                member['emp_no'], 
                state['team_evaluation_id'] or 0,
                {"ai_overall_contribution_summary_comment": comment}
            )
            if state['feedback_report_ids'] is None:
                state['feedback_report_ids'] = []
            state['feedback_report_ids'].append(feedback_report_id)
        else:  # annual
            final_report_id = save_final_evaluation_report(
                member['emp_no'],
                state['team_evaluation_id'] or 0, 
                {"ai_annual_performance_summary_comment": comment}
            )
            if state['final_evaluation_report_ids'] is None:
                state['final_evaluation_report_ids'] = []
            state['final_evaluation_report_ids'].append(final_report_id)

def generate_team_overall_comment_unified(state: Module2State):
    """팀 전체 분석 코멘트 생성 (통합 시스템)"""
    if 'final_evaluation_report_ids' not in state or state['final_evaluation_report_ids'] is None:
        state['final_evaluation_report_ids'] = []
    # 팀 KPI 데이터 수집
    team_kpis_data = []
    for kpi_id in state['target_team_kpi_ids']:
        kpi_data = fetch_team_kpi_data(kpi_id)
        if kpi_data:
            team_kpis_data.append(kpi_data)
    
    period_type = "annual" if state['report_type'] == "annual" else "quarterly"
    
    # 통합 코멘트 생성기 사용
    generator = CommentGenerator("team", period_type, state['team_context_guide'])
    comment = generator.generate({
        "kpis": team_kpis_data,
        "team_context": state['team_context_guide'].get('team_context', '') if state['team_context_guide'] else '',
        "performance_level": state['team_context_guide'].get('performance_level', '') if state['team_context_guide'] else ''
    })
    
    # team_evaluations 업데이트
    update_team_evaluations(state['team_evaluation_id'] or 0, {
        "ai_team_overall_analysis_comment": comment
    })

def db_update_submodule(state: Module2State) -> Module2State:
    """최종 DB 업데이트 서브모듈 - 트랜잭션 처리"""
    print(f"   💾 최종 DB 업데이트 중...")
    
    try:
        with engine.begin() as transaction:
            # 이미 각 서브모듈에서 업데이트했으므로 최종 검증만 수행
            
            # 1. 분기별 추가 업데이트 (ranking, cumulative 데이터)
            if state['report_type'] == "quarterly":
                update_quarterly_specific_data(state)
            
            # 2. 연말 추가 업데이트 (final_evaluation_reports 추가 필드)
            elif state['report_type'] == "annual":
                update_annual_specific_data(state)
            
            # 3. 업데이트 결과 검증
            validation_result = validate_final_update_results(state)
            
            if not validation_result['success']:
                raise DataIntegrityError(f"Final validation failed: {validation_result['errors']}")
            
            # 4. 업데이트 통계 로깅
            updated_tasks = len(state['updated_task_ids'] or [])
            updated_kpis = len(state['updated_team_kpi_ids'] or [])
            updated_feedback_reports = len(state['feedback_report_ids'] or [])
            updated_final_reports = len(state['final_evaluation_report_ids'] or [])
            
            print(f"      • Task 업데이트: {updated_tasks}개")
            print(f"      • KPI 업데이트: {updated_kpis}개")
            print(f"      • 피드백 리포트: {updated_feedback_reports}개")
            print(f"      • 최종 리포트: {updated_final_reports}개")
            
            # 5. 최종 상태 로깅
            if state['report_type'] == "quarterly":
                print(f"      • 분기 평가 완료")
            else:
                print(f"      • 연말 평가 완료")
                
            return state
                
    except Exception as e:
        print(f"   ❌ 최종 DB 업데이트 실패: {e}")
        raise

def update_quarterly_specific_data(state: Module2State):
    """분기별 전용 데이터 업데이트 - 개인 달성률 기반 순위 매기기"""
    print(f"      📊 분기별 전용 데이터 업데이트 중...")
    
    # 1. 팀 내 개인별 달성률 기반 순위 계산 및 업데이트
    team_ranking_result = calculate_team_ranking(state)
    
    # 2. 순위 결과를 feedback_reports에 저장 (기여도는 이미 하이브리드 3단계에서 저장됨)
    update_team_ranking_to_feedback_reports(state, team_ranking_result)
    
    print(f"      ✅ 분기별 순위 업데이트 완료: {len(team_ranking_result)}명")
    print(f"      📊 팀 내 달성률 순위:")
    for i, member in enumerate(team_ranking_result):
        print(f"        {i+1}위: {member['emp_name']}({member['emp_no']}) - {member['avg_achievement_rate']:.1f}%")

def calculate_team_ranking(state: Module2State) -> List[Dict]:
    """팀 내 개인별 달성률 기반 순위 계산"""
    print(f"        🏆 팀 내 순위 계산 중...")
    
    team_members = fetch_team_members(state['team_id'])
    member_achievements = []
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
            
        # 개인별 Task 수집
        individual_tasks = []
        for task_id in state['target_task_ids']:
            task_data = fetch_cumulative_task_data(task_id, state['period_id'])
            if task_data and task_data.get('emp_no') == member['emp_no']:
                individual_tasks.append(task_data)
        
        if individual_tasks:
            # 가중평균 달성률 계산 (기여도는 이미 하이브리드 3단계에서 계산되어 저장됨)
            result = calculate_individual_weighted_achievement_rate(individual_tasks)
            
            # 계산 과정 상세 로깅
            print(f"          📈 {member['emp_name']}({member['emp_no']}) 달성률 계산:")
            total_weighted_score = 0
            for task in individual_tasks:
                task_name = task.get('task_name', f'Task{task.get("task_id")}')
                task_weight = task.get('weight', 0)
                task_achievement = task.get('ai_achievement_rate', 0)
                weighted_score = task_achievement * task_weight
                total_weighted_score += weighted_score
                print(f"            • {task_name}: {task_achievement}% × {task_weight} = {weighted_score}")
            
            print(f"            = {result['achievement_rate']:.1f}% (총 가중점수: {total_weighted_score}, 총 가중치: {result['total_weight']})")
            
            member_achievements.append({
                'emp_no': member['emp_no'],
                'emp_name': member['emp_name'],
                'position': member.get('position', ''),
                'cl': member.get('cl', ''),
                'avg_achievement_rate': result['achievement_rate'],
                # 'avg_contribution_rate': result['contribution_rate'],  # 삭제 - 하이브리드 3단계에서 이미 계산됨
                'task_count': len(individual_tasks),
                'total_weight': result['total_weight'],
                'total_weighted_score': total_weighted_score
            })
        else:
            print(f"          ⚠️  {member['emp_name']}({member['emp_no']}): 참여 Task 없음")
    
    # 달성률 기준 내림차순 정렬 (높은 달성률이 1위)
    member_achievements.sort(key=lambda x: x['avg_achievement_rate'], reverse=True)
    
    # 동점자 처리 (같은 달성률인 경우 가중점수로 재정렬)
    for i in range(len(member_achievements) - 1):
        if member_achievements[i]['avg_achievement_rate'] == member_achievements[i + 1]['avg_achievement_rate']:
            # 동점자인 경우 가중점수로 재정렬
            if member_achievements[i]['total_weighted_score'] < member_achievements[i + 1]['total_weighted_score']:
                member_achievements[i], member_achievements[i + 1] = member_achievements[i + 1], member_achievements[i]
    
    return member_achievements

def update_team_ranking_to_feedback_reports(state: Module2State, team_ranking: List[Dict]):
    """팀 순위 결과를 feedback_reports에 저장"""
    print(f"        💾 순위 결과를 feedback_reports에 저장 중...")
    
    updated_count = 0
    
    for i, member_data in enumerate(team_ranking):
        ranking = i + 1
        
        # feedback_reports 업데이트 데이터 - 기여도는 이미 올바르게 저장되어 있으므로 제외
        feedback_data = {
            'ranking': ranking,  # 팀 내 순위 (1, 2, 3, ...)
            'ai_achievement_rate': int(member_data['avg_achievement_rate']),  # 가중평균 달성률
            # 'contribution_rate': int(member_data['avg_contribution_rate'])  # 삭제 - 하이브리드 3단계 기여도가 이미 저장됨
        }
        
        # 기존 feedback_report 업데이트 또는 새로 생성
        feedback_report_id = save_feedback_report(
            member_data['emp_no'],
            state['team_evaluation_id'] or 0,
            feedback_data
        )
        
        updated_count += 1
        
        # 순위 저장 결과 로깅
        print(f"          {ranking}위: {member_data['emp_name']}({member_data['emp_no']}) - {member_data['avg_achievement_rate']:.1f}% → feedback_report_id: {feedback_report_id}")
    
    print(f"        ✅ {updated_count}명의 순위 정보 저장 완료")

def update_annual_specific_data(state: Module2State):
    """연말 전용 데이터 업데이트 - Task Weight 기반 가중평균"""
    print(f"      📊 연말 전용 데이터 업데이트 중...")
    
    team_members = fetch_team_members(state['team_id'])
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
            
        # 개인별 Task 수집
        individual_tasks = []
        for task_id in state['target_task_ids']:
            task_data = fetch_cumulative_task_data(task_id, state['period_id'])
            if task_data and task_data.get('emp_no') == member['emp_no']:
                individual_tasks.append(task_data)
        
        if individual_tasks:
            # 가중평균 계산
            result = calculate_individual_weighted_achievement_rate(individual_tasks)
            
            # 계산 과정 로깅
            print(f"        📈 {member['emp_name']}({member['emp_no']}) 연간 가중평균 계산:")
            for task in individual_tasks:
                task_name = task.get('task_name', f'Task{task.get("task_id")}')
                task_weight = task.get('weight', 0)
                task_achievement = task.get('ai_achievement_rate', 0)
                print(f"          • {task_name}: {task_achievement}% × {task_weight} = {task_achievement * task_weight}")
            print(f"          = {result['achievement_rate']:.1f}% (총 가중치: {result['total_weight']})")
            
            # final_evaluation_reports 업데이트
            final_data = {
                'ai_annual_achievement_rate': int(result['achievement_rate'])
            }
            
            save_final_evaluation_report(
                member['emp_no'],
                state['team_evaluation_id'] or 0,
                final_data
            )
    
    print(f"      ✅ 연말 데이터 업데이트 완료: {len([m for m in team_members if m.get('role') != 'MANAGER'])}명")

def validate_final_update_results(state: Module2State) -> Dict[str, Any]:
    """최종 업데이트 결과 검증"""
    errors = []
    warnings = []
    
    try:
        # 1. Task 업데이트 검증
        for task_id in (state['updated_task_ids'] or []):
            task_data = fetch_cumulative_task_data(task_id, state['period_id'])
            
            # 필수 필드 검증
            if task_data.get('ai_achievement_rate') is None:
                errors.append(f"Task {task_id}: ai_achievement_rate not updated")
            
            if task_data.get('ai_contribution_score') is None:
                errors.append(f"Task {task_id}: ai_contribution_score not updated")
            
            if not task_data.get('ai_analysis_comment_task'):
                warnings.append(f"Task {task_id}: ai_analysis_comment_task empty")
            
            # 연말 전용 검증
            if state['report_type'] == "annual" and not task_data.get('ai_assessed_grade'):
                warnings.append(f"Task {task_id}: ai_assessed_grade not set for annual evaluation")
        
        # 2. Team KPI 업데이트 검증
        for kpi_id in (state['updated_team_kpi_ids'] or []):
            kpi_data = fetch_team_kpi_data(kpi_id)
            
            if kpi_data.get('ai_kpi_progress_rate') is None:
                errors.append(f"KPI {kpi_id}: ai_kpi_progress_rate not updated")
            
            if not kpi_data.get('ai_kpi_analysis_comment'):
                warnings.append(f"KPI {kpi_id}: ai_kpi_analysis_comment empty")
        
        # 3. Team evaluation 검증
        if state['team_evaluation_id']:
            with engine.connect() as connection:
                query_text = """
                    SELECT average_achievement_rate, ai_team_overall_analysis_comment,
                           year_over_year_growth
                    FROM team_evaluations 
                    WHERE team_evaluation_id = :team_evaluation_id
                """
                from sqlalchemy import text
                query = text(query_text)
                result = connection.execute(query, {"team_evaluation_id": state['team_evaluation_id']})
                row = result.fetchone()
                team_eval = row_to_dict(row) if row else {}
                
                if team_eval.get('average_achievement_rate') is None:
                    errors.append("Team evaluation: average_achievement_rate not updated")
                
                if not team_eval.get('ai_team_overall_analysis_comment'):
                    warnings.append("Team evaluation: ai_team_overall_analysis_comment empty")
        
        # 4. 분기별 팀 순위 검증 (새로 추가)
        if state['report_type'] == "quarterly":
            ranking_validation = validate_team_ranking_data(state)
            if not ranking_validation['success']:
                errors.extend(ranking_validation['errors'])
            warnings.extend(ranking_validation['warnings'])
            
            print(f"      📊 팀 순위 검증 결과:")
            print(f"        • 순위 데이터: {ranking_validation['ranking_count']}명")
            print(f"        • 팀원 수: {ranking_validation['team_member_count']}명")
        
        # 5. 레포트 검증
        feedback_report_ids = state.get('feedback_report_ids') or []
        final_evaluation_report_ids = state.get('final_evaluation_report_ids') or []
        
        if state['report_type'] == "quarterly" and feedback_report_ids:
            for report_id in feedback_report_ids:
                # feedback_reports 검증 로직
                pass
        
        elif state['report_type'] == "annual" and final_evaluation_report_ids:
            for report_id in final_evaluation_report_ids:
                # final_evaluation_reports 검증 로직
                pass
        
        # 6. 데이터 일관성 검증
        consistency_errors = validate_data_consistency(state)
        errors.extend(consistency_errors)
        
        success = len(errors) == 0
        
        if warnings:
            print(f"      ⚠️  검증 경고: {len(warnings)}건")
        
        return {
            'success': success,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'tasks_validated': len(state['updated_task_ids'] or []),
                'kpis_validated': len(state['updated_team_kpi_ids'] or []),
                'reports_validated': len(state['feedback_report_ids'] or []) + len(state['final_evaluation_report_ids'] or []),
                'ranking_validated': state['report_type'] == "quarterly"
            }
        }
        
    except Exception as e:
        print(f"      ❌ 검증 프로세스 실패: {e}")
        return {
            'success': False,
            'errors': [f"Validation process error: {str(e)}"],
            'warnings': [],
            'stats': {}
        }

def validate_team_ranking_data(state: Module2State) -> Dict[str, Any]:
    """팀 순위 데이터 검증"""
    print(f"        🔍 팀 순위 데이터 검증 중...")
    
    errors = []
    warnings = []
    
    try:
        from sqlalchemy import text
        with engine.connect() as connection:
            # feedback_reports에서 순위 데이터 조회
            query = text("""
                SELECT emp_no, ranking, ai_achievement_rate, contribution_rate
                FROM feedback_reports 
                WHERE team_evaluation_id = :team_evaluation_id
                ORDER BY ranking
            """)
            results = connection.execute(query, {"team_evaluation_id": state['team_evaluation_id']})
            ranking_data = [row_to_dict(row) for row in results]
            
            if not ranking_data:
                errors.append("팀 순위 데이터가 없습니다")
                return {'success': False, 'errors': errors, 'warnings': warnings}
            
            # 1. 순위 연속성 검증
            expected_rankings = list(range(1, len(ranking_data) + 1))
            actual_rankings = [r['ranking'] for r in ranking_data]
            
            if actual_rankings != expected_rankings:
                errors.append(f"순위가 연속적이지 않음: 예상 {expected_rankings}, 실제 {actual_rankings}")
            
            # 2. 달성률 범위 검증
            for rank_data in ranking_data:
                achievement_rate = rank_data.get('ai_achievement_rate', 0)
                if not (0 <= achievement_rate <= 200):
                    errors.append(f"사번 {rank_data['emp_no']}: 달성률 {achievement_rate}%가 범위를 벗어남")
                
                contribution_rate = rank_data.get('contribution_rate', 0)
                if not (0 <= contribution_rate <= 100):
                    warnings.append(f"사번 {rank_data['emp_no']}: 기여도 {contribution_rate}%가 범위를 벗어남")
            
            # 3. 순위와 달성률 일관성 검증
            for i in range(len(ranking_data) - 1):
                current_rate = ranking_data[i]['ai_achievement_rate']
                next_rate = ranking_data[i + 1]['ai_achievement_rate']
                
                if current_rate < next_rate:
                    errors.append(f"순위 {i+1}위({ranking_data[i]['emp_no']})의 달성률 {current_rate}%가 {i+2}위({ranking_data[i+1]['emp_no']})의 달성률 {next_rate}%보다 낮음")
            
            # 4. 팀원 수와 순위 수 일치 검증
            team_members = fetch_team_members(state['team_id'])
            non_manager_count = len([m for m in team_members if m.get('role') != 'MANAGER'])
            
            if len(ranking_data) != non_manager_count:
                warnings.append(f"팀원 수({non_manager_count}명)와 순위 수({len(ranking_data)}명)가 일치하지 않음")
            
            success = len(errors) == 0
            
            if warnings:
                print(f"          ⚠️  검증 경고: {len(warnings)}건")
            
            return {
                'success': success,
                'errors': errors,
                'warnings': warnings,
                'ranking_count': len(ranking_data),
                'team_member_count': non_manager_count
            }
            
    except Exception as e:
        print(f"          ❌ 순위 데이터 검증 실패: {e}")
        return {
            'success': False,
            'errors': [f"순위 데이터 검증 오류: {str(e)}"],
            'warnings': [],
            'ranking_count': 0,
            'team_member_count': 0
        }

def validate_data_consistency(state: Module2State) -> List[str]:
    """데이터 일관성 검증"""
    errors = []
    
    try:
        # 2. 달성률 범위 검증
        for task_id in state['updated_task_ids'] or []:
            task_data = fetch_cumulative_task_data(task_id, state['period_id'])
            achievement_rate = task_data.get('ai_achievement_rate', 0)
            
            if not (0 <= achievement_rate <= 200):
                errors.append(f"Task {task_id}: achievement_rate {achievement_rate} out of range")
        
        # 3. 팀 평균 달성률과 개별 달성률 일관성 검증
        if state['team_evaluation_id']:
            from sqlalchemy import text
            with engine.connect() as connection:
                query = text("""
                    SELECT average_achievement_rate 
                    FROM team_evaluations 
                    WHERE team_evaluation_id = :team_evaluation_id
                """)
                result = connection.execute(query, {"team_evaluation_id": state['team_evaluation_id']})
                team_avg = result.scalar_one_or_none()
                
                # 개별 Task들의 가중평균과 팀 평균이 크게 다르지 않은지 확인
                calculated_avg = calculate_team_average_achievement_rate(state['target_team_kpi_ids'])
                
                if team_avg and abs(team_avg - calculated_avg) > 15:
                    errors.append(f"Team average inconsistency: stored {team_avg} vs calculated {calculated_avg}")
        
    except Exception as e:
        errors.append(f"Consistency validation error: {str(e)}")
    
    return errors

# ================================================================
# LangGraph 워크플로우 생성
# ================================================================

def create_module2_graph():
    """모듈 2 그래프 생성 및 반환"""
    # StateGraph에 사용할 타입: TypedDict 사용 (LangGraph 권장)
    module2_workflow = StateGraph(Module2State)

    # 각 서브모듈을 노드로 등록
    module2_workflow.add_node("data_collection", data_collection_submodule)
    module2_workflow.add_node("achievement_and_grade", achievement_and_grade_calculation_submodule)
    module2_workflow.add_node("contribution", contribution_calculation_submodule)
    module2_workflow.add_node("team_analysis", team_analysis_submodule)
    module2_workflow.add_node("comment_generation", comment_generation_submodule)
    module2_workflow.add_node("db_update", db_update_submodule)

    # 엣지(실행 순서) 정의
    module2_workflow.add_edge(START, "data_collection")
    module2_workflow.add_edge("data_collection", "achievement_and_grade")
    module2_workflow.add_edge("achievement_and_grade", "contribution")
    module2_workflow.add_edge("contribution", "team_analysis")
    module2_workflow.add_edge("team_analysis", "comment_generation")
    module2_workflow.add_edge("comment_generation", "db_update")
    module2_workflow.add_edge("db_update", END)

    # 그래프 컴파일
    return module2_workflow.compile()