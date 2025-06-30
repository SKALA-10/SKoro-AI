# agent.py
# 🧠 Module 11 핵심 비즈니스 로직

import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, TypedDict

from db_utils import Module11DataAccess, DatabaseError, Module11Error
from llm_utils import (
    init_llm_client, parse_json_field, extract_json_from_llm_response,
    validate_risk_json_structure, validate_plan_json_structure,
    create_fallback_risk_analysis, create_fallback_annual_plan,
    create_fallback_annual_comment_text, create_fallback_quarterly_comment_text,
    get_year_from_period
)

# 로깅 설정
logger = logging.getLogger(__name__)

# ====================================
# 상태 관리 데이터클래스
# ====================================

class Module11AgentState(TypedDict):
    """Module 11 에이전트 상태 관리"""
    # 필수 키값들
    team_id: int
    period_id: int
    team_evaluation_id: int
    is_final: bool
    
    # 분석 과정에서 도출된 핵심 인사이트만 (가벼운 데이터)
    key_risks: Optional[List[str]]
    collaboration_bias_level: Optional[str]  # "high", "medium", "low"
    performance_trend: Optional[str]
    
    # 최종 JSON 결과
    ai_risk_result: Optional[Dict[str, Any]]
    ai_plan_result: Optional[Dict[str, Any]]
    overall_comment_result: Optional[str]  # TEXT 형태

# ====================================
# 메인 에이전트 클래스
# ====================================

class Module11TeamRiskManagementAgent:
    """Module 11 팀 리스크 관리 에이전트"""
    
    def __init__(self, data_access: Module11DataAccess):
        self.data_access = data_access
        self.llm_client = init_llm_client()
    
    async def execute(self, team_id: int, period_id: int, team_evaluation_id: int) -> Module11AgentState:
        """모듈 11 메인 실행 함수"""
        logger.info(f"Module11 시작: team_id={team_id}, period_id={period_id}")
        
        try:
            # 1. State 초기화 - 딕셔너리 리터럴 방식으로 개선
            state: Module11AgentState = {
                "team_id": team_id,
                "period_id": period_id,
                "team_evaluation_id": team_evaluation_id,
                "is_final": self._check_is_final(period_id),
                "key_risks": None,
                "collaboration_bias_level": None,
                "performance_trend": None,
                "ai_risk_result": None,
                "ai_plan_result": None,
                "overall_comment_result": None
            }
            
            # 2. 데이터 수집
            data = self._collect_all_data_sequential(state)
            
            # 3. 리스크 분석
            state["ai_risk_result"] = await self._analyze_parallel(state, data)
            
            # 4. 결과 생성 
            state = await self._generate_outputs_parallel(state, data)
            
            # 5. 저장
            self._save_results(state)
            
            logger.info(f"Module11 완료: team_id={team_id}")
            return state
            
        except Exception as e:
            logger.error(f"Module11 실행 중 오류: {str(e)}")
            raise Module11Error(f"모듈 11 실행 실패: {str(e)}")
    
    def _check_is_final(self, period_id: int) -> bool:
        """기간이 연말인지 확인"""
        try:
            period_info = self.data_access.get_period_info(period_id)
            return bool(period_info['is_final'])
        except Exception as e:
            logger.error(f"연말 여부 확인 실패: {str(e)}")
            raise DatabaseError(f"기간 정보 조회 실패: {str(e)}")
    
    def _collect_all_data_sequential(self, state: Module11AgentState) -> Dict[str, Any]:
        """순차적 데이터 수집"""
        logger.info(f"데이터 수집 시작: team_evaluation_id={state['team_evaluation_id']}")
        
        data = {}
        
        try:
            # 1. 기본 정보
            data['period_info'] = self.data_access.get_period_info(state['period_id'])
            data['team_info'] = self.data_access.get_team_info(state['team_id'])
            data['team_members'] = self.data_access.get_team_members(state['team_id'])
            
            # 2. 성과 데이터
            year = data['period_info']['year']
            data['team_kpis'] = self.data_access.get_team_kpis(state['team_id'], year)
            data['team_performance'] = self.data_access.get_team_performance(state['team_id'], state['period_id'])
            
            # 3. 협업 분석 (JSON 파싱)
            collaboration_data = self.data_access.get_collaboration_data(state['team_evaluation_id'])
            data['collaboration_matrix'] = parse_json_field(collaboration_data.get('ai_collaboration_matrix'))
            data['team_coaching'] = collaboration_data.get('ai_team_coaching')  # TEXT 필드
            
            # ai_team_comparison은 분기에만 필요
            if not state['is_final']:
                data['team_comparison'] = collaboration_data.get('ai_team_comparison')  # TEXT 필드
            
            # 4. 개인 리스크
            data['individual_risks'] = self.data_access.get_individual_risks(state['team_evaluation_id'], state['is_final'])
            
            # 5. 분기별/연말별 추가 데이터
            if not state['is_final']:
                # 분기: 전분기 데이터
                data['previous_quarter'] = self.data_access.get_previous_quarter_data(state['team_id'], data['period_info'])
            else:
                # 연말: temp_evaluations 데이터
                data['temp_evaluations'] = self.data_access.get_temp_evaluations(state['team_id'])
            
            logger.info(f"데이터 수집 완료: {len(data)} 개 데이터셋")
            return data
            
        except Exception as e:
            logger.error(f"데이터 수집 실패: {str(e)}")
            raise DatabaseError(f"데이터 수집 중 오류 발생: {str(e)}")

    # ====================================
    # 리스크 분석 메서드들
    # ====================================

    async def _analyze_parallel(self, state: Module11AgentState, data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 기반 병렬 리스크 분석"""
        logger.info("LLM 기반 리스크 분석 시작 (병렬)")
        
        try:
            # 독립적인 LLM 분석들을 병렬 처리
            tasks = [
                self._analyze_collaboration_risks_with_llm_async(data.get('collaboration_matrix'), data.get('team_members', [])),
                self._analyze_individual_risk_patterns_with_llm_async(data.get('individual_risks', []), data.get('team_members', [])),
                self._analyze_performance_trends_with_llm_async(data.get('team_performance', {}), data.get('team_kpis', []))
            ]
            
            risk_analyses = await asyncio.gather(*tasks)
            
            # LLM으로 최종 통합 분석
            integrated_analysis = await self._integrate_risk_analysis_with_llm_async(
                collaboration_risks=risk_analyses[0],
                individual_patterns=risk_analyses[1], 
                performance_trends=risk_analyses[2],
                state=state,
                data=data
            )
            
            logger.info("LLM 기반 리스크 분석 완료")
            return integrated_analysis
            
        except Exception as e:
            logger.error(f"LLM 리스크 분석 실패: {str(e)}")
            raise Module11Error(f"LLM 리스크 분석 중 오류: {str(e)}")

    async def _integrate_risk_analysis_with_llm_async(self, collaboration_risks: Dict, individual_patterns: Dict, 
                                                    performance_trends: Dict, state: Module11AgentState, data: Dict) -> Dict[str, Any]:
        """개선된 LLM 기반 리스크 분석 통합 및 최종 JSON 생성"""
        
        # 팀 기본 정보
        team_info = data.get('team_info', {})
        period_info = data.get('period_info', {})
        team_members = data.get('team_members', [])
        team_performance = data.get('team_performance', {})
        
        prompt = f"""
당신은 팀 운영 및 리스크 관리 전문가입니다. 다음 세부 분석 결과들을 종합하여 team_evaluations.ai_risk에 저장될 구조화된 JSON을 생성해주세요.

## 팀 기본 정보:
- 팀명: {team_info.get('team_name', '')}
- 평가기간: {period_info.get('period_name', '')} ({'연말' if state['is_final'] else '분기'} 평가)
- 팀원 수: {len(team_members)}명
- 팀 성과: 달성률 {team_performance.get('average_achievement_rate', 0)}%, 상대성과 {team_performance.get('relative_performance', 0)}%

## 협업 리스크 분석 결과:
{json.dumps(collaboration_risks, ensure_ascii=False, indent=2)}

## 개인별 리스크 패턴 분석 결과:
{json.dumps(individual_patterns, ensure_ascii=False, indent=2)}

## 성과 트렌드 분석 결과:
{json.dumps(performance_trends, ensure_ascii=False, indent=2)}

위 분석 결과들을 종합하여 다음과 같은 정확한 구조의 JSON을 생성해주세요:

```json
{{
  "risk_analysis": {{
    "major_risks": [
      {{
        "risk_name": "실제 데이터에 기반한 구체적 리스크명",
        "severity": "high/medium/low",
        "description": "분석 결과를 종합한 구체적이고 실용적인 리스크 설명",
        "causes": [
          "데이터에서 확인된 구체적 발생 원인 1",
          "데이터에서 확인된 구체적 발생 원인 2"
        ],
        "impacts": [
          {{
            "impact_scope": "individual/team/organization",
            "timeline": "immediate/short_term/long_term",
            "description": "실제 예상되는 구체적 영향 설명"
          }}
        ],
        "strategies": [
          {{
            "description": "데이터 분석 결과에 기반한 구체적이고 실행 가능한 운영 개선 전략"
          }}
        ]
      }}
    ]
  }}
}}
```

## 중요 요구사항:
1. 실제 분석 결과를 기반으로 한 구체적이고 실용적인 내용 작성
2. major_risks는 심각도 순으로 정렬하여 최대 5개까지
3. 각 리스크는 실제 데이터 근거와 함께 설명
4. causes, impacts, strategies는 추상적이지 않고 구체적으로 작성
5. 팀 실정에 맞는 실행 가능한 전략 제시
6. JSON 형식을 정확히 준수
"""

        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result_json = extract_json_from_llm_response(str(response.content))
            final_result = json.loads(result_json)
            
            # 추가 검증 및 후처리
            if not validate_risk_json_structure(final_result):
                logger.warning("리스크 JSON 구조 검증 실패 - 폴백 사용")
                return create_fallback_risk_analysis()
            
            # State에 핵심 정보 저장
            major_risks = final_result.get('risk_analysis', {}).get('major_risks', [])
            state['key_risks'] = [risk['risk_name'] for risk in major_risks]
            state['collaboration_bias_level'] = collaboration_risks.get('bias_level', 'medium')
            
            logger.info(f"✅ 고품질 리스크 분석 완료: {len(major_risks)}개 리스크 식별")
            return final_result
            
        except Exception as e:
            logger.error(f"최종 리스크 분석 통합 실패: {str(e)}")
            return create_fallback_risk_analysis()

    async def _analyze_collaboration_risks_with_llm_async(self, collaboration_matrix: Optional[Dict], team_members: List[Dict]) -> Dict[str, Any]:
        """개선된 LLM 기반 협업 리스크 분석"""
        
        if not collaboration_matrix:
            return {
                'risks': [],
                'collaboration_insights': ['협업 데이터 부족으로 분석 불가'],
                'bias_level': 'unknown'
            }
        
        # 팀원 정보 구조화
        member_info = []
        for member in team_members:
            member_info.append({
                'emp_no': member.get('emp_no'),
                'name': member.get('emp_name'),
                'position': member.get('position'),
                'cl': member.get('cl')
            })
        
        prompt = f"""
당신은 조직 협업 분석 전문가입니다. 다음 협업 매트릭스 데이터를 분석하여 구체적인 리스크를 식별해주세요.

## 팀원 정보:
{json.dumps(member_info, ensure_ascii=False, indent=2)}

## 협업 매트릭스 데이터:
{json.dumps(collaboration_matrix, ensure_ascii=False, indent=2)}

협업 리스크를 다음 JSON 구조로 분석해주세요:

```json
{{
  "risks": [
    {{
      "risk_name": "구체적 협업 리스크명",
      "severity": "high/medium/low",
      "description": "리스크에 대한 구체적 설명 (데이터 근거 포함)",
      "evidence": [
        "협업 매트릭스에서 발견된 구체적 증거 1",
        "협업 매트릭스에서 발견된 구체적 증거 2"
      ],
      "affected_members": ["emp_no1", "emp_no2"]
    }}
  ],
  "collaboration_insights": [
    "협업 패턴에서 발견된 주요 인사이트 1",
    "협업 패턴에서 발견된 주요 인사이트 2"
  ],
  "bias_level": "high/medium/low"
}}
```

## 분석 기준:
1. 협업 매트릭스의 실제 수치를 기반으로 분석
2. 팀원간 협업 불균형, 소외된 구성원, 과도한 의존성 등을 식별
3. 구체적인 데이터 근거와 함께 리스크 설명
4. 실제 영향받을 팀원들을 명시
"""

        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result_json = extract_json_from_llm_response(str(response.content))
            return json.loads(result_json)
        except Exception as e:
            logger.error(f"협업 리스크 LLM 분석 실패: {str(e)}")
            return {
                'risks': [{
                    'risk_name': '협업 데이터 분석 오류',
                    'severity': 'medium',
                    'description': f'LLM 분석 중 오류 발생: {str(e)[:100]}',
                    'evidence': ['시스템 분석 한계'],
                    'affected_members': []
                }],
                'collaboration_insights': ['시스템 분석 한계로 인한 제한적 분석'],
                'bias_level': 'unknown'
            }

    async def _analyze_individual_risk_patterns_with_llm_async(self, individual_risks: List[Dict], team_members: List[Dict]) -> Dict[str, Any]:
        """개선된 LLM 기반 개인별 리스크 패턴 분석"""
        
        if not individual_risks:
            return {
                'risks': [],
                'performance_patterns': ['개인 평가 데이터 부족으로 분석 불가']
            }
        
        # 개인별 성과 데이터 구조화
        performance_data = []
        for risk in individual_risks:
            performance_data.append({
                'emp_no': risk.get('emp_no'),
                'score': risk.get('score'),
                'contribution_rate': risk.get('contribution_rate'),
                'attitude': risk.get('attitude'),
                'growth_coaching': risk.get('ai_growth_coaching'),
                'summary_comment': risk.get('ai_overall_contribution_summary_comment') or risk.get('ai_annual_performance_summary_comment')
            })
        
        prompt = f"""
당신은 인사 성과 분석 전문가입니다. 다음 개인별 성과 데이터를 분석하여 팀 차원의 리스크를 식별해주세요.

## 팀원 기본 정보:
{json.dumps([{'emp_no': m.get('emp_no'), 'name': m.get('emp_name'), 'position': m.get('position'), 'cl': m.get('cl')} for m in team_members], ensure_ascii=False, indent=2)}

## 개인별 성과 데이터:
{json.dumps(performance_data, ensure_ascii=False, indent=2)}

개인 성과 패턴을 분석하여 다음 JSON 구조로 팀 리스크를 도출해주세요:

```json
{{
  "risks": [
    {{
      "risk_name": "개인 성과 기반 팀 리스크명",
      "severity": "high/medium/low", 
      "description": "성과 데이터를 근거로 한 구체적 리스크 설명",
      "affected_members": ["emp_no1", "emp_no2"],
      "evidence": [
        "성과 데이터에서 발견된 구체적 증거 1",
        "성과 데이터에서 발견된 구체적 증거 2"
      ]
    }}
  ],
  "performance_patterns": [
    "팀 성과 패턴에서 발견된 주요 인사이트 1",
    "팀 성과 패턴에서 발견된 주요 인사이트 2"
  ]
}}
```

## 분석 포인트:
1. 성과 편차가 큰 구성원들의 팀 영향도
2. 저성과자의 팀 사기 및 분위기 영향
3. 고성과자의 번아웃 또는 이직 위험성
4. 성장 코칭 필요성이 높은 구성원들의 패턴
5. 실제 수치를 근거로 구체적 분석
"""

        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result_json = extract_json_from_llm_response(str(response.content))
            return json.loads(result_json)
        except Exception as e:
            logger.error(f"개인 리스크 LLM 분석 실패: {str(e)}")
            return {
                'risks': [{
                    'risk_name': '개인 성과 분석 한계',
                    'severity': 'medium',
                    'description': f'개인 성과 데이터 분석 중 오류: {str(e)[:100]}',
                    'affected_members': [],
                    'evidence': ['시스템 분석 한계']
                }],
                'performance_patterns': ['시스템 한계로 인한 제한적 분석']
            }

    async def _analyze_performance_trends_with_llm_async(self, team_performance: Dict, team_kpis: List[Dict]) -> Dict[str, Any]:
        """개선된 LLM 기반 성과 트렌드 분석"""
        
        if not team_performance:
            return {
                'risks': [],
                'trends': ['성과 데이터 부족으로 분석 불가']
            }
        
        # KPI 데이터 구조화
        kpi_summary = []
        for kpi in team_kpis:
            kpi_summary.append({
                'kpi_name': kpi.get('kpi_name'),
                'weight': kpi.get('weight'),
                'progress_rate': kpi.get('ai_kpi_progress_rate'),
                'analysis_comment': kpi.get('ai_kpi_analysis_comment')
            })
        
        prompt = f"""
당신은 팀 성과 분석 전문가입니다. 다음 성과 데이터를 종합 분석하여 팀의 성과 관련 리스크를 식별해주세요.

## 팀 전체 성과:
{json.dumps(team_performance, ensure_ascii=False, indent=2)}

## 팀 KPI 현황:
{json.dumps(kpi_summary, ensure_ascii=False, indent=2)}

성과 트렌드를 분석하여 다음 JSON 구조로 리스크를 도출해주세요:

```json
{{
  "risks": [
    {{
      "risk_name": "성과 관련 구체적 리스크명",
      "severity": "high/medium/low",
      "description": "성과 데이터를 근거로 한 구체적 리스크 설명",
      "affected_kpis": ["kpi_name1", "kpi_name2"],
      "evidence": [
        "성과 데이터에서 발견된 구체적 수치적 근거 1",
        "성과 데이터에서 발견된 구체적 수치적 근거 2"
      ]
    }}
  ],
  "performance_trends": [
    "성과 트렌드에서 발견된 주요 패턴 1",
    "성과 트렌드에서 발견된 주요 패턴 2"
  ]
}}
```

## 분석 기준:
1. 평균 달성률, 상대 성과, 전년 대비 성장률의 실제 수치 분석
2. KPI별 진행률과 가중치를 고려한 위험도 평가
3. 성과 트렌드의 지속가능성 및 개선 가능성 판단
4. 구체적 수치를 근거로 한 명확한 리스크 식별
"""

        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result_json = extract_json_from_llm_response(str(response.content))
            return json.loads(result_json)
        except Exception as e:
            logger.error(f"성과 트렌드 LLM 분석 실패: {str(e)}")
            return {
                'risks': [{
                    'risk_name': '성과 데이터 분석 한계',
                    'severity': 'medium',
                    'description': f'성과 분석 중 오류 발생: {str(e)[:100]}',
                    'affected_kpis': [],
                    'evidence': ['시스템 분석 한계']
                }],
                'performance_trends': ['시스템 한계로 인한 제한적 분석']
            }

    # ====================================
    # 결과 생성 메서드들
    # ====================================

    async def _generate_outputs_parallel(self, state: Module11AgentState, data: Dict[str, Any]) -> Module11AgentState:
        """LLM 기반 병렬 결과 생성"""
        logger.info("최종 결과 생성 시작 (병렬)")
        
        try:
            if state['is_final']:
                # 연말: ai_plan과 overall_comment 병렬 생성
                tasks = [
                    self._generate_annual_plan_with_llm_async(state, data),
                    self._generate_overall_comment_with_llm_async(state, data)
                ]
                
                ai_plan, overall_comment = await asyncio.gather(*tasks)
                state["ai_plan_result"] = ai_plan
                state["overall_comment_result"] = overall_comment
                
            else:
                # 분기: overall_comment만
                state["overall_comment_result"] = await self._generate_overall_comment_with_llm_async(state, data)
            
            logger.info("최종 결과 생성 완료")
            return state
            
        except Exception as e:
            logger.error(f"최종 결과 생성 실패: {str(e)}")
            raise Module11Error(f"최종 결과 생성 중 오류: {str(e)}")

    async def _generate_annual_plan_with_llm_async(self, state: Module11AgentState, data: Dict[str, Any]) -> Dict[str, Any]:
        """개선된 LLM 기반 연말 계획 생성"""
        
        team_info = data.get('team_info', {})
        period_info = data.get('period_info', {})
        team_members = data.get('team_members', [])
        temp_evaluations = data.get('temp_evaluations', [])
        individual_risks = data.get('individual_risks', [])
        
        # 팀원별 중간평가 결과 구조화
        member_evaluations = {}
        for temp_eval in temp_evaluations:
            emp_no = temp_eval.get('emp_no')
            if emp_no:
                member_evaluations[emp_no] = {
                    'score': temp_eval.get('score'),
                    'manager_score': temp_eval.get('manager_score'),
                    'reason': temp_eval.get('reason'),
                    'status': temp_eval.get('status')
                }
        
        prompt = f"""
당신은 팀 운영 전략 수립 전문가입니다. 다음 종합 데이터를 바탕으로 차년도({period_info.get('year', get_year_from_period(state['period_id'])) + 1}년) 팀 운영 계획을 수립해주세요.

## 팀 기본 정보:
{json.dumps(team_info, ensure_ascii=False, indent=2)}

## 팀원 현황:
{json.dumps([{'emp_no': m.get('emp_no'), 'name': m.get('emp_name'), 'position': m.get('position'), 'cl': m.get('cl'), 'salary': m.get('salary')} for m in team_members], ensure_ascii=False, indent=2)}

## 중간평가 결과:
{json.dumps(member_evaluations, ensure_ascii=False, indent=2)}

## 최종 평가 결과:
{json.dumps(individual_risks, ensure_ascii=False, indent=2)}

## 식별된 리스크:
{json.dumps(state["ai_risk_result"], ensure_ascii=False, indent=2)}

## 협업 분석 결과:
{json.dumps(data.get('collaboration_matrix'), ensure_ascii=False, indent=2)}

차년도 운영 계획을 다음과 같은 정확한 JSON 구조로 작성해주세요:

```json
{{
  "annual_plans": [
    {{
      "personnel_strategies": [
        {{
          "target": "구체적 대상자명 또는 포지션",
          "action": "실제 실행 가능한 구체적 방안 (교육, 승진, 채용, 역할 변경 등)"
        }}
      ],
      "collaboration_improvements": [
        {{
          "current_issue": "현재 확인된 구체적 협업 문제점",
          "improvement": "문제 해결을 위한 구체적 개선 방안",
          "expected_benefit": "개선으로 인한 구체적 기대효과",
          "target": "측정 가능한 구체적 목표 지표"
        }}
      ]
    }}
  ]
}}
```

## 작성 가이드라인:
1. **personnel_strategies**: 
   - 실제 팀원들의 성과와 평가 결과를 기반으로 구체적 전략 수립
   - 고성과자 유지, 저성과자 개선, 신규 채용 필요성 등을 실데이터 기반으로 판단
   - 실제 실행 가능한 액션 아이템으로 작성

2. **collaboration_improvements**: 
   - 협업 매트릭스와 리스크 분석 결과를 기반으로 실제 문제점 식별
   - 구체적이고 실행 가능한 개선 방안 제시
   - 측정 가능한 목표 지표 설정

3. **전체적으로**: 
   - 추상적이지 않고 구체적인 내용
   - 실제 데이터와 분석 결과에 기반
   - 차년도에 실제로 실행할 수 있는 계획
"""
        
        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result_json = extract_json_from_llm_response(str(response.content))
            final_result = json.loads(result_json)
            
            # 구조 검증
            if not validate_plan_json_structure(final_result):
                logger.warning("연말 계획 JSON 구조 검증 실패 - 폴백 사용")
                return create_fallback_annual_plan(period_info)
            
            logger.info("✅ 고품질 연말 계획 생성 완료")
            return final_result
            
        except Exception as e:
            logger.error(f"연말 계획 LLM 생성 실패: {str(e)}")
            return create_fallback_annual_plan(period_info)

    async def _generate_overall_comment_with_llm_async(self, state: Module11AgentState, data: Dict[str, Any]) -> str:
        """개선된 LLM 기반 총평 생성"""
        
        if state['is_final']:
            return await self._generate_annual_overall_comment_text_improved(state, data)
        else:
            return await self._generate_quarterly_overall_comment_text_improved(state, data)

    async def _generate_annual_overall_comment_text_improved(self, state: Module11AgentState, data: Dict[str, Any]) -> str:
        """개선된 연말 총평 생성"""
        
        team_info = data.get('team_info', {})
        period_info = data.get('period_info', {})
        team_performance = data.get('team_performance', {})
        team_kpis = data.get('team_kpis', [])
        temp_evaluations = data.get('temp_evaluations', [])
        
        # 성과 지표 요약
        performance_summary = {
            'achievement_rate': team_performance.get('average_achievement_rate', 0),
            'relative_performance': team_performance.get('relative_performance', 0),
            'year_over_year_growth': team_performance.get('year_over_year_growth', 0),
            'kpi_count': len(team_kpis),
            'team_size': len(data.get('team_members', [])),
            'temp_eval_avg': sum(eval.get('score', 0) or 0 for eval in temp_evaluations) / len(temp_evaluations) if temp_evaluations else 0
        }
        
        prompt = f"""
당신은 경영진에게 보고하는 팀 운영 전략 전문가입니다. 다음 종합 데이터를 바탕으로 연말 총평을 작성해주세요.

## 팀 기본 정보:
- 팀명: {team_info.get('team_name')} 
- 소속: {team_info.get('part_name')} > {team_info.get('headquarter_name')}
- 팀원 수: {performance_summary['team_size']}명

## 연간 성과 요약:
- 평균 달성률: {performance_summary['achievement_rate']}%
- 상대적 성과: {performance_summary['relative_performance']}%  
- 전년 대비 성장률: {performance_summary['year_over_year_growth']}%
- KPI 개수: {performance_summary['kpi_count']}개
- 중간평가 평균: {performance_summary['temp_eval_avg']:.1f}점

## 상세 KPI 현황:
{json.dumps([{'name': kpi.get('kpi_name'), 'weight': kpi.get('weight'), 'progress': kpi.get('ai_kpi_progress_rate')} for kpi in team_kpis], ensure_ascii=False, indent=2)}

## 식별된 주요 리스크:
{json.dumps(state['key_risks'], ensure_ascii=False)}

## 차년도 계획 요약:
{json.dumps(state["ai_plan_result"], ensure_ascii=False, indent=2)}

다음 구조로 연말 총평을 작성해주세요:

**[팀 성과 방향]**
연간 성과 달성도와 성장 궤적을 구체적 수치와 함께 평가하고, 팀의 전략적 방향성과 성과 창출 능력을 분석해주세요. (3-4문장)

**[구조적 인식]**  
팀의 조직적 강점과 구조적 도전과제를 리스크 분석 결과와 연계하여 설명하고, 지속가능한 성장을 위한 핵심 요소를 식별해주세요. (3-4문장)

**[향후 운영 전략]**
차년도 계획과 연계하여 전략적 우선순위와 성공을 위한 핵심 실행 과제를 제시하고, 구체적인 성과 목표와 실행 방안을 제시해주세요. (3-4문장)

## 작성 요구사항:
1. 실제 수치와 데이터를 근거로 구체적으로 작성
2. 추상적 표현보다는 실용적이고 실행 가능한 내용 중심
3. 경영진이 의사결정에 활용할 수 있는 명확한 인사이트 제공
4. 각 섹션은 독립적이면서도 전체적으로 일관된 스토리 구성
"""

        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result = str(response.content).strip()
            
            # 내용 품질 검증
            if len(result) < 300:
                logger.warning("연말 총평이 너무 짧음 - 폴백 사용")
                return create_fallback_annual_comment_text()
            
            logger.info(f"✅ 고품질 연말 총평 생성 완료: {len(result)}자")
            return result
            
        except Exception as e:
            logger.error(f"연말 총평 LLM 생성 실패: {str(e)}")
            return create_fallback_annual_comment_text()

    async def _generate_quarterly_overall_comment_text_improved(self, state: Module11AgentState, data: Dict[str, Any]) -> str:
        """개선된 분기 총평 생성"""
        
        team_info = data.get('team_info', {})
        period_info = data.get('period_info', {})
        team_performance = data.get('team_performance', {})
        previous_quarter = data.get('previous_quarter', {})
        team_comparison = data.get('team_comparison', '')
        
        # 전분기 대비 변화 계산
        current_achievement = team_performance.get('average_achievement_rate', 0)
        prev_achievement = previous_quarter.get('average_achievement_rate', 0) if previous_quarter else 0
        achievement_change = current_achievement - prev_achievement
        
        prompt = f"""
당신은 분기별 성과 분석 전문가입니다. 다음 데이터를 바탕으로 분기 총평을 작성해주세요.

## 팀 기본 정보:
- 팀명: {team_info.get('team_name')}
- 평가 기간: {period_info.get('period_name')}

## 현재 분기 성과:
- 평균 달성률: {current_achievement}%
- 상대적 성과: {team_performance.get('relative_performance', 0)}%

## 전분기 성과 (비교):
- 전분기 달성률: {prev_achievement}%
- 달성률 변화: {achievement_change:+.1f}%p

## 유사팀 비교 분석:
{team_comparison}

## 식별된 주요 리스크:
{json.dumps(state['key_risks'], ensure_ascii=False)}

다음 구조로 분기 총평을 작성해주세요:

**[전분기 대비 변화]**
전분기 대비 주요 변화사항과 성과 트렌드를 구체적 수치와 함께 분석해주세요. (2-3문장)

**[유사조직 대비 현황]**
상대적 위치와 벤치마킹 인사이트를 제시하고, 개선 또는 유지해야 할 포인트를 명확히 해주세요. (2-3문장)

**[종합 평가]**
핵심 인사이트와 다음 분기까지 즉시 집중해야 할 영역을 실행 가능한 액션과 함께 제시해주세요. (2-3문장)

## 작성 요구사항:
1. 구체적 수치와 변화량을 반드시 포함
2. 실행 가능한 개선 방향 제시
3. 다음 분기 성과 향상을 위한 명확한 가이드 제공
"""

        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            result = str(response.content).strip()
            
            # 내용 품질 검증
            if len(result) < 200:
                logger.warning("분기 총평이 너무 짧음 - 폴백 사용")
                return create_fallback_quarterly_comment_text()
            
            logger.info(f"✅ 고품질 분기 총평 생성 완료: {len(result)}자")
            return result
            
        except Exception as e:
            logger.error(f"분기 총평 LLM 생성 실패: {str(e)}")
            return create_fallback_quarterly_comment_text()

    # ====================================
    # 저장 메서드들
    # ====================================

    def _save_results(self, state: Module11AgentState) -> None:
        """분석 결과를 DB에 저장"""
        logger.info(f"분석 결과 저장 시작: team_evaluation_id={state['team_evaluation_id']}")
        
        try:
            # 1. 저장할 데이터 준비
            save_data = self._prepare_save_data(state)
            
            if not save_data:
                logger.warning("저장할 데이터가 없습니다.")
                return
            
            # 2. 저장 전 검증 (존재하는 레코드인지 확인)
            if not self.data_access.verify_team_evaluation_exists(state['team_evaluation_id']):
                raise DatabaseError(f"team_evaluation_id {state['team_evaluation_id']}가 존재하지 않습니다.")
            
            # 3. 실제 업데이트
            affected_rows = self.data_access.update_team_evaluations(state['team_evaluation_id'], save_data)
            
            if affected_rows == 0:
                raise DatabaseError(f"업데이트된 행이 없음: team_evaluation_id={state['team_evaluation_id']}")
            
            # 4. 저장 후 검증
            self.data_access.verify_save_success(state['team_evaluation_id'], save_data)
            
            logger.info(f"✅ 분석 결과 저장 완료: team_evaluation_id={state['team_evaluation_id']}")
            
        except Exception as e:
            logger.error(f"❌ 분석 결과 저장 실패: {str(e)}")
            raise DatabaseError(f"DB 저장 중 오류 발생: {str(e)}")

    def _prepare_save_data(self, state: Module11AgentState) -> dict:
        """저장용 데이터 준비"""
        save_data = {}
        
        # ai_risk는 JSON으로 저장
        if state["ai_risk_result"]:
            save_data['ai_risk'] = json.dumps(state["ai_risk_result"], ensure_ascii=False)
            logger.info(f"ai_risk 데이터 준비 완료: {len(save_data['ai_risk'])}자")
        
        # ai_plan은 JSON으로 저장 (연말만)
        if state['is_final'] and state["ai_plan_result"]:
            save_data['ai_plan'] = json.dumps(state["ai_plan_result"], ensure_ascii=False)
            logger.info(f"ai_plan 데이터 준비 완료: {len(save_data['ai_plan'])}자")
        
        # overall_comment는 텍스트로 저장
        if state["overall_comment_result"]:
            save_data['overall_comment'] = state["overall_comment_result"]
            logger.info(f"overall_comment 데이터 준비 완료: {len(save_data['overall_comment'])}자")
        
        logger.info(f"총 {len(save_data)}개 필드 준비 완료: {list(save_data.keys())}")
        return save_data