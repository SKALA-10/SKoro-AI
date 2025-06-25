# ================================================================
# llm_utils_module10.py - 모듈 10 LLM 관련 유틸리티
# ================================================================

import re
import json
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ================================================================
# LLM 호출 함수들
# ================================================================

def call_llm_for_growth_analysis(basic_info: Dict, performance_data: Dict, 
                                peer_talk_data: Dict, fourp_data: Dict, 
                                collaboration_data: Dict) -> Dict:
    """성장 분석을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "직원")
    cl = basic_info.get("cl", "CL2")
    position = basic_info.get("position", "직책 정보 없음")
    
    # 4P 점수 추출 및 분석
    fourp_scores = {
        "passionate": fourp_data.get("passionate", {}).get("score", 3.0),
        "proactive": fourp_data.get("proactive", {}).get("score", 3.0), 
        "professional": fourp_data.get("professional", {}).get("score", 3.0),
        "people": fourp_data.get("people", {}).get("score", 3.0)
    }
    
    # 4P 최고/최저 영역 계산
    max_4p = max(fourp_scores.keys(), key=lambda k: fourp_scores[k])
    min_4p = min(fourp_scores.keys(), key=lambda k: fourp_scores[k])
    
    system_prompt = """
    당신은 데이터 기반 성장 컨설턴트입니다. 
    직원의 구체적 데이터를 분석하여 실행 가능한 성장 방안을 제시해야 합니다.
    
    ⚠️ 중요 원칙:
    1. 일반적이거나 뻔한 제안 금지 ("팀워크 강화", "소통 개선" 등)
    2. 데이터에 근거한 구체적 분석 필수
    3. 바로 실행 가능한 액션 아이템만 제시
    4. CL/직무별 현실적 발전 경로 고려
    
    분석 방법:
    - 성장 포인트: 4P 최고점수 영역을 활용한 구체적 발전 방향
    - 보완 영역: 4P 최저점수 + Peer Talk 우려 → 명확한 스킬/행동 개선점
    - 추천 활동: 3개월 내 실행 가능한 구체적 액션 (교육명, 프로젝트명, 구체적 행동)
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    CL: {cl} (CL3=시니어급, CL2=중간급, CL1=주니어급)
    직책: {position}
    </직원 정보>

    <성과 데이터 분석>
    달성률: {performance_data.get('ai_achievement_rate', 0)}% 
    → {'목표 초과달성' if performance_data.get('ai_achievement_rate', 0) >= 100 else '목표 미달성'}
    기여도: {performance_data.get('contribution_rate', 0)}%
    팀 내 달성률 순위: {performance_data.get('ranking', 0)}위
    → {'상위권' if performance_data.get('ranking', 5) <= 3 else '중위권' if performance_data.get('ranking', 5) <= 5 else '하위권'}
    </성과 데이터 분석>

    <4P 점수 분석>
    Passionate: {fourp_scores['passionate']}점 {'(강점)' if max_4p == 'passionate' else '(보완)' if min_4p == 'passionate' else '(보통)'}
    Proactive: {fourp_scores['proactive']}점 {'(강점)' if max_4p == 'proactive' else '(보완)' if min_4p == 'proactive' else '(보통)'}
    Professional: {fourp_scores['professional']}점 {'(강점)' if max_4p == 'professional' else '(보완)' if min_4p == 'professional' else '(보통)'}
    People: {fourp_scores['people']}점 {'(강점)' if max_4p == 'people' else '(보완)' if min_4p == 'people' else '(보통)'}
    
    최고 강점: {max_4p}({fourp_scores[max_4p]}점)
    최대 보완점: {min_4p}({fourp_scores[min_4p]}점)
    </4P 점수 분석>

    <Peer Talk 핵심 인사이트>
    강점 키워드: {peer_talk_data.get('strengths', '정보 없음')}
    우려 키워드: {peer_talk_data.get('concerns', '정보 없음')}
    협업 관찰: {peer_talk_data.get('collaboration_observations', '정보 없음')}
    </Peer Talk 핵심 인사이트>

    <협업 패턴 분석>
    협업률: {collaboration_data.get('collaboration_rate', 0)}% 
    → {'과도한 협업' if collaboration_data.get('collaboration_rate', 0) >= 90 else '적정 협업' if collaboration_data.get('collaboration_rate', 0) >= 70 else '협업 부족'}
    팀 역할: {collaboration_data.get('team_role', '정보 없음')}
    협업 편중도: {collaboration_data.get('collaboration_bias', '보통')}
    </협업 패턴 분석>

    위 데이터를 바탕으로 다음 기준에 따라 분석하세요:

    📈 성장 포인트 (2-3개):
    - {max_4p} 강점({fourp_scores[max_4p]}점)을 활용한 구체적 발전 방향
    - 성과 데이터 기반 검증된 역량 영역
    - 예: "복잡한 기술 문제 해결 능력을 활용한 아키텍처 설계 역할 확대"

    🎯 보완 영역 (1-2개):  
    - {min_4p} 영역({fourp_scores[min_4p]}점) 개선을 위한 명확한 스킬/행동
    - Peer Talk 우려사항 기반 구체적 개선점
    - 예: "코드리뷰 시 설명 방식 개선을 통한 지식 전달 스킬 향상"

    🚀 추천 활동 (3개):
    - 3개월 내 실행 가능한 구체적 액션
    - {cl} + {position} 레벨에 적합한 현실적 활동
    - 예: "사내 기술 세미나 발표 1회 진행", "타팀과의 API 설계 협업 프로젝트 참여"

    JSON 응답:
    {{
        "growth_points": [
            "{max_4p} 강점을 활용한 구체적이고 실행 가능한 발전 방향 1",
            "성과 데이터 기반 검증된 역량을 확장하는 방향 2"
        ],
        "improvement_areas": [
            "{min_4p} 영역 개선을 위한 명확하고 측정 가능한 행동 1",
            "Peer Talk 우려사항 해결을 위한 구체적 스킬 향상 2"
        ],
        "recommended_activities": [
            "{cl} {position}에게 적합한 3개월 내 실행 가능한 구체적 액션 1",
            "측정 가능한 결과를 낼 수 있는 구체적 활동 2", 
            "협업/성과 개선에 직접 도움이 되는 실행 가능한 액션 3"
        ]
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        return json.loads(json_output)
        
    except Exception as e:
        print(f"성장 분석 LLM 호출 실패: {e}")
        return {
            "growth_points": ["데이터 분석 중 오류 발생"],
            "improvement_areas": ["데이터 분석 중 오류 발생"], 
            "recommended_activities": ["데이터 분석 중 오류 발생"]
        }

def call_llm_for_focus_coaching_analysis(peer_talk_data: Dict, performance_data: Dict, 
                                       collaboration_data: Dict, fourp_data: Dict) -> Dict:
    """집중 코칭 필요성 분석을 위한 LLM 호출"""
    
    system_prompt = """
    당신은 HR 전문가입니다. 직원의 데이터를 분석하여 집중 코칭이 필요한지 판단해주세요.
    
    집중 코칭 필요 기준:
    1. 성과 이슈: 달성률 70% 미만, 팀 내 하위권
    2. 협업 이슈: 협업률 60% 미만, Peer Talk 심각한 우려사항 2개 이상
    3. 태도 이슈: People 점수 3.0 미만, 부정적 피드백 다수
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <분석 데이터>
    달성률: {performance_data.get('ai_achievement_rate', 0)}%
    달성률 순위: {performance_data.get('ranking', 0)}위
    협업률: {collaboration_data.get('collaboration_rate', 0)}%
    People 점수: {fourp_data.get('people', {}).get('score', 3.0)}점
    
    Peer Talk 우려사항: {peer_talk_data.get('concerns', '없음')}
    협업 관찰: {peer_talk_data.get('collaboration_observations', '없음')}
    </분석 데이터>

    JSON 응답:
    {{
        "focus_coaching_needed": true/false,
        "issue_summary": "핵심 이슈 요약 (집중 코칭 필요한 경우만)",
        "root_cause_analysis": "근본 원인 분석",
        "risk_factors": "리스크 요소 및 점검 포인트", 
        "coaching_plan": "구체적인 집중 코칭 계획"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        return json.loads(json_output)
        
    except Exception as e:
        print(f"집중 코칭 분석 LLM 호출 실패: {e}")
        return {
            "focus_coaching_needed": False,
            "issue_summary": "",
            "root_cause_analysis": "",
            "risk_factors": "", 
            "coaching_plan": ""
        }

def call_llm_for_individual_result(basic_info: Dict, growth_analysis: Dict, 
                                 performance_data: Dict, report_type: str) -> Dict:
    """개인용 결과 생성을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "")
    
    system_prompt = """
    당신은 직원 개인에게 성장 피드백을 제공하는 HR 전문가입니다.
    격려하고 동기부여하는 톤으로 개인 친화적인 피드백을 작성해주세요.
    
    작성 원칙:
    - "당신의", "귀하의" 등 개인 대상 표현 사용
    - 긍정적이고 건설적인 표현
    - 구체적인 개선 방안 제시
    - 격려와 동기부여 포함
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    period_text = "분기" if report_type == "quarterly" else "연간"
    
    human_prompt = f"""
    <기본 정보>
    이름: {emp_name}
    평가 유형: {period_text}
    </기본 정보>

    <성장 분석 결과>
    성장 포인트: {growth_analysis.get('growth_points', [])}
    보완 영역: {growth_analysis.get('improvement_areas', [])}
    추천 활동: {growth_analysis.get('recommended_activities', [])}
    </성장 분석 결과>

    <성과 데이터>
    달성률: {performance_data.get('ai_achievement_rate', 0)}%
    기여도: {performance_data.get('contribution_rate', 0)}%
    </성과 데이터>

    JSON 응답:
    {{
        "growth_points": [
            "당신의 강점을 개인 친화적 톤으로 표현한 항목 1",
            "강점 항목 2",
            "강점 항목 3"
        ],
        "improvement_areas": [
            "발전 가능성이 큰 영역을 건설적 톤으로 표현한 항목 1", 
            "보완 영역 2"
        ],
        "recommended_activities": [
            "구체적이고 실행 가능한 추천 활동 1",
            "추천 활동 2",
            "추천 활동 3"
        ]
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        return json.loads(json_output)
        
    except Exception as e:
        print(f"개인용 결과 생성 LLM 호출 실패: {e}")
        return {
            "growth_points": ["결과 생성 중 오류 발생"],
            "improvement_areas": ["결과 생성 중 오류 발생"],
            "recommended_activities": ["결과 생성 중 오류 발생"]
        }

def call_llm_for_overall_comment(basic_info: Dict, performance_data: Dict, 
                                peer_talk_data: Dict, fourp_data: Dict, 
                                collaboration_data: Dict, growth_analysis: Dict,
                                module7_score_data: Dict, module9_final_data: Dict,
                                report_type: str) -> str:
    """전체 레포트 종합 총평 생성을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "")
    cl = basic_info.get("cl", "CL2")
    position = basic_info.get("position", "직책 정보 없음")
    
    # 4P 점수 추출
    fourp_scores = {
        "passionate": fourp_data.get("passionate", {}).get("score", 3.0),
        "proactive": fourp_data.get("proactive", {}).get("score", 3.0), 
        "professional": fourp_data.get("professional", {}).get("score", 3.0),
        "people": fourp_data.get("people", {}).get("score", 3.0)
    }
    
    fourp_avg = sum(fourp_scores.values()) / len(fourp_scores)
    
    system_prompt = """
    당신은 종합 성과 평가 전문가입니다.
    직원의 모든 평가 결과를 종합하여 전체적인 총평을 작성해주세요.
    
    총평 작성 원칙:
    1. 개인 친화적이고 격려하는 톤 사용
    2. 모든 모듈의 결과를 균형있게 반영
    3. 구체적 성과와 데이터 언급
    4. 향후 성장 방향 제시
    5. 250-300자 분량
    
    반드시 일반적인 문구("열심히 하세요", "앞으로도 화이팅")는 피하고
    구체적인 성과와 개선 방향을 포함해주세요.
    
    결과는 문자열로만 응답하세요.
    """
    
    period_text = "분기" if report_type == "quarterly" else "연간"
    
    # 점수 정보 (연말만)
    score_summary = ""
    if report_type == "annual":
        if module7_score_data.get("score"):
            team_score = module7_score_data.get("score", 0)
            score_summary += f"팀 내 정규화 {team_score}점"
        
        if module9_final_data.get("score"):
            final_score = module9_final_data.get("score", 0)
            ranking = module9_final_data.get("ranking", 0)
            score_summary += f", 부문 정규화 후 최종 {final_score}점(팀 내 {ranking}위)"
    
    human_prompt = f"""
    <종합 평가 데이터>
    직원: {emp_name}({cl} {position})
    평가 기간: {period_text}
    
    📊 성과 결과 (모듈 2):
    - 달성률: {performance_data.get('ai_achievement_rate', 0)}%
    - 기여도: {performance_data.get('contribution_rate', 0)}%
    - 팀 내 달성률 순위: {performance_data.get('ranking', 0)}위
    
    🤝 협업 분석 (모듈 3):
    - 협업률: {collaboration_data.get('collaboration_rate', 0)}%
    - 팀 역할: {collaboration_data.get('team_role', '정보 없음')}
    - 협업 편중도: {collaboration_data.get('collaboration_bias', '보통')}
    
    👥 Peer Talk (모듈 4):
    - 강점: {peer_talk_data.get('strengths', '정보 없음')}
    - 우려사항: {peer_talk_data.get('concerns', '정보 없음')}
    
    🎯 4P 평가 (모듈 6):
    - Passionate: {fourp_scores['passionate']}점
    - Proactive: {fourp_scores['proactive']}점  
    - Professional: {fourp_scores['professional']}점
    - People: {fourp_scores['people']}점
    - 평균: {fourp_avg:.1f}점
    
    {f'📈 점수 평가 (모듈 7,9): {score_summary}' if score_summary else ''}
    
    🚀 성장 제안 (모듈 10):
    - 성장 포인트: {len(growth_analysis.get('growth_points', []))}개
    - 보완 영역: {len(growth_analysis.get('improvement_areas', []))}개
    - 추천 활동: {len(growth_analysis.get('recommended_activities', []))}개
    </종합 평가 데이터>

    위 모든 결과를 종합하여 {emp_name}님께 드리는 {period_text} 종합 총평을 작성해주세요.
    구체적인 성과 수치와 강점을 언급하고, 향후 발전 방향을 제시하는 격려의 메시지로 작성해주세요.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        content = response.content if isinstance(response.content, str) else str(response.content)
        return content.strip()
        
    except Exception as e:
        print(f"종합 총평 LLM 호출 실패: {e}")
        return f"{emp_name}님의 {period_text} 종합 총평 생성 중 오류가 발생했습니다."

def call_llm_for_manager_result(basic_info: Dict, growth_analysis: Dict, 
                              performance_data: Dict, collaboration_data: Dict,
                              focus_coaching_analysis: Dict, focus_coaching_needed: bool) -> Dict:
    """팀장용 결과 생성을 위한 LLM 호출"""
    
    emp_name = basic_info.get("emp_name", "")
    emp_no = basic_info.get("emp_no", "")
    
    system_prompt = """
    당신은 팀장에게 팀원 관리 정보를 제공하는 HR 전문가입니다.
    객관적이고 분석적인 톤으로 관리자 관점의 코칭 정보를 작성해주세요.
    
    작성 원칙:
    - "해당 직원의", "○○○님의" 등 관리 대상 표현 사용
    - 팀 운영과 인사 관리 포커스
    - strengths, improvement_points, collaboration_style, performance_summary는 핵심 키워드 중심으로 간결하게 요약
    - next_quarter_coaching은 구체적이고 실행 가능한 다음 분기 액션 플랜을 1~2문장의 간결한 서술형으로 작성
    
    ⚠️ 중요: 반드시 제공된 JSON 구조를 그대로 사용하세요.
    - emp_no와 name 필드는 정확히 제공된 값으로 설정
    - JSON 구조를 변경하지 마세요
    - 하나의 직원에 대한 하나의 general_coaching 항목만 생성하세요
    
    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <직원 정보>
    사번: {emp_no}
    이름: {emp_name}
    </직원 정보>

    <성장 분석 결과>
    성장 포인트: {growth_analysis.get('growth_points', [])}
    보완 영역: {growth_analysis.get('improvement_areas', [])}
    추천 활동: {growth_analysis.get('recommended_activities', [])}
    </성장 분석 결과>

    <성과 및 협업 데이터>
    달성률: {performance_data.get('ai_achievement_rate', 0)}%
    기여도: {performance_data.get('contribution_rate', 0)}%
    달성률 순위: {performance_data.get('ranking', 0)}위
    협업률: {collaboration_data.get('collaboration_rate', 0)}%
    팀 역할: {collaboration_data.get('team_role', '정보 없음')}
    </성과 및 협업 데이터>

    <집중 코칭 필요성>
    집중 코칭 필요: {focus_coaching_needed}
    집중 코칭 분석: {focus_coaching_analysis if focus_coaching_needed else '해당 없음'}
    </집중 코칭 필요성>

    위 데이터를 바탕으로 다음 JSON 구조로 정확히 응답하세요:

    {{
        "general_coaching": [
            {{
                "emp_no": "{emp_no}",
                "name": "{emp_name}",
                "strengths": "핵심 강점 키워드 1, 키워드 2",
                "improvement_points": "개선 필요 역량 키워드 1, 키워드 2",
                "collaboration_style": "협업 스타일 키워드 (예: 리더형, 조율자형), 특징",
                "performance_summary": "핵심 성과 지표 요약 (예: 달성률 X%, 팀 내 달성률 Y위)",
                "next_quarter_coaching": "다음 분기에 실행할 구체적이고 간결한 코칭 제안. (1~2 문장)"
            }}
        ],
        "focused_coaching": []
    }}

    ⚠️ 주의사항:
    1. emp_no와 name은 반드시 "{emp_no}"와 "{emp_name}"으로 설정
    2. general_coaching은 하나의 항목만 생성
    3. focused_coaching은 빈 배열로 설정 (별도 처리됨)
    4. 상위 4개 항목은 키워드 위주로, next_quarter_coaching은 간결한 서술형으로 작성하세요.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content if isinstance(response.content, str) else str(response.content)
        json_output = _extract_json_from_llm_response(json_output_raw)
        result = json.loads(json_output)
        
        # 🔥 중요: LLM 응답 후 emp_no와 name을 강제로 설정
        if "general_coaching" in result and result["general_coaching"]:
            for coaching_item in result["general_coaching"]:
                coaching_item["emp_no"] = emp_no
                coaching_item["name"] = emp_name
        
        # focused_coaching 처리
        if focus_coaching_needed and focus_coaching_analysis:
            focused_coaching_item = {
                "emp_no": emp_no,
                "name": emp_name,
                "issue_summary": focus_coaching_analysis.get("issue_summary", ""),
                "root_cause_analysis": focus_coaching_analysis.get("root_cause_analysis", ""),
                "risk_factors": focus_coaching_analysis.get("risk_factors", ""),
                "coaching_plan": focus_coaching_analysis.get("coaching_plan", "")
            }
            result["focused_coaching"] = [focused_coaching_item]
        else:
            result["focused_coaching"] = []
            
        return result
        
    except Exception as e:
        print(f"팀장용 결과 생성 LLM 호출 실패: {e}")
        return {
            "general_coaching": [{
                "emp_no": emp_no,
                "name": emp_name,
                "strengths": "결과 생성 중 오류 발생",
                "improvement_points": "결과 생성 중 오류 발생",
                "collaboration_style": "결과 생성 중 오류 발생",
                "performance_summary": "결과 생성 중 오류 발생",
                "next_quarter_coaching": "결과 생성 중 오류 발생"
            }],
            "focused_coaching": []
        }