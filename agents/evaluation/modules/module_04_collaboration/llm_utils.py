# ================================================================
# llm_utils_module4.py - 모듈 4 LLM 관련 유틸리티
# ================================================================

import re
import json
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

from agents.evaluation.modules.module_04_collaboration.db_utils import *

# LLM 클라이언트 초기화
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답 텍스트에서 ```json ... ``` 블록만 추출합니다."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def _get_llm_content(response):
    content = response.content
    if isinstance(content, list):
        # 리스트면 문자열만 추출
        content = "\n".join([str(x) for x in content if isinstance(x, str)])
    return content

# ================================================================
# LLM 호출 함수들
# ================================================================

def call_llm_for_collaboration_detection(task_summary: str, task_name: str, potential_collaborators: List[str], emp_name: str) -> Dict:
    """Task Summary에서 실제 협업 관계를 감지합니다."""
    
    system_prompt = """
    당신은 SK 조직의 업무 협업 분석 전문가입니다.
    주어진 Task Summary 내용을 분석하여 실제로 다른 동료와 협업했는지 판단하고,
    협업한 경우 구체적으로 누구와 협업했는지 식별해주세요.

    분석 기준:
    - "함께", "협력", "지원", "도움", "협업", "공동" 등의 키워드 존재
    - 다른 사람의 이름이나 역할 언급
    - 회의, 논의, 검토 등 상호작용 활동 언급
    - 단순한 보고나 개별 작업은 협업으로 간주하지 않음

    결과는 JSON 형식으로만 응답해주세요.
    """

    potential_collaborators_str = ", ".join(potential_collaborators) if potential_collaborators else "없음"
    
    human_prompt = f"""
    <분석 대상 Task>
    담당자: {emp_name}
    Task 이름: {task_name}
    Task 요약: {task_summary}
    </분석 대상 Task>

    <잠재적 협업자 목록>
    {potential_collaborators_str}
    </잠재적 협업자 목록>

    JSON 응답:
    {{
        "is_collaboration": [true/false - 실제 협업 여부],
        "collaborators": ["협업자 사번 리스트"],
        "collaboration_description": "[협업 내용 간단 설명]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = _get_llm_content(response)
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        is_collaboration = llm_parsed_data.get("is_collaboration", False)
        collaborators = llm_parsed_data.get("collaborators", [])
        description = llm_parsed_data.get("collaboration_description", "")
        
        return {
            "is_collaboration": is_collaboration,
            "collaborators": collaborators if isinstance(collaborators, list) else [],
            "description": description
        }
        
    except Exception as e:
        print(f"LLM 협업 감지 오류: {e}")
        return {"is_collaboration": False, "collaborators": [], "description": "분석 실패"}

def call_llm_for_team_role_analysis(task_summaries: List[str], emp_name: str, emp_no: str) -> Dict:
    """개인의 Task Summary들을 종합하여 팀 내 역할을 분석합니다."""
    
    task_summaries_text = "\n".join([f"- {summary}" for summary in task_summaries])
    
    system_prompt = """
    당신은 SK 조직의 역할 분석 전문가입니다.
    개인의 Task Summary들을 분석하여 해당 직원의 주요 업무 내용과 팀 내 역할 유형을 파악해주세요.

    역할 유형 분류:
    - 핵심 개발자: 주요 기술 개발 담당
    - 기획 리더: 프로젝트 기획 및 방향성 설정
    - 조율자: 팀 간 협업 및 일정 관리
    - 품질 관리자: 테스트, 검증, 품질 보증
    - 분석 전문가: 데이터 분석, 리서치
    - 지원 전문가: 기술 지원, 문제 해결
    - 독립형 전문가: 전문 영역 독립 수행

    결과는 JSON 형식으로만 응답해주세요.
    """

    human_prompt = f"""
    <분석 대상>
    이름: {emp_name}
    사번: {emp_no}
    Task Summary 목록:
    {task_summaries_text}
    </분석 대상>

    JSON 응답:
    {{
        "main_work_content": "[주요 업무 내용 (한줄)]",
        "role_type": "[역할 유형]",
        "team_role": "[주요 업무 내용 | 역할 유형]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = _get_llm_content(response)
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        return {
            "main_work_content": llm_parsed_data.get("main_work_content", "업무 내용 분석 실패"),
            "role_type": llm_parsed_data.get("role_type", "분석 실패"),
            "team_role": llm_parsed_data.get("team_role", "분석 실패")
        }
        
    except Exception as e:
        print(f"LLM 역할 분석 오류: {e}")
        return {
            "main_work_content": "분석 실패",
            "role_type": "분석 실패", 
            "team_role": "분석 실패"
        }

def call_llm_for_collaboration_bias_analysis(collaboration_data: Dict, emp_name: str, emp_no: str) -> Dict:
    """협업 편중도를 신뢰성 있게 분석합니다."""
    
    total_tasks = collaboration_data.get("total_tasks", 0)
    collaboration_tasks = collaboration_data.get("collaboration_tasks", 0)
    collaborator_counts = collaboration_data.get("collaborator_counts", {})
    dependency_metrics = collaboration_data.get("dependency_metrics", {})
    
    system_prompt = """
    당신은 SK 조직의 협업 편중도 분석 전문가입니다.
    제공된 협업 데이터를 기반으로 해당 직원의 협업 편중도를 분석해주세요.

    분석 기준:
    1. 높음(과의존 위험): 
       - 다른 팀원들이 이 사람에게 과도하게 의존
       - 핵심 업무를 혼자 담당하여 병목 위험
       - 협업 비율이 매우 높고 다른 사람들의 업무 진행에 필수적
    
    2. 보통(적절한 분산):
       - 협업과 독립 업무의 균형이 적절
       - 대체 가능한 구조로 운영
       - 다른 팀원들과 고르게 협업
    
    3. 낮음(협업 부족):
       - 협업 참여도가 낮음
       - 다른 팀원들과의 연결점 부족
       - 주변부 역할만 담당

    결과는 JSON 형식으로만 응답해주세요.
    """

    collaboration_info = f"""
    총 Task 수: {total_tasks}
    협업 Task 수: {collaboration_tasks}
    협업률: {(collaboration_tasks/total_tasks*100) if total_tasks > 0 else 0:.1f}%
    협업자별 횟수: {collaborator_counts}
    의존도 지표: {dependency_metrics}
    """

    human_prompt = f"""
    <분석 대상>
    이름: {emp_name}
    사번: {emp_no}
    협업 데이터:
    {collaboration_info}
    </분석 대상>

    JSON 응답:
    {{
        "bias_level": "[높음/보통/낮음]",
        "bias_description": "[편중도 상세 설명]",
        "risk_assessment": "[위험도 평가 및 이유]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = _get_llm_content(response)
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        bias_level = llm_parsed_data.get("bias_level", "보통")
        if bias_level not in ["높음", "보통", "낮음"]:
            bias_level = "보통"
            
        return {
            "bias_level": bias_level,
            "bias_description": llm_parsed_data.get("bias_description", ""),
            "risk_assessment": llm_parsed_data.get("risk_assessment", "")
        }
        
    except Exception as e:
        print(f"LLM 편중도 분석 오류: {e}")
        return {
            "bias_level": "보통",
            "bias_description": "분석 실패",
            "risk_assessment": "분석 실패"
        }

def call_llm_for_peer_talk_summary(peer_talk_content: str, emp_name: str) -> str:
    """Peer Talk 내용을 한 줄로 요약합니다."""
    
    system_prompt = """
    당신은 SK 조직의 동료평가 요약 전문가입니다.
    주어진 Peer Talk 내용(강점, 우려, 협업 관찰)을 핵심만 추려서 한 줄로 요약해주세요.
    
    요약 형식: "핵심 강점 키워드, 주요 우려사항"
    예시: "리더십 강함, 완벽주의 성향 주의"

    따옴표 없이 내용만 응답해주세요.
    """

    human_prompt = f"""
    <Peer Talk 내용>
    {peer_talk_content}
    </Peer Talk 내용>

    한 줄 요약:
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        summary = _get_llm_content(response).strip()
        
        # 따옴표 제거 처리
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        
        return summary if summary else f"{emp_name} 동료평가 요약 없음"
        
    except Exception as e:
        print(f"LLM Peer Talk 요약 오류: {e}")
        return f"{emp_name} 동료평가 분석 실패"

def call_llm_for_overall_evaluation(collaboration_analysis: Dict, emp_name: str, emp_no: str) -> str:
    """개인의 종합 협업 평가를 생성합니다."""
    
    system_prompt = """
    당신은 SK 조직의 협업 평가 전문가입니다.
    제공된 협업 분석 데이터를 바탕으로 해당 직원의 협업 스타일과 기여도를 포함한
    2-3줄의 간단한 종합 평가를 작성해주세요.

    포함 요소:
    - 협업 스타일 (리더형, 서포터형, 독립형 등)
    - 팀 기여도 (높음, 보통, 개선 필요 등)
    - 개선 제안 (필요시)
    """

    analysis_text = f"""
    팀 내 역할: {collaboration_analysis.get('team_role', '')}
    협업률: {collaboration_analysis.get('collaboration_rate', 0)}%
    핵심 협업자: {', '.join(collaboration_analysis.get('key_collaborators', []))}
    협업 편중도: {collaboration_analysis.get('collaboration_bias', '')}
    """

    human_prompt = f"""
    <협업 분석 데이터>
    이름: {emp_name}({emp_no})
    {analysis_text}
    </협업 분석 데이터>

    종합 평가 (2-3줄):
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        evaluation = _get_llm_content(response).strip()
        return evaluation if evaluation else f"{emp_name}({emp_no})님의 협업 평가를 완료하지 못했습니다."
        
    except Exception as e:
        print(f"LLM 종합 평가 오류: {e}")
        return f"{emp_name}({emp_no})님의 협업 분석에 오류가 발생했습니다."

def call_llm_for_team_summary(collaboration_matrix: List[Dict], team_summary_stats: Optional[Dict] = None) -> str:
    """팀 협업 매트릭스 전체를 기반으로 종합적인 팀 요약 코멘트를 생성합니다."""
    
    # team_summary_stats가 제공되지 않은 경우 collaboration_matrix에서 계산
    if team_summary_stats is None:
        total_members = len(collaboration_matrix)
        avg_collaboration_rate = sum([item["collaboration_rate"] for item in collaboration_matrix]) / total_members if total_members > 0 else 0
        avg_contribution_score = sum([item["avg_contribution_score"] for item in collaboration_matrix]) / total_members if total_members > 0 else 0
        high_bias_members = [item["name"] for item in collaboration_matrix if "과의존 위험" in item.get("collaboration_bias", "")]
        
        team_summary_stats = {
            "avg_collaboration_rate": avg_collaboration_rate,
            "avg_contribution_score": avg_contribution_score,
            "high_bias_members": high_bias_members,
            "total_members": total_members
        }
    
    system_prompt = """
    당신은 SK조직의 팀 협업 분석 전문가입니다.
    제공된 팀의 협업 분석 데이터를 종합하여, 팀의 전반적인 협업 현황에 대한 심층적인 분석 코멘트를 생성해주세요.
    단순히 수치를 나열하는 것을 넘어, 팀의 협업 패턴, 강점, 약점, 그리고 개선점을 구체적으로 제시해야 합니다.
    긍정적인 부분과 부정적인 부분을 균형있게 다루어 주세요.
    결과는 2-3 문장의 완성된 코멘트 형태로 응답해주세요.
    """

    matrix_summary_list = []
    for member in collaboration_matrix:
        summary_line = (
            f"- {member['name']}: "
            f"역할({member['team_role']}), "
            f"협업률({member['collaboration_rate']}), "
            f"기여도({member['avg_contribution_score']}), "
            f"편중도({member['collaboration_bias']})"
        )
        matrix_summary_list.append(summary_line)
    matrix_summary = "\n".join(matrix_summary_list)

    human_prompt = f"""
    <팀 협업 분석 데이터>
    - 전체 평균 협업률: {team_summary_stats['avg_collaboration_rate']:.1f}%
    - 전체 평균 기여도: {team_summary_stats['avg_contribution_score']:.1f}점
    - 과의존 위험 멤버: {', '.join(team_summary_stats['high_bias_members']) if team_summary_stats['high_bias_members'] else '없음'}
    - 팀원별 분석 요약:
{matrix_summary}
    </팀 협업 분석 데이터>

    위 데이터를 바탕으로 팀 전체의 협업 상태를 분석하는 종합 코멘트를 작성해주세요.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        summary = _get_llm_content(response).strip()
        return summary if summary else "팀 협업 종합 분석 코멘트 생성에 실패했습니다."
        
    except Exception as e:
        print(f"LLM 팀 요약 생성 오류: {e}")
        return "팀 협업 종합 분석 중 오류가 발생했습니다."