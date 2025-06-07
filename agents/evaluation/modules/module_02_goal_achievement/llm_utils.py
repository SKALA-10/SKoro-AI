
from db_utils import *

import re
import json 
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv() 

# LangChain LLM 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# --- LLM 클라이언트 인스턴스 (전역 설정) ---
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
print(f"LLM Client initialized with model: {llm_client.model_name}, temperature: {llm_client.temperature}")

# --- LLM 응답에서 JSON 코드 블록 추출 도우미 함수 ---
def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답 텍스트에서 ```json ... ``` 블록만 추출합니다."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip() # JSON 내용만 반환하고 양쪽 공백 제거
    return text.strip()


def call_llm_for_task_contribution(task_summary_text: str) -> Dict:
    print(f"LLM Call (Task Contribution): '{task_summary_text[:30]}...'")

    system_prompt = """
    당신은 SK 조직의 성과 평가 전문가입니다.
    아래 Task 요약 내용을 보고, 해당 Task가 전체 프로젝트/팀 목표에 얼마나 기여했는지, 
    그리고 업무의 난이도, 완성도, 중요도를 종합적으로 고려하여 100점 만점으로 기여도 점수를 산정하고, 
    간략한 분석 코멘트를 생성해주세요.

    평가 시 다음을 고려합니다:
    - Task의 복잡성과 달성 난이도
    - Task 결과물의 품질과 완성도
    - Task가 다음 단계 또는 다른 팀원에게 미친 긍정적 영향 (선행 조건 해결 등)
    - Task가 팀 목표 달성에 기여한 정도

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """
    
    human_prompt = f"""
    <Task 요약>
    {task_summary_text}
    </Task 요약>

    JSON 응답:
    {{
        "기여도 점수": [기여도 점수 (0-100점, 소수점 첫째 자리까지)],
        "분석 코멘트": "[Task에 대한 분석 코멘트]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"task_summary_text": task_summary_text})
        json_output_raw = response.content

        json_output = _extract_json_from_llm_response(json_output_raw)

        llm_parsed_data = json.loads(json_output)

        score = llm_parsed_data.get("기여도 점수") 
        comment = llm_parsed_data.get("분석 코멘트") 


        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            raise ValueError(f"LLM 반환 점수 {score}가 유효하지 않습니다.")
        if not isinstance(comment, str) or not comment:
            raise ValueError(f"LLM 반환 코멘트 {comment}가 유효하지 않습니다.")

        return {"score": round(float(score), 2), "comment": comment}
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"score": 0.0, "comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"score": 0.0, "comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"score": 0.0, "comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


def call_llm_for_task_achievement(target_level_text: str, task_performance_text: str, grade_definitions: Dict) -> Dict:
    
    system_prompt = """
    당신은 SK 조직의 성과 평가 전문가입니다.
    아래 Task 목표와 실제 성과를 비교하여, Task의 달성률(0-100점 이상)과 적절한 등급(S, A, B, C, D)을 판단하고,
    상세 분석 코멘트를 생성해주세요.

    평가 기준은 다음과 같습니다:
    - 달성률은 0점부터 시작하며, 100점을 초과하여 목표 초과 달성을 나타낼 수 있습니다. (예: 100.1% 이상)
    - 등급은 S, A, B, C, D 중 하나여야 합니다.

    <등급 정의 (LLM 참고용)>
    """
    for grade, desc in grade_definitions.items():
        system_prompt += f"- {grade} 등급: {desc}\n"
    system_prompt += "</등급 정의>\n"
    system_prompt += "결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다."

    human_prompt = f"""
    <Task 목표>
    {target_level_text}
    </Task 목표>

    <실제 성과>
    {task_performance_text}
    </실제 성과>

    JSON 응답:
    {{
      "달성률": [달성률 (0-100점 이상)],
      "등급": "[S, A, B, C, D 중 하나]",
      "상세 분석 코멘트": "[Task에 대한 상세 분석 코멘트]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"target_level_text": target_level_text, "task_performance_text": task_performance_text, "grade_definitions": grade_definitions})
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        rate = llm_parsed_data.get("달성률") 
        grade = llm_parsed_data.get("등급") 
        analysis = llm_parsed_data.get("상세 분석 코멘트") 

        # 수정된 부분: 달성률 유효성 검사 상한 제거
        if not isinstance(rate, (int, float)) or not (0 <= rate): 
            raise ValueError(f"LLM 반환 달성률 {rate}가 유효하지 않습니다 (0 이상이어야 합니다).")
        if grade not in ["S", "A", "B", "C", "D"]:
            raise ValueError(f"LLM 반환 등급 {grade}가 유효하지 않습니다.")
        if not isinstance(analysis, str) or not analysis:
            raise ValueError(f"LLM 반환 분석 코멘트 {analysis}가 유효하지 않습니다.")

        return {"grade": grade, "rate": round(float(rate), 2), "analysis": analysis}

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"grade": "D", "rate": 0.0, "analysis": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"grade": "D", "rate": 0.0, "analysis": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"grade": "D", "rate": 0.0, "analysis": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


def call_llm_for_overall_contribution_summary(all_individual_task_results: List[Dict], emp_name: str, emp_no: str) -> Dict: 
    print(f"LLM Call (Overall Contribution Summary): '{emp_name} ({emp_no})' Task {len(all_individual_task_results)}개 기반 요약 요청.") 

    task_details_str = ""
    for task in all_individual_task_results:
        task_details_str += f"- Task: {task.get('task_name')} (ID: {task.get('task_id')})\n"
        task_details_str += f"  Summary: {task.get('task_summary', task.get('task_performance', ''))}\n"
        if task.get('ai_contribution_score') is not None:
            task_details_str += f"  AI 기여도: {task.get('ai_contribution_score')}점\n"
        if task.get('ai_achievement_rate') is not None:
            task_details_str += f"  AI 달성률: {task.get('ai_achievement_rate')}%\n"
        if task.get('ai_assessed_grade'):
            task_details_str += f"  AI 등급: {task.get('ai_assessed_grade')}\n"
        task_details_str += "\n"

    system_prompt = """
    당신은 SK 조직의 HR 성과 전문가입니다.
    아래 제공된 개인의 모든 Task 정보, Task Summary, 그리고 AI가 분석한 개별 Task 기여도/달성률 점수를 종합적으로 고려하여,
    이 개인의 총체적인 기여도 점수 (팀 내 상대 비율, 0-100%)를 추정하고,
    이름과 사번을 명시하며 개인의 전체적인 성과와 기여에 대한 간략한 종합 코멘트를 생성해주세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    직원 이름을 언급할 때는 반드시 "이름(사번)님" 형태로 작성해주세요.
    """

    human_prompt = f"""
    <개인 Task 종합 정보>
    {task_details_str}
    </개인 Task 종합 정보>
    <평가 대상 개인 정보>
    이름: {emp_name}
    사번: {emp_no}
    </평가 대상 개인 정보>

    JSON 응답:
    {{
      "total_contribution": [개인의 총체적인 기여도 점수 (0-100점)],
      "comment": "[{emp_name}({emp_no})님의 전체 성과와 기여에 대한 종합 코멘트]",
      "average_rate": [Task 달성률들의 평균 또는 종합적인 달성률 추정 (0-100점 이상)]
    }}
    """


    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"all_individual_task_results": all_individual_task_results, "emp_name": emp_name, "emp_no": emp_no})
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        total_contribution = llm_parsed_data.get("total_contribution")
        comment = llm_parsed_data.get("comment")
        average_rate = llm_parsed_data.get("average_rate")

        if not isinstance(total_contribution, (int, float)) or not (0 <= total_contribution <= 100):
            raise ValueError(f"LLM 반환 총 기여도 {total_contribution}가 유효하지 않습니다.")
        if not isinstance(comment, str) or not comment:
            raise ValueError(f"LLM 반환 코멘트 {comment}가 유효하지 않습니다.")
        if not isinstance(average_rate, (int, float)) or not (0 <= average_rate): # 0-120점 -> 0점 이상으로 수정
            raise ValueError(f"LLM 반환 평균 달성률 {average_rate}가 유효하지 않습니다 (0 이상이어야 합니다).")

        return {"total_contribution": round(float(total_contribution), 2), "comment": comment, "average_rate": round(float(average_rate), 2)}

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"total_contribution": 0.0, "comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}...", "average_rate": 0.0}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"total_contribution": 0.0, "comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}...", "average_rate": 0.0}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"total_contribution": 0.0, "comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}...", "average_rate": 0.0}


def call_llm_for_team_overall_analysis(all_team_kpis_results: List[Dict]) -> Dict:
    print(f"LLM Call (Team Overall Analysis): KPI {len(all_team_kpis_results)}개 기반 분석 요청.")

    kpi_details_str = ""
    for kpi in all_team_kpis_results:
        kpi_details_str += f"- KPI: {kpi.get('kpi_name')} (ID: {kpi.get('team_kpi_id')})\n"
        kpi_details_str += f"  Description: {kpi.get('kpi_description')}\n"
        if kpi.get('ai_kpi_overall_progress_rate') is not None:
            kpi_details_str += f"  AI 진행률: {kpi.get('ai_kpi_overall_progress_rate')}%\n"
        if kpi.get('ai_kpi_analysis_comment'):
            kpi_details_str += f"  AI 코멘트: {kpi.get('ai_kpi_analysis_comment')}\n"
        kpi_details_str += "\n"

    system_prompt = """
    당신은 SK 조직의 고위 경영진을 위한 팀 성과 분석 전문가입니다.
    아래 제공된 팀의 KPI 정보, 설명, 그리고 AI가 분석한 각 KPI의 진행률 및 코멘트를 종합적으로 검토하여,
    이 팀의 전반적인 목표 달성률을 추정하고, 팀 성과의 주요 특징과 개선점에 대한 분석 코멘트를 생성해주세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """

    human_prompt = f"""
    <팀 KPI 종합 정보>
    {kpi_details_str}
    </팀 KPI 종합 정보>

    JSON 응답:
    {{
      "overall_rate": [팀 전체의 목표 달성률 추정 (0-100점)],
      "comment": "[팀 성과에 대한 전반적인 분석 코멘트]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"all_team_kpis_results": all_team_kpis_results})
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        overall_rate = llm_parsed_data.get("overall_rate")
        comment = llm_parsed_data.get("comment")

        if not isinstance(overall_rate, (int, float)) or not (0 <= overall_rate <= 100):
            raise ValueError(f"LLM 반환 전체 달성률 {overall_rate}가 유효하지 않습니다.")
        if not isinstance(comment, str) or not comment:
            raise ValueError(f"LLM 반환 코멘트 {comment}가 유효하지 않습니다.")

        return {"overall_rate": round(float(overall_rate), 2), "comment": comment}

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"overall_rate": 0.0, "comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"overall_rate": 0.0, "comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"overall_rate": 0.0, "comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


def call_llm_for_kpi_relative_contribution(kpi_analysis_input: Dict) -> Dict:
    kpi_goal = kpi_analysis_input.get("kpi_goal", "알 수 없는 목표")
    kpi_description = kpi_analysis_input.get("kpi_description", "")
    team_tasks = kpi_analysis_input.get("team_members_tasks", [])
    
    print(f"LLM Call (KPI Relative Contribution): '{kpi_goal[:30]}...' KPI 내 개인별 상대 기여도 분석 요청.")
    
    actual_emp_nos_in_kpi = sorted(list(set(task.get('emp_no') for task in team_tasks if task.get('emp_no'))))

    system_prompt = """
    당신은 팀 KPI 성과에 대한 개인별 기여도를 평가하는 전문가입니다.
    아래는 특정 팀 KPI의 목표, 설명, 그리고 이 KPI에 기여한 팀원들의 Task 상세 내용 및 AI가 분석한 개별 Task 기여도 점수입니다.
    
    이 정보를 종합적으로 검토하여 다음을 수행하세요:
    1. 이 KPI에 대한 각 개인의 **상대적인 기여도 점수 (총합 100%)**를 판단하세요.
       - 반환하는 JSON의 `individual_relative_contributions_in_kpi` 딕셔너리에는 아래 <실제 팀원 사번 목록>에 있는 모든 사번에 대해 기여도를 포함해야 합니다.
       - 각 개인의 기여도 점수(0-100점)는 소수점 두 자리까지 허용합니다.
       - 어떤 팀원의 기여도가 0%이더라도 해당 사번과 0점을 명시적으로 포함해야 합니다.
       - 모든 팀원의 기여도 합계가 100%가 되도록 조정해야 합니다.
    2. KPI 전체의 진행 상황에 대한 간략한 분석 코멘트를 생성하세요.

    평가 시 다음을 고려해야 합니다:
    - 각 Task의 내용이 KPI 목표 달성에 얼마나 중요한가?
    - 각 Task의 AI 기여도 점수는 어떤 의미인가? (개별 Task의 품질 및 중요도)
    - 팀원 간 Task의 상호 의존성, 선행/후행 관계, 협업 기여도
    - 특정 팀원이 여러 Task를 수행했거나, 더 중요한 Task를 수행했는가?
    - 결과물 JSON에 불필요한 텍스트를 포함하지 마세요.
    - 직원 이름을 언급할 때는 반드시 "이름(사번)님" 형태로 작성해주세요.


    결과는 다음 JSON 형식으로만 응답해주세요:
    """

    team_tasks_str = ""
    for task in team_tasks:
        emp_name = task.get('emp_name', '이름없음') 
        emp_no = task.get('emp_no', '사번없음')
        team_tasks_str += f"- 팀원: {emp_name}({emp_no})님, Task: {task.get('task_name')}\n" 
        team_tasks_str += f"  요약: {task.get('task_summary')}\n"
        if task.get('ai_contribution_score_from_individual_analysis') is not None:
            team_tasks_str += f"  개별 AI 기여도 점수 (참고용): {task.get('ai_contribution_score_from_individual_analysis')}점\n"
        team_tasks_str += "\n"


    individual_contributions_json_example = ",\n".join([f'"{emp_no}": [상대 기여도 (0-100점)]' for emp_no in actual_emp_nos_in_kpi])
    if not individual_contributions_json_example:
        individual_contributions_json_example = '"EMP_NO_X": [상대 기여도 (0-100점)]'

    human_prompt = f"""
    <팀 KPI 목표>
    {kpi_goal}
    </팀 KPI 목표>
    <팀 KPI 설명>
    {kpi_description}
    </팀 KPI 설명>
    <팀원 Task 정보>
    {team_tasks_str}
    </팀원 Task 정보>
    <실제 팀원 사번 목록>
    {', '.join(actual_emp_nos_in_kpi) if actual_emp_nos_in_kpi else '없음'}
    </실제 팀원 사번 목록>

    JSON 응답:
    {{
      "kpi_overall_rate": [KPI 전체의 진행 상황에 대한 점수 (0-100점)],
      "kpi_analysis_comment": "[KPI 전체 진행 상황에 대한 분석 코멘트]",
      "individual_relative_contributions_in_kpi": {{
        {individual_contributions_json_example}
      }}
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"kpi_analysis_input": kpi_analysis_input})
        json_output_raw = response.content
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        kpi_overall_rate = llm_parsed_data.get("kpi_overall_rate")
        kpi_analysis_comment = llm_parsed_data.get("kpi_analysis_comment")
        individual_relative_contributions_raw_from_llm = llm_parsed_data.get("individual_relative_contributions_in_kpi")

        if not isinstance(kpi_overall_rate, (int, float)) or not (0 <= kpi_overall_rate <= 100):
            raise ValueError(f"LLM 반환 KPI 전체 진행률 {kpi_overall_rate}가 유효하지 않습니다.")
        if not isinstance(kpi_analysis_comment, str) or not kpi_analysis_comment:
            raise ValueError(f"LLM 반환 KPI 분석 코멘트 {kpi_analysis_comment}가 유효하지 않습니다.")
        if not isinstance(individual_relative_contributions_raw_from_llm, dict):
            raise ValueError(f"LLM 반환 개인 상대 기여도 형식 {individual_relative_contributions_raw_from_llm}가 유효하지 않습니다.")
        
        # --- 파싱 로직 보강: LLM이 반환한 사번 외의 사번 처리 및 합계 검증 ---
        final_relative_contributions = {}
        for emp_no in actual_emp_nos_in_kpi:
            final_relative_contributions[emp_no] = 0.0
        
        for emp_no_from_llm, score in individual_relative_contributions_raw_from_llm.items():
            if emp_no_from_llm in final_relative_contributions and isinstance(score, (int, float)):
                final_relative_contributions[emp_no_from_llm] = round(float(score), 2)
            else:
                print(f"Warning: LLM이 예상치 못한 사번 '{emp_no_from_llm}'를 반환했거나 점수가 유효하지 않아 무시됩니다. 점수: {score}")

        total_relative_sum = sum(final_relative_contributions.values())
        if total_relative_sum > 0 and not (99.9 <= total_relative_sum <= 100.1):
            print(f"Warning: 개인 상대 기여도 합계가 100%와 다릅니다: {total_relative_sum}%. 재조정 시도.")
            adjustment_factor = 100.0 / total_relative_sum if total_relative_sum > 0 else 1.0
            adjusted_contributions = {k: round(v * adjustment_factor, 2) for k, v in final_relative_contributions.items()}
            final_relative_contributions = adjusted_contributions
            print(f"재조정된 기여도: {final_relative_contributions}")
            
        return {
            "kpi_overall_rate": round(float(kpi_overall_rate), 2),
            "kpi_analysis_comment": kpi_analysis_comment,
            "individual_relative_contributions_in_kpi": final_relative_contributions
        }
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {
            "kpi_overall_rate": 0.0,
            "kpi_analysis_comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}...",
            "individual_relative_contributions_in_kpi": {}
        }
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {
            "kpi_overall_rate": 0.0,
            "kpi_analysis_comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}...",
            "individual_relative_contributions_in_kpi": {}
        }
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {
            "kpi_overall_rate": 0.0,
            "kpi_analysis_comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}...",
            "individual_relative_contributions_in_kpi": {}
        }
    


def call_llm_for_individual_contribution_reason_comment(
    task_info: Dict, 
    adjusted_contribution_score: float, 
    kpi_goal: str, 
    kpi_overall_comment: str) -> Dict:
    """
    개인의 Task 상세 내역, 조정된 기여도 점수, KPI 맥락을 종합하여
    Task에 대한 기여도 근거 코멘트를 생성합니다.
    """
    emp_name = task_info.get("emp_name", "이름 없음") # Task info에 emp_name이 없다면 db에서 조회해야 함
    emp_no = task_info.get("emp_no", "사번 없음")
    task_name = task_info.get("task_name", "알 수 없는 Task")
    task_summary_text = task_info.get("task_summary", task_info.get("task_performance", "상세 내용 없음"))

    print(f"LLM Call (Individual Contribution Reason): '{emp_name} ({emp_no})'의 '{task_name[:30]}...' Task 근거 요청.")

    system_prompt = """
    당신은 SK 조직의 성과 평가 전문가이자 명확한 근거를 제시하는 분석가입니다.
    아래 제공된 개인의 특정 Task 상세 내용, 이 Task가 속한 KPI의 목표, 그리고 팀 전체에 대한 KPI 분석 코멘트를 종합적으로 고려하여,
    이 Task의 최종 조정된 기여도 점수(KPI 내 상대적 기여도)가 왜 그렇게 산정되었는지에 대한 구체적이고 복합적인 근거 코멘트를 작성해주세요.

    코멘트는 다음 요소를 포함해야 합니다:
    - Task 자체의 내용과 중요도 (Task Summary 기반)
    - LLM이 판단한 KPI 내 상대적 기여도 점수 (제시된 점수 활용)
    - 이 Task가 팀 KPI 목표 달성에 어떻게 기여했는지 (KPI 목표, 전체 KPI 코멘트 기반)
    - Task 간의 상호 관계나 협업 등의 맥락이 기여도에 미친 영향 (제공된 정보 내에서 추론)
    - 직원 이름을 언급할 때는 반드시 "이름(사번)님" 형태로 작성해주세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """

    human_prompt = f"""
    <Task 상세 정보>
    이름: {emp_name}
    사번: {emp_no}
    Task 이름: {task_name}
    Task 요약/성과: {task_summary_text}
    조정된 기여도 점수: {adjusted_contribution_score}점
    </Task 상세 정보>

    <KPI 정보>
    KPI 목표: {kpi_goal}
    KPI 전체 분석 코멘트: {kpi_overall_comment}
    </KPI 정보>

    JSON 응답:
    {{
      "comment_reason": "[{emp_name}({emp_no})님의 해당 Task에 대한 구체적 근거 코멘트]"
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({
            "task_info": task_info, 
            "adjusted_contribution_score": adjusted_contribution_score, 
            "kpi_goal": kpi_goal, 
            "kpi_overall_comment": kpi_overall_comment
        })
        json_output_raw = response.content
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        comment_reason = llm_parsed_data.get("comment_reason")

        if not isinstance(comment_reason, str) or not comment_reason:
            raise ValueError(f"LLM 반환 근거 코멘트 {comment_reason}가 유효하지 않습니다.")

        return {"comment": comment_reason}
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"comment": f"AI 근거 생성 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 응답: {json_output}")
        return {"comment": f"AI 근거 생성 실패: 유효성 오류 - {json_output[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"comment": f"AI 근거 생성 실패: 예기치 않은 오류 - {str(e)[:100]}..."}
