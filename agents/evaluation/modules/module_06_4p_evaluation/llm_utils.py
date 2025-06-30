# ================================================================
# llm_utils.py - LLM 관련 유틸리티
# ================================================================

import re
import json
import hashlib
import os
from typing import Dict, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from db_utils import fetch_evaluation_criteria_from_db

# ================================================================
# LLM 클라이언트 설정
# ================================================================

llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"LLM Client initialized: {llm_client.model_name}")


def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ================================================================
# 파일 기반 캐시 관리
# ================================================================

def get_cache_file_path() -> str:
    """캐시 파일 경로 반환"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
    cache_dir = os.path.join(project_root, 'data', 'cache')
    
    # 캐시 디렉토리가 없으면 생성
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, 'evaluation_criteria_cache.json')


def load_cache_from_file() -> Dict:
    """파일에서 캐시 로드"""
    cache_file = get_cache_file_path()
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                print(f"✅ 파일 캐시 로드됨: {cache_file}")
                return cache_data
        except Exception as e:
            print(f"⚠️ 캐시 파일 로드 실패: {e}")
    
    # 기본 캐시 구조 반환
    return {
        "raw_text": None,
        "raw_text_hash": None,
        "parsed_criteria": None,
        "last_updated": None
    }


def save_cache_to_file(cache_data: Dict) -> bool:
    """캐시를 파일에 저장"""
    cache_file = get_cache_file_path()
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 캐시 파일 저장됨: {cache_file}")
        return True
    except Exception as e:
        print(f"❌ 캐시 파일 저장 실패: {e}")
        return False


def get_text_hash(text: str) -> str:
    """텍스트의 해시값 계산"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def parse_criteria_with_llm(raw_text: str) -> Dict[str, str]:
    """독립적인 LLM 파싱 함수 - 문자열 입력"""
    
    system_prompt = """
당신은 성과 평가 기준을 분석하는 평가 전문가입니다.
사용자는 Passionate, Proactive, Professional, People 네 가지 항목의 평가 기준을 하나의 텍스트에 모두 작성했습니다.
다만 항목 구분이 명확하지 않을 수 있으므로 문맥을 통해 항목별로 내용을 분리해야 합니다.

당신의 작업 결과는 반드시 아래와 같은 JSON 형식이어야 합니다:

```json
{
  "passionate": "passionate 평가 기준 텍스트...",
  "proactive": "proactive 평가 기준 텍스트...",
  "professional": "professional 평가 기준 텍스트...",
  "people": "people 평가 기준 텍스트..."
}
```
"""

    human_prompt = f"""
다음은 DB에서 가져온 전체 평가 기준 텍스트입니다:

{raw_text}

이 텍스트를 분석하여 4개의 평가 항목으로 나눠주세요.
반드시 JSON 형식으로 응답해주세요.
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt), 
        HumanMessage(content=human_prompt)
    ])

    chain = prompt | llm_client
    
    try:
        response = chain.invoke({})
        content = str(response.content)  # 타입 안전성 확보
        match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
        extracted = match.group(1).strip() if match else content.strip()
        parsed = json.loads(extracted)

        expected_keys = {"passionate", "proactive", "professional", "people"}
        if not expected_keys.issubset(parsed.keys()):
            raise ValueError("4개의 평가 기준 키 중 일부가 누락됨")

        return parsed
        
    except Exception as e:
        print(f"❌ LLM 파싱 실패: {e}")
        raise e


def load_and_cache_evaluation_criteria() -> Dict[str, str]:
    """파일 기반 캐시를 사용하는 평가 기준 로더"""
    
    # 1. 파일에서 캐시 로드
    cache_data = load_cache_from_file()
    
    # 2. DB에서 현재 평가 기준 텍스트 가져오기
    try:
        current_raw_text = fetch_evaluation_criteria_from_db()
        current_hash = get_text_hash(current_raw_text)
        
        print(f"🔍 DB 평가 기준 해시: {current_hash[:8]}...")
        
    except Exception as e:
        print(f"❌ DB 조회 실패: {e}")
        raise e
    
    # 3. 캐시된 데이터와 비교
    cached_hash = cache_data.get("raw_text_hash")
    cached_parsed = cache_data.get("parsed_criteria")
    
    if cached_hash == current_hash and cached_parsed:
        print("✅ 파일 캐시된 평가 기준 사용 (DB 텍스트 변경 없음)")
        return cached_parsed
    
    # 4. 캐시 미스 또는 텍스트 변경 - 새로 파싱
    print("🔄 평가 기준 새로 파싱 중...")
    
    try:
        # ✅ 수정: 독립 함수 호출
        parsed_criteria = parse_criteria_with_llm(current_raw_text)
        
        # 5. 캐시 업데이트 및 파일 저장
        updated_cache = {
            "raw_text": current_raw_text,
            "raw_text_hash": current_hash,
            "parsed_criteria": parsed_criteria,
            "last_updated": datetime.now().isoformat()
        }
        
        save_cache_to_file(updated_cache)
        
        print("✅ 평가 기준 파싱 완료 및 파일 캐시 업데이트")
        return parsed_criteria
        
    except Exception as e:
        print(f"❌ 평가 기준 파싱 실패: {e}")
        raise e


# ================================================================
# LLM 평가 함수들
# ================================================================

def call_llm_for_passionate_evaluation(
    task_data: List[Dict], basic_info: Dict, evaluation_criteria: Dict[str, str]
) -> Dict:
    """Passionate (열정적 몰입) LLM 평가"""

    emp_name = basic_info.get("emp_name", "")
    task_details = ""

    for task in task_data:
        task_details += f"- 업무 요약: {task.get('task_summary', '')}\n"
        if task.get("task_performance"):
            task_details += f"  성과: {task.get('task_performance')}\n"
        task_details += "\n"

    if not task_details.strip():
        task_details = "분석할 업무 데이터가 없습니다."

    # 평가기준을 동적으로 삽입
    bars_text = evaluation_criteria.get("passionate", "").strip()
    if not bars_text:
        bars_text = "평가 기준 없음. 기본 점수로 평가 진행"

    system_prompt = f"""
    당신은 SK AX 4P 평가 전문가입니다.
    Passionate (열정적 몰입) 기준으로 직원을 평가하세요.

    평가 기준:
    {bars_text}

    Passionate 정의: "이 가치는 규범을 넘어서 헌신과 열정을 가지고 일을 수행하는 것을 강조합니다. 직원들은 에너지와 헌신으로 업무에 임하며, 탁월한 결과를 추구해야 합니다."
    """

    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    </직원 정보>

    <업무 데이터>
    {task_details}
    </업무 데이터>

    위 데이터를 바탕으로 Passionate 관점에서 평가하세요.


    응답은 반드시 다음 JSON 형식으로 작성하세요:
    ```json
    {{
        "score": [1-5점 사이의 숫자],
        "evidence": ["구체적 근거1", "구체적 근거2", "구체적 근거3"],
        "reasoning": "평가 근거 설명",
        "bars_level": "해당 활동이 부합한 평가 기준의 레이블 (예: '탁월한 열정', '성실한 수행' 등)",
        "improvement_points": ["개선점1", "개선점2"]
    }}
    ```
    """

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )

    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        content = str(response.content)  # 타입 안전성 확보
        json_output = _extract_json_from_llm_response(content)
        result = json.loads(json_output)

        # 유효성 검증
        if not isinstance(result.get("score"), (int, float)) or not (
            1 <= result["score"] <= 5
        ):
            result["score"] = 3.0
        if not isinstance(result.get("evidence"), list):
            result["evidence"] = ["평가 근거 생성 실패"]
        if not result.get("reasoning"):
            result["reasoning"] = "기본 평가"
        if not result.get("bars_level"):
            result["bars_level"] = "기본 열정"
        if not isinstance(result.get("improvement_points"), list):
            result["improvement_points"] = ["지속적 개선 필요"]

        return result

    except Exception as e:
        print(f"Passionate 평가 LLM 오류: {e}")
        return {
            "score": 3.0,
            "evidence": ["AI 평가 실패"],
            "reasoning": f"평가 중 오류 발생: {str(e)[:100]}",
            "bars_level": "기본 열정",
            "improvement_points": ["평가 재시도 필요"],
        }


def call_llm_for_proactive_evaluation(
    task_data: List[Dict], basic_info: Dict, evaluation_criteria: Dict[str, str]
) -> Dict:
    """Proactive (능동적 주도) LLM 평가"""

    emp_name = basic_info.get("emp_name", "")
    task_details = ""

    for task in task_data:
        task_details += f"- 업무 요약: {task.get('task_summary', '')}\n"
        if task.get("task_performance"):
            task_details += f"  성과: {task.get('task_performance')}\n"
        task_details += "\n"

    if not task_details.strip():
        task_details = "분석할 업무 데이터가 없습니다."

    # 평가기준을 동적으로 삽입
    bars_text = evaluation_criteria.get("proactive", "").strip()
    if not bars_text:
        bars_text = "평가 기준 없음. 기본 점수로 평가 진행"

    system_prompt = f"""
    당신은 SK AX 4P 평가 전문가입니다.
    Proactive (능동적 주도) 기준으로 직원을 평가하세요.

    평가 기준:
    {bars_text}

    Proactive 정의: "주도적인 태도를 취하고 미래를 대비하는 자세를 장려합니다. 직원들은 도전 과제를 예측하고, 기회를 찾으며, 긍정적인 결과를 이끌어내기 위해 능동적으로 행동해야 합니다."
    """

    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    </직원 정보>

    <업무 데이터>
    {task_details}
    </업무 데이터>

    위 데이터를 바탕으로 Proactive 관점에서 평가하세요.


    응답은 반드시 다음 JSON 형식으로 작성하세요:
    ```json
    {{
        "score": [1-5점 사이의 숫자],
        "evidence": ["구체적 근거1", "구체적 근거2", "구체적 근거3"],
        "reasoning": "평가 근거 설명",
        "bars_level": "해당 활동이 부합한 평가 기준의 레이블 (예: '탁월한 열정', '성실한 수행' 등)",
        "improvement_points": ["개선점1", "개선점2"]
    }}
    ```
    """

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )

    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        content = str(response.content)  # 타입 안전성 확보
        json_output = _extract_json_from_llm_response(content)
        result = json.loads(json_output)

        if not isinstance(result.get("score"), (int, float)) or not (
            1 <= result["score"] <= 5
        ):
            result["score"] = 3.0
        if not isinstance(result.get("evidence"), list):
            result["evidence"] = ["평가 근거 생성 실패"]
        if not result.get("reasoning"):
            result["reasoning"] = "기본 평가"
        if not result.get("bars_level"):
            result["bars_level"] = "기본 주도성"
        if not isinstance(result.get("improvement_points"), list):
            result["improvement_points"] = ["지속적 개선 필요"]

        return result

    except Exception as e:
        print(f"Proactive 평가 LLM 오류: {e}")
        return {
            "score": 3.0,
            "evidence": ["AI 평가 실패"],
            "reasoning": f"평가 중 오류 발생: {str(e)[:100]}",
            "bars_level": "기본 주도성",
            "improvement_points": ["평가 재시도 필요"],
        }


def call_llm_for_professional_evaluation(
    task_data: List[Dict], basic_info: Dict, evaluation_criteria: Dict[str, str]
) -> Dict:
    """Professional (전문성) LLM 평가"""

    emp_name = basic_info.get("emp_name", "")
    position = basic_info.get("position", "")
    task_details = ""

    for task in task_data:
        task_details += f"- 업무 요약: {task.get('task_summary', '')}\n"
        if task.get("task_performance"):
            task_details += f"  성과: {task.get('task_performance')}\n"
        task_details += "\n"

    if not task_details.strip():
        task_details = "분석할 업무 데이터가 없습니다."

    bars_text = evaluation_criteria.get("professional", "").strip()
    if not bars_text:
        bars_text = "평가 기준 없음. 기본 점수로 평가 진행"

    system_prompt = f"""
    당신은 SK AX 4P 평가 전문가입니다.
    Professional (전문성) 기준으로 직원을 평가하세요.

    평가 기준:
    {bars_text}

    Professional 정의: "모든 업무에서 전문성을 유지하는 중요성을 강조합니다. 직원들은 높은 윤리적 기준과 직무 능력을 바탕으로 일을 수행하고 회사의 가치를 대표해야 합니다."
    """

    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    직책: {position}
    </직원 정보>

    <업무 데이터>
    {task_details}
    </업무 데이터>

    위 데이터를 바탕으로 Professional 관점에서 평가하세요.


    응답은 반드시 다음 JSON 형식으로 작성하세요:
    ```json
    {{
        "score": [1-5점 사이의 숫자],
        "evidence": ["구체적 근거1", "구체적 근거2", "구체적 근거3"],
        "reasoning": "평가 근거 설명",
        "bars_level": "해당 활동이 부합한 평가 기준의 레이블 (예: '탁월한 열정', '성실한 수행' 등)",
        "improvement_points": ["개선점1", "개선점2"]
    }}
    ```
    """

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )

    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        content = str(response.content)  # 타입 안전성 확보
        json_output = _extract_json_from_llm_response(content)
        result = json.loads(json_output)

        if not isinstance(result.get("score"), (int, float)) or not (
            1 <= result["score"] <= 5
        ):
            result["score"] = 3.0
        if not isinstance(result.get("evidence"), list):
            result["evidence"] = ["평가 근거 생성 실패"]
        if not result.get("reasoning"):
            result["reasoning"] = "기본 평가"
        if not result.get("bars_level"):
            result["bars_level"] = "기본 전문성"
        if not isinstance(result.get("improvement_points"), list):
            result["improvement_points"] = ["지속적 개선 필요"]

        return result

    except Exception as e:
        print(f"Professional 평가 LLM 오류: {e}")
        return {
            "score": 3.0,
            "evidence": ["AI 평가 실패"],
            "reasoning": f"평가 중 오류 발생: {str(e)[:100]}",
            "bars_level": "기본 전문성",
            "improvement_points": ["평가 재시도 필요"],
        }


def call_llm_for_people_evaluation(
    task_data: List[Dict],
    collaboration_data: Dict,
    peer_talk_data: Dict,
    basic_info: Dict,
    evaluation_criteria: Dict[str, str],
) -> Dict:
    """People (공동체) LLM 평가"""

    emp_name = basic_info.get("emp_name", "")

    # 협업 데이터 정리
    collaboration_info = ""
    if collaboration_data:
        collaboration_info = f"""
        팀 역할: {collaboration_data.get('team_role', '')}
        협업률: {collaboration_data.get('collaboration_rate', 0)}%
        핵심 협업자: {', '.join(collaboration_data.get('key_collaborators', []))}
        동료평가 요약: {collaboration_data.get('peer_talk_summary', '')}
        전체 평가: {collaboration_data.get('overall_evaluation', '')}
        """
    else:
        collaboration_info = "협업 데이터 없음"

    # Peer Talk 데이터 (JSON 구조 반영)
    strengths = peer_talk_data.get("strengths", "")
    concerns = peer_talk_data.get("concerns", "")
    collaboration_observations = peer_talk_data.get("collaboration_observations", "")

    peer_talk_section = f"""
    [동료 피드백 요약]
    - 강점: {strengths if strengths else '정보 없음'}
    - 우려/개선점: {concerns if concerns else '정보 없음'}
    - 협업 관찰: {collaboration_observations if collaboration_observations else '정보 없음'}
    """

    # Task 데이터에서 협업 관련 내용 추출
    collaboration_tasks = ""
    for task in task_data:
        if any(
            keyword in task.get("task_summary", "")
            for keyword in ["협업", "함께", "공동", "팀", "동료"]
        ):
            collaboration_tasks += f"- {task.get('task_summary', '')}\n"

    # 평가기준을 동적으로 삽입
    bars_text = evaluation_criteria.get("people", "").strip()
    if not bars_text:
        bars_text = "평가 기준 없음. 기본 점수로 평가 진행"

    system_prompt = f"""
    당신은 SK AX 4P 평가 전문가입니다.
    People (공동체) 기준으로 직원을 평가하세요.

    평가 기준:
    {bars_text}

    People 정의: "조직 내에서 의미 있는 관계와 팀워크를 형성하는 데 중점을 둡니다. 동료, 이해관계자, 고객과의 협력, 공감, 존중을 장려합니다."
    """

    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    </직원 정보>

    <협업 데이터>
    {collaboration_info}
    </협업 데이터>

    {peer_talk_section}

    <협업 관련 업무>
    {collaboration_tasks if collaboration_tasks else '협업 관련 업무 데이터 없음'}
    </협업 관련 업무>

    위 데이터를 바탕으로 People 관점에서 평가하세요.

    응답은 반드시 다음 JSON 형식으로 작성하세요:
    ```json
    {{
        "score": [1-5점 사이의 숫자],
        "evidence": ["구체적 근거1", "구체적 근거2", "구체적 근거3"],
        "reasoning": "평가 근거 설명",
        "bars_level": "해당 활동이 부합한 평가 기준의 레이블 (예: '탁월한 열정', '성실한 수행' 등)",
        "improvement_points": ["개선점1", "개선점2"]
    }}
    ```
    """

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )

    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        content = str(response.content)  # 타입 안전성 확보
        json_output = _extract_json_from_llm_response(content)
        result = json.loads(json_output)

        if not isinstance(result.get("score"), (int, float)) or not (
            1 <= result["score"] <= 5
        ):
            result["score"] = 3.0
        if not isinstance(result.get("evidence"), list):
            result["evidence"] = ["평가 근거 생성 실패"]
        if not result.get("reasoning"):
            result["reasoning"] = "기본 평가"
        if not result.get("bars_level"):
            result["bars_level"] = "기본적 협력"
        if not isinstance(result.get("improvement_points"), list):
            result["improvement_points"] = ["지속적 개선 필요"]

        return result

    except Exception as e:
        print(f"People 평가 LLM 오류: {e}")
        return {
            "score": 3.0,
            "evidence": ["AI 평가 실패"],
            "reasoning": f"평가 중 오류 발생: {str(e)[:100]}",
            "bars_level": "기본적 협력",
            "improvement_points": ["평가 재시도 필요"],
        }