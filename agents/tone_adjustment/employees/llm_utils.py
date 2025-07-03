"""
LLM 처리 및 JSON 파싱 유틸리티
LLM processing and JSON parsing utilities
"""

import json
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# JSON 유틸리티 함수들 (재구성 방식)
def extract_value(pattern, text, default=""):
    """정규식으로 값 추출"""
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else default

def extract_number(pattern, text, default=0):
    """정규식으로 숫자 추출"""
    match = re.search(pattern, text)
    try:
        return int(match.group(1)) if match else default
    except:
        return default

def extract_float(pattern, text, default=0.0):
    """정규식으로 실수 추출"""
    match = re.search(pattern, text)
    try:
        return float(match.group(1)) if match else default
    except:
        return default

def safe_json_loads(raw_text: str) -> dict:
    """안전한 JSON 로드 - 재구성 방식"""
    if not raw_text:
        return None
    
    try:
        # 1차: 원본 그대로 시도
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    
    try:
        # 2차: 간단한 정리 후 시도
        cleaned = raw_text.replace('\r\n', '\n').replace('\r', '\n')
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\t')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # 3차: 완전 재구성
    print("⚠️ JSON 파싱 실패, 데이터 재구성 중...")
    return reconstruct_json_from_text(raw_text)

def reconstruct_json_from_text(raw_text: str) -> dict:
    """텍스트에서 JSON 데이터 재구성"""
    try:
        # 기본 정보 추출
        name = extract_value(r'"성명":\s*"([^"]+)"', raw_text, "알 수 없음")
        position = extract_value(r'"직위":\s*"([^"]+)"', raw_text, "")
        dept = extract_value(r'"소속":\s*"([^"]+)"', raw_text, "")
        period = extract_value(r'"업무_수행_기간":\s*"([^"]+)"', raw_text, "")
        
        # 수치 데이터 추출
        achievement_rate = extract_number(r'"개인_종합_달성률":\s*(\d+)', raw_text, 0)
        
        # Task 데이터 추출 (여러 개 가능)
        task_patterns = list(re.finditer(r'"Task명":\s*"([^"]+)"', raw_text))
        tasks = []
        
        for i, task_match in enumerate(task_patterns[:3]):  # 최대 3개까지
            task_name = task_match.group(1)
            
            # 해당 Task의 다른 정보들 추출
            start_pos = task_match.start()
            # 다음 Task 시작점 또는 섹션 끝까지
            if i + 1 < len(task_patterns):
                end_pos = task_patterns[i + 1].start()
            else:
                end_pos = raw_text.find('"개인_종합_달성률"', start_pos)
                if end_pos == -1:
                    end_pos = len(raw_text)
            
            task_section = raw_text[start_pos:end_pos]
            
            core_task = extract_value(r'"핵심 Task":\s*"([^"]*(?:\\.[^"]*)*)"', task_section, "업무 진행 중")
            task_rate = extract_number(r'"누적 달성률 \(%\)":\s*(\d+)', task_section, 0)
            comment = extract_value(r'"분석 코멘트":\s*"([^"]*(?:\\.[^"]*)*)"', task_section, f"{name}님의 업무 성과")
            
            # 긴 텍스트는 안전하게 자르기
            if len(core_task) > 500:
                core_task = core_task[:500] + "..."
            if len(comment) > 1000:
                comment = comment[:1000] + "..."
            
            tasks.append({
                "Task명": task_name,
                "핵심 Task": core_task,
                "누적 달성률 (%)": task_rate,
                "분석 코멘트": comment
            })
        
        # Peer Talk 추출
        strength = extract_value(r'"강점":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "팀워크 우수")
        concern = extract_value(r'"우려":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "지속적인 성장 필요")
        collaboration = extract_value(r'"협업_관찰":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "동료들과 원활한 협업")
        
        # 4P 평가 추출
        passionate = extract_value(r'"Passionate":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "업무에 대한 열정")
        proactive = extract_value(r'"Proactive":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "능동적인 업무 수행")
        professional = extract_value(r'"Professional":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "전문성 기반 업무")
        people = extract_value(r'"People":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "우수한 대인관계")
        overall_eval = extract_value(r'"종합_평가":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "우수한 역량")
        
        # 성장 제안 추출
        growth_points = []
        improvement_areas = []
        recommended_activities = []
        
        # 배열 데이터는 간단하게 처리
        growth_section = extract_value(r'"성장_포인트":\s*\[(.*?)\]', raw_text, "")
        if growth_section:
            growth_matches = re.findall(r'"([^"]*(?:\\.[^"]*)*)"', growth_section)
            growth_points = growth_matches[:3]  # 최대 3개
        
        improvement_section = extract_value(r'"보완_영역":\s*\[(.*?)\]', raw_text, "")
        if improvement_section:
            improvement_matches = re.findall(r'"([^"]*(?:\\.[^"]*)*)"', improvement_section)
            improvement_areas = improvement_matches[:3]
        
        activity_section = extract_value(r'"추천_활동":\s*\[(.*?)\]', raw_text, "")
        if activity_section:
            activity_matches = re.findall(r'"([^"]*(?:\\.[^"]*)*)"', activity_section)
            recommended_activities = activity_matches[:3]
        
        # 총평 추출 (안전하게)
        total_comment = extract_value(r'"총평":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "")
        if len(total_comment) > 1000:
            total_comment = total_comment[:1000] + "..."
        if not total_comment:
            total_comment = f"{name}님의 이번 분기 성과는 우수합니다."
        
        # 안전한 JSON 구조 생성
        reconstructed_json = {
            "기본_정보": {
                "성명": name,
                "직위": position,
                "소속": dept,
                "업무_수행_기간": period
            },
            "팀_업무_목표_및_개인_달성률": {
                "업무표": tasks,
                "개인_종합_달성률": achievement_rate,
                "종합_기여_코멘트": f"{name}님은 팀의 목표 달성에 기여하고 있습니다.",
                "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
            },
            "Peer_Talk": {
                "강점": strength,
                "우려": concern,
                "협업_관찰": collaboration
            },
            "업무_실행_및_태도": {
                "Passionate": passionate,
                "Proactive": proactive,
                "Professional": professional,
                "People": people,
                "종합_평가": overall_eval
            },
            "성장_제안_및_개선_피드백": {
                "성장_포인트": growth_points if growth_points else [f"{name}님의 전문성 확대 가능"],
                "보완_영역": improvement_areas if improvement_areas else ["지속적인 학습 필요"],
                "추천_활동": recommended_activities if recommended_activities else ["팀 내 지식 공유"]
            },
            "총평": total_comment
        }
        
        print(f"✅ JSON 재구성 완료: {name}")
        return reconstructed_json
        
    except Exception as e:
        print(f"❌ JSON 재구성 실패: {e}")
        # 최소한의 안전한 데이터라도 반환
        return {
            "기본_정보": {
                "성명": "데이터 오류",
                "직위": "",
                "소속": "",
                "업무_수행_기간": ""
            },
            "팀_업무_목표_및_개인_달성률": {
                "업무표": [],
                "개인_종합_달성률": 0,
                "종합_기여_코멘트": "데이터 처리 중 오류 발생",
                "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
            },
            "Peer_Talk": {
                "강점": "데이터 오류",
                "우려": "데이터 오류",
                "협업_관찰": "데이터 오류"
            },
            "업무_실행_및_태도": {
                "Passionate": "데이터 오류",
                "Proactive": "데이터 오류", 
                "Professional": "데이터 오류",
                "People": "데이터 오류",
                "종합_평가": "데이터 오류"
            },
            "성장_제안_및_개선_피드백": {
                "성장_포인트": ["데이터 오류"],
                "보완_영역": ["데이터 오류"],
                "추천_활동": ["데이터 오류"]
            },
            "총평": "데이터 처리 중 오류가 발생했습니다."
        }

class LLMClient:
    """LLM 클라이언트 (LangChain ChatOpenAI 사용)"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.client = llm_client
    
    def get_employee_tone_correction_prompt(self, report_type) -> str:
        """팀원용 레포트 톤 보정 프롬프트 반환"""
        base_prompt = """
# 팀원용 레포트 톤 보정 프롬프트

## 대상 레포트
* **팀원용 분기별 레포트**: 팀원 개인의 분기 성과 평가
* **팀원용 연말 레포트**: 팀원 개인의 연말 성과 평가

## 레포트 식별 기준
* 개인 기본정보 포함 (성명, 직위, 소속)
* 개인 중심의 성과 평가 내용
* Peer_Talk, 개인 달성률, 성장 제안 등 개인 성장 관련 섹션

## 톤 보정 규칙

### 1. 자연스러운 주어 처리 (개인 레포트 특화)
* **통일 기준**: "[이름]님은"으로 일관성 있게 사용
* **수정 대상**:
   * "해당 직원은" → 생략하거나 "[이름]님은"로 변경
   * "[이름]은" → "[이름]님은"
   * 불필요한 3인칭 반복 주어 제거하여 자연스럽게 작성
* **예시**: "김개발은" → "김개발님은"

### 2. 문체 일관성
* **통일 기준**: 격식체(존댓말) 사용
* **수정 대상**:
   * 평어체(~다, ~였다) → 격식체(~습니다, ~였습니다)
   * 모든 문장을 존댓말로 통일
* **예시**: "보였다" → "보였습니다"

### 3. 반복 표현 개선
* **문제**: 동일한 협업 표현의 과도한 반복
* **해결책**: 다양한 표현으로 변경
   * "[이름]님과 [이름]님과의 협업을 통해" →
      * "[이름]님, [이름]님과 협력하여"
      * "동료들과 협업하여"
      * "팀원들과의 협업 과정에서"
      * "[이름]님과 함께"

### 4. 부정적 피드백 전달 방식 개선 (팀원용 특화)
* **원칙**: 개인 성장 중심의 건설적이고 격려적인 표현 사용
* **수정 예시**:
   * "실수가 잦고 기술 이해력이 부족" → "업무 정확성과 기술적 이해도 향상을 통해 더욱 발전하실 수 있는 영역이 있습니다"
   * "무관심한 태도" → "업무 몰입도를 높여 더욱 성장하실 수 있는 기회가 있습니다"
   * "산만한 모습" → "집중력 향상을 통해 더 나은 성과를 이루실 수 있습니다"
   * "개선이 필요합니다" → "더욱 성장하실 수 있는 영역입니다"

### 5. 시간 표현 구체화
* **원칙**: 기본_정보의 업무_수행_기간과 일치시키기
* **수정 예시**:
   * "이번 연도" → "2024년"
   * "올해" → "2024년"
   * "이번 연간 평가에서" → "2024년 성과 평가에서"
   * "이번 분기" → "2024년 X분기"
   * "이번 분기 동안" → JSON의 업무_수행_기간 값 사용

## 팀원용 연말 레포트 특화 글자수 제한 규칙

### 6. 연말 레포트 글자수 제한 (중요!)
**팀원용 연말 레포트에만 적용되는 특별 규칙**

#### A. 분기별_업무_기여도 배열의 실적_요약 글자수 제한
* **JSON 경로**: `json.분기별_업무_기여도[배열인덱스].실적_요약`
* **현재 문제**: 실제 데이터에서 437자 등 300자를 크게 초과
* **제한**: 각 분기별 실적_요약이 300자를 초과하는 경우 **약 300자로 축약**
* **축약 원칙**:
   - 핵심 성과 수치와 달성률은 반드시 유지 (예: "평균 달성률 53.3%", "기여도 79.3점")
   - 중요한 Task별 기여도는 보존 (예: "AI 제안서 기능 고도화 개발에서 20% 달성률, 기여도 90점")
   - 불필요한 수식어와 반복 표현 제거 ("김개발(SK0002)님은" → "김개발님은")
   - 협업 내용은 간략하게 요약
   - 팀 평균 달성률과의 비교는 유지 (예: "팀 평균 달성률 85.3%에 비해")

#### B. 팀_업무_목표_및_개인_달성률의 업무표 배열 내 분석_코멘트 글자수 제한
* **JSON 경로**: `json.팀_업무_목표_및_개인_달성률.업무표[배열인덱스].분석_코멘트`
* **현재 문제**: 실제 데이터에서 375자 등 240자를 크게 초과
* **제한**: 각 Task의 분석_코멘트가 240자를 초과하는 경우 **약 230자로 축약**
* **축약 원칙**:
   - Task 달성률과 기여도 수치는 필수 유지 (예: "100% 완성", "35점의 기여도")
   - 핵심 성과와 협업 내용 요약 (예: "제안서 작성 시간 42% 단축", "이설계님, 박DB님과 협업")
   - 팀 기여도와 성장 관련 내용 보존
   - 장황한 설명과 반복 표현 제거 ("2023년 동안 팀은" → "팀은")
   - 격려적 톤은 유지하면서 간결하게 표현

## 팀원용 레포트 특화 톤

### 핵심 톤 특성
* **개인 성장 지향적**: 개인의 발전과 성장에 초점
* **격려 중심**: 동기부여와 격려가 느껴지는 표현
* **개인적이고 따뜻함**: 개인에게 직접 말하는 듯한 친근한 톤
* **성장 가능성 강조**: 개선점도 성장 기회로 표현
* **자연스러운 문체**: 개인 레포트답게 불필요한 주어 반복 최소화

### 특화 표현 가이드라인

**1. 성과 인정 표현**
* "[이름]님의 뛰어난 성과를 축하드립니다"
* "[이름]님께서 이루신 성과는 정말 인상적입니다"
* "이번 성과를 통해 [이름]님의 전문성이 더욱 빛났습니다"

**2. 성장 제안 표현**
* "[이름]님의 지속적인 성장을 위해"
* "더욱 발전하실 수 있는 기회가 있습니다"
* "[이름]님이 더 큰 성과를 이루실 수 있도록"

**3. 격려 표현**
* "앞으로도 [이름]님의 성장을 기대합니다"
* "[이름]님의 노력과 열정에 감사드립니다"
* "계속해서 발전하는 모습을 기대하겠습니다"

**4. 개선점 표현**
* "개선이 필요합니다" → "더욱 성장하실 수 있는 영역입니다"
* "부족한 부분" → "발전 가능성이 있는 부분"
* "문제점" → "성장 포인트"

## 유지해야 할 요소

### 1. 수식어는 그대로 유지
* "진심으로", "매우", "크게", "성공적으로" 등의 수식어는 제거하지 않음
* 감정적 표현과 강조 표현은 보존
* 단, 글자수 제한 시에는 중복되는 수식어만 정리

### 2. 구체적 수치와 데이터
* 달성률, 기여도, 점수 등의 수치 정보는 정확히 유지
* 분석 내용의 실질적 정보는 변경하지 않음

### 3. 평가 기준과 구조
* JSON의 원본 구조와 항목 구성 유지
* 평가 기준과 해석 방식은 그대로 보존

## 팀원용 레포트 특별 고려사항
* 개인의 자존감과 동기부여에 긍정적 영향을 주는 톤
* 성장과 발전에 대한 희망적 메시지 포함
* 개인의 강점을 적극적으로 인정하고 격려
* 개선점은 성장 기회로 재프레임하여 표현
* **연말 레포트의 경우 글자수 제한을 준수하여 간결하면서도 의미 있는 내용 유지**

## 지시사항
아래 JSON 형태의 업무 평가 레포트의 톤을 위 규칙에 따라 보정해주세요.
원본 JSON 구조와 데이터는 유지하되, 텍스트 내용만 보정하여 완전한 JSON 형태로 응답해주세요.
**특히 팀원용 연말 레포트의 경우 글자수 제한 규칙을 반드시 준수해주세요.**
"""
        
        if str(report_type) == "ReportType.EMPLOYEE_QUARTERLY":
            base_prompt += "\n이 레포트는 팀원용 분기별 레포트입니다."
        else:
            base_prompt += "\n이 레포트는 팀원용 연말 레포트입니다."
        
        return base_prompt
    
    def correct_tone(self, report_json: Dict[str, Any], report_type) -> str:
        """LLM을 사용한 톤 보정"""
        system_prompt = self.get_employee_tone_correction_prompt(report_type)
        report_text = json.dumps(report_json, ensure_ascii=False, indent=2)
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=report_text)
            ]
            
            response = self.client.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"LLM API 오류: {e}")
            return report_text