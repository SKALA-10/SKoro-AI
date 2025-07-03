"""
중간 레포트 LLM 처리 및 JSON 파싱 유틸리티
Middle report LLM processing and JSON parsing utilities
"""

import json
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# JSON 유틸리티 함수들 (재사용)
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

def safe_json_loads_middle(raw_text: str) -> dict:
    """중간 레포트 안전한 JSON 로드"""
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
    
    # 3차: 중간 레포트 재구성
    print("⚠️ 중간 레포트 JSON 파싱 실패, 데이터 재구성 중...")
    return reconstruct_middle_json_from_text(raw_text)

def reconstruct_middle_json_from_text(raw_text: str) -> dict:
    """중간 레포트 텍스트에서 JSON 데이터 재구성"""
    try:
        # 기본 정보 추출
        team_name = extract_value(r'"팀명":\s*"([^"]+)"', raw_text, "알 수 없는 팀")
        evaluation_period = extract_value(r'"평가_기간":\s*"([^"]+)"', raw_text, "")
        report_date = extract_value(r'"중간평가_작성일":\s*"([^"]+)"', raw_text, "")
        
        # 팀원별 중간 평가 추출
        member_evaluations = []
        member_patterns = list(re.finditer(r'"성명":\s*"([^"]+)"', raw_text))
        
        for i, member_match in enumerate(member_patterns[:10]):  # 최대 10명까지
            member_name = member_match.group(1)
            
            # 해당 팀원의 섹션 범위 결정
            start_pos = member_match.start()
            if i + 1 < len(member_patterns):
                end_pos = member_patterns[i + 1].start()
            else:
                end_pos = raw_text.find('"종합_관찰_요약"', start_pos)
                if end_pos == -1:
                    end_pos = len(raw_text)
            
            member_section = raw_text[start_pos:end_pos]
            
            # 팀원 정보 추출
            position = extract_value(r'"직위":\s*"([^"]*)"', member_section, "")
            ai_score = extract_float(r'"AI_추천_점수":\s*([0-9.]+)', member_section, 0.0)
            achievement_trend = extract_value(r'"달성률_추이":\s*"([^"]*(?:\\.[^"]*)*)"', member_section, "관찰 중")
            performance_observation = extract_value(r'"성과_관찰":\s*"([^"]*(?:\\.[^"]*)*)"', member_section, f"{member_name}님에 대한 관찰")
            collaboration_pattern = extract_value(r'"협업_패턴":\s*"([^"]*(?:\\.[^"]*)*)"', member_section, "협업 관찰 중")
            improvement_trend = extract_value(r'"개선_동향":\s*"([^"]*(?:\\.[^"]*)*)"', member_section, "추이 관찰 중")
            evaluation_notes = extract_value(r'"연말평가_참고사항":\s*"([^"]*(?:\\.[^"]*)*)"', member_section, "추가 관찰 필요")
            
            # 긴 텍스트 자르기
            if len(performance_observation) > 600:
                performance_observation = performance_observation[:600] + "..."
            if len(evaluation_notes) > 500:
                evaluation_notes = evaluation_notes[:500] + "..."
            
            member_evaluations.append({
                "성명": member_name,
                "직위": position,
                "AI_추천_점수": ai_score,
                "달성률_추이": achievement_trend,
                "성과_관찰": performance_observation,
                "협업_패턴": collaboration_pattern,
                "개선_동향": improvement_trend,
                "연말평가_참고사항": evaluation_notes
            })
        
        # 팀 전체 분석 추출
        team_performance_trend = extract_value(r'"팀_성과_추이":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "팀 성과 관찰 중")
        risk_observations = extract_value(r'"리스크_관찰사항":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "특별한 리스크 관찰되지 않음")
        growth_potential = extract_value(r'"성장_잠재력_분석":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "성장 요소 관찰 중")
        
        # AI 점수 산출 기준 추출
        ai_criteria = extract_value(r'"AI_점수_산출기준":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "AI 분석 기준에 따른 점수")
        data_analysis_method = extract_value(r'"데이터_분석_방법":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "정량적 지표 기반 분석")
        
        # 종합 관찰 요약 추출
        comprehensive_summary = extract_value(r'"종합_관찰_요약":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "")
        if len(comprehensive_summary) > 1000:
            comprehensive_summary = comprehensive_summary[:1000] + "..."
        if not comprehensive_summary:
            comprehensive_summary = f"{team_name}의 중간 평가 시점 관찰 결과입니다."
        
        # 중간 레포트 JSON 구조 생성
        reconstructed_json = {
            "중간평가_기본정보": {
                "팀명": team_name,
                "평가_기간": evaluation_period,
                "중간평가_작성일": report_date
            },
            "팀원별_중간평가": {
                "팀원_평가표": member_evaluations,
                "평가_참고사항": f"{team_name} 팀원들의 중간 평가 시점 관찰 결과입니다."
            },
            "팀_전체_분석": {
                "팀_성과_추이": team_performance_trend,
                "리스크_관찰사항": risk_observations,
                "성장_잠재력_분석": growth_potential
            },
            "AI_점수_산출_기준": {
                "AI_점수_산출기준": ai_criteria,
                "데이터_분석_방법": data_analysis_method
            },
            "종합_관찰_요약": comprehensive_summary
        }
        
        print(f"✅ 중간 레포트 JSON 재구성 완료: {team_name}")
        return reconstructed_json
        
    except Exception as e:
        print(f"❌ 중간 레포트 JSON 재구성 실패: {e}")
        # 최소한의 안전한 데이터 반환
        return {
            "중간평가_기본정보": {
                "팀명": "데이터 오류",
                "평가_기간": "",
                "중간평가_작성일": ""
            },
            "팀원별_중간평가": {
                "팀원_평가표": [],
                "평가_참고사항": "데이터 처리 중 오류 발생"
            },
            "팀_전체_분석": {
                "팀_성과_추이": "데이터 오류",
                "리스크_관찰사항": "데이터 오류",
                "성장_잠재력_분석": "데이터 오류"
            },
            "AI_점수_산출_기준": {
                "AI_점수_산출기준": "데이터 오류",
                "데이터_분석_방법": "데이터 오류"
            },
            "종합_관찰_요약": "데이터 처리 중 오류가 발생했습니다."
        }

class MiddleLLMClient:
    """중간 레포트 LLM 클라이언트"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.client = llm_client
    
    def get_middle_tone_correction_prompt(self) -> str:
        """중간 레포트 톤 보정 프롬프트 반환"""
        prompt = """
# 팀장용 중간 레포트 톤 보정 프롬프트

## 대상 레포트
* **팀장용 중간 레포트**: 연말 성과평가 지원을 위한 중간 평가 레포트
* **최종전 중간평가 레포트**: 연말 평가 전 팀장 참고용 분석 자료

## 톤 보정 규칙

### 1. 호칭 통일
* **통일 기준**: "[이름]님은"으로 일관성 있게 사용
* **수정 대상**:
   * "해당 직원은" → "[이름]님은"
   * "[이름]은" → "[이름]님은"
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
      * "협업 과정에서"
      * "공동 작업을 통해"
      * "팀원 간 협력으로"

### 4. 부정적 피드백 전달 방식 개선 (중간평가 특화)
* **원칙**: 관찰 사실 중심의 분석적이고 참고용 표현 사용
* **수정 예시**:
   * "실수가 잦고 기술 이해력이 부족" → "업무 정확성 및 기술적 이해도 측면에서 연말 평가 시 고려가 필요한 부분이 관찰됩니다"
   * "무관심한 태도" → "업무 몰입도 관련하여 추가 관찰이 필요한 상황입니다"
   * "산만한 모습" → "집중력 부분에서 연말 평가 시 참고할 만한 사항이 있습니다"

### 5. 시간 표현 구체화
* **원칙**: 기본_정보의 업무_수행_기간과 일치시키기
* **수정 예시**:
   * "이번 기간 동안" → JSON의 업무_수행_기간 값 사용
   * "현재까지의 관찰 결과" → "2024년 연말 시점까지의 관찰 결과"

## 팀장용 중간 레포트 특화 톤

### 핵심 톤 특성
* **분석적·관찰 중심**: 객관적 사실과 데이터 기반 분석
* **참고용·예비적**: 확정적이지 않은 관찰과 추세 분석
* **연말 평가 지원**: 최종 평가를 위한 근거 자료 제공
* **추세 및 패턴 중심**: 성과 변화와 발전 패턴 분석

### 특화 표현 가이드라인

**1. 관찰 및 분석 표현**
* "중간 평가 시점에서 관찰된 사항"
* "현재까지의 성과 추이를 분석하면"
* "연말 평가를 위한 참고 사항으로"
* "관찰 결과에 따르면"

**2. 추세 분석 표현**
* "성과 변화 패턴을 보면"
* "분기별 추이를 분석할 때"
* "발전 양상을 살펴보면"
* "성장 곡선 관점에서"

**3. 예비적 판단 표현**
* "연말 평가 시 고려할 사항"
* "최종 평가에서 참고할 만한 부분"
* "추가 관찰이 필요한 영역"
* "잠정적 평가 근거"

**4. 데이터 기반 분석 표현**
* "AI 추천 점수 분석 결과"
* "정량적 지표 기준으로"
* "누적 데이터를 통해"
* "객관적 평가 지표에 따르면"

**5. 미래 지향적 표현**
* "연말까지의 전망"
* "향후 개선 가능성"
* "잠재적 성장 요소"
* "연말 평가에서 기대되는 부분"

## 중간 평가 특별 표현 방식

### 확정적 표현 지양
* "~입니다" → "~로 관찰됩니다"
* "~해야 합니다" → "~를 고려해볼 필요가 있습니다"
* "~문제입니다" → "~부분에서 추가 관찰이 필요합니다"

### 관찰 중심 표현 강화
* "~한 모습을 보였습니다" → "~한 양상이 관찰되었습니다"
* "~성과를 달성했습니다" → "~수준의 성과가 측정되었습니다"
* "~능력이 뛰어납니다" → "~영역에서 높은 역량이 확인되었습니다"

## 글자수 제한 규칙 (필수 준수)

### JSON 구조 분석 및 글자수 제한 대상
다음은 실제 JSON 구조를 분석하여 확인한 글자수 제한 대상입니다:

```
JSON 최상위 구조:
├── 기본_정보
├── 팀원_평가_요약표  
└── 팀원별_평가_근거 (배열) ← 여기서 글자수 제한 적용
    ├── [0] 첫 번째 팀원 (예: 김개발)
    ├── [1] 두 번째 팀원 (예: 이설계)  
    └── [2] 세 번째 팀원 (예: 박DB)
```

### 필수 글자수 제한 대상 (7개 위치)

** 위치 1: 업적 실적 요약**
- **정확한 JSON 경로**: `팀원별_평가_근거[i].AI_점수_산출_기준.업적.실적 요약`
- **최대 글자수**: **180자**
- **현재 내용 예시**: "김개발(SK0002)님은 올해 총 3개의 주요 Task에 참여하여 평균 달성률 95.0%를 기록하며 팀의 성과에 기여하였습니다..."
- **단축 시 유지해야 할 핵심**: 달성률 수치, 기여도 점수, Task 개수

** 위치 2: Passionate 평가 근거**  
- **정확한 JSON 경로**: `팀원별_평가_근거[i].AI_점수_산출_기준.SK_Values.Passionate.평가 근거`
- **최대 글자수**: **180자**
- **현재 내용 예시**: "김개발은 다양한 프로젝트에서 주도적으로 역할을 수행하며, 목표를 초과 달성하는 성과를 지속적으로 보여주었습니다..."
- **단축 시 유지해야 할 핵심**: 열정적 태도 사례, 주도적 성과

** 위치 3: Proactive 평가 근거**
- **정확한 JSON 경로**: `팀원별_평가_근거[i].AI_점수_산출_기준.SK_Values.Proactive.평가 근거`
- **최대 글자수**: **180자**  
- **현재 내용 예시**: "김개발은 팀의 생산성 향상과 신규 고객 확보를 위해 능동적으로 문제를 분석하고 해결책을 제시했습니다..."
- **단축 시 유지해야 할 핵심**: 능동적 문제해결 사례, 혁신적 아이디어

** 위치 4: Professional 평가 근거**
- **정확한 JSON 경로**: `팀원별_평가_근거[i].AI_점수_산출_기준.SK_Values.Professional.평가 근거` 
- **최대 글자수**: **180자**
- **현재 내용 예시**: "김개발은 AI 제안서 자동 생성 시스템의 핵심 개발자로서 요구사항 분석과 알고리즘 설계에 기여하며..."
- **단축 시 유지해야 할 핵심**: 전문기술 발휘 사례, 구체적 기여 내용

** 위치 5: People 평가 근거**
- **정확한 JSON 경로**: `팀원별_평가_근거[i].AI_점수_산출_기준.SK_Values.People.평가 근거`
- **최대 글자수**: **180자**
- **현재 내용 예시**: "김개발은 팀워크를 극대화하고, 동료들과의 협업을 통해 팀의 목표 달성에 크게 기여했습니다..."
- **단축 시 유지해야 할 핵심**: 협업 성과, 팀워크 기여도

** 위치 6: 평가 근거 요약**
- **정확한 JSON 경로**: `팀원별_평가_근거[i].AI_점수_산출_기준.평가 근거 요약`
- **최대 글자수**: **200자**
- **현재 내용 예시**: "김개발(SK0002)님은 올해 총 3개의 주요 Task에 참여하여 평균 달성률 95.0%를 기록하였습니다. 달성률 점수는 3.2점으로 B등급에 해당하며..."
- **단축 시 유지해야 할 핵심**: 최종 점수, 등급, 주요 성과 수치

** 위치 7: Task별 분석 코멘트**
- **정확한 JSON 경로**: `팀원별_평가_근거[i].연간_핵심_성과_기여도.Task_표[j].분석 코멘트`
- **최대 글자수**: **200자**
- **현재 내용 예시**: "2023년 동안 팀은 평균 104.8%의 성과 달성률을 기록하며, 각 구성원이 중요한 기여를 하였습니다. 특히 김개발(SK0002)님은..."
- **단축 시 유지해야 할 핵심**: 달성률, 기여도 점수, 협업 내용

### 글자수 단축 시 우선순위 (모든 위치 공통)
1. **수치 데이터 최우선 보존**: 점수, 달성률, 기여도는 절대 삭제 금지
2. **구체적 성과 사례 유지**: 추상적 표현보다 구체적 성과 우선
3. **불필요한 수식어 제거**: "매우", "크게", "상당히" 등 제거
4. **중복 표현 통합**: 같은 의미 반복 문장 통합
5. **문장 구조 간소화**: 복합문을 단순문으로 변경

## 유지해야 할 요소

### 1. 수식어는 그대로 유지
* "진심으로", "매우", "크게", "성공적으로" 등의 수식어는 제거하지 않음
* 단, 확정적 어조는 관찰적 어조로 조정

### 2. 구체적 수치와 데이터
* 달성률, 기여도, 점수 등의 수치 정보는 정확히 유지
* AI 추천 점수 등의 정량적 데이터는 그대로 보존

### 3. 평가 기준과 구조
* JSON의 원본 구조와 항목 구성 유지
* 평가 근거와 산출 방식은 그대로 보존

## 중간 레포트 특별 고려사항
* 연말 평가의 객관성과 공정성을 뒷받침하는 근거 자료로서의 역할
* 확정적 판단보다는 관찰과 분석에 기반한 참고 정보 제공
* 팀장이 연말 평가 시 고려해야 할 다양한 관점과 맥락 정보 포함
* 데이터의 신뢰성과 분석의 객관성을 최우선으로 하는 톤 유지

## 지시사항
아래 JSON 형태의 중간평가 레포트의 톤을 위 규칙에 따라 보정해주세요.
**특히 글자수 제한 규칙을 반드시 준수하여, 지정된 글자수를 초과하는 모든 텍스트를 단축해주세요.**
원본 JSON 구조와 데이터는 유지하되, 텍스트 내용만 보정하여 완전한 JSON 형태로 응답해주세요.
"""
        return prompt
    
    def correct_tone(self, report_json: Dict[str, Any]) -> str:
        """LLM을 사용한 중간 레포트 톤 보정"""
        system_prompt = self.get_middle_tone_correction_prompt()
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
            return report_text  # 오류 시 원본 반환