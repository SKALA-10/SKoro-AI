"""
팀장용 LLM 처리 및 JSON 파싱 유틸리티 (글자수 제한 강화 버전)
Team manager LLM processing and JSON parsing utilities (enhanced character limit version)
"""

import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class ManagerLLMClient:
    """팀장용 LLM 클라이언트"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.client = llm_client
    
    def get_manager_tone_correction_prompt(self, report_type) -> str:
        """팀장용 레포트 톤 보정 프롬프트 반환 (글자수 제한 강화)"""
        base_prompt = """
# 팀장용 레포트 톤 보정 프롬프트 (글자수 제한 강화 버전)

## ⚠️ 최우선 규칙: 200자 글자수 제한 (절대 준수)

### 📏 글자수 제한 규칙 (반드시 준수)
* **대상**: `팀원_성과_분석` → `팀원별_기여도` → 각 팀원의 `기여_내용` 필드
* **제한**: 각 팀원의 `기여_내용` 필드가 **정확히 200자 이내**
* **우선순위**: 글자수 제한 > 모든 다른 규칙
* **검증**: 각 팀원별로 글자수를 반드시 확인하고 200자 초과 시 강제 단축

### 🎯 글자수 단축 전략 (우선순위 순)
1. **핵심 정보만 보존**:
   - Task 수, 달성률, 기여도 점수 (필수)
   - 팀 내 순위/비교 (필수)
   - 주요 Task명 1-2개만 선택

2. **제거 대상**:
   - "이러한 분석을 통해", "향후 성과 향상을 위해서는" 등 연결어구
   - "현재", "특히", "그러나", "따라서" 등 과도한 접속사
   - 중복되는 수식어나 부연 설명
   - 과도한 상세 분석 내용
   - "상대적으로", "전반적으로", "현재 상황에서" 등 불필요한 표현

3. **문장 압축**:
   - 긴 문장을 2-3개 짧은 문장으로 분할
   - 핵심만 남기고 부연설명 제거
   - 수치 중심의 간결한 서술

### 📝 글자수 단축 실행 예시
**Before (300자+)**:
"김개발(SK0002)님은 현재 3개의 Task에 참여하여 평균 달성률 60.0%와 평균 기여도 40.3점을 기록하였습니다. 특히, AI 제안서 기능 고도화 개발에서 50%의 달성률을 보이며 27점의 기여도를 나타냈으나, 이는 팀 평균인 80.9%에 비해 낮은 수치입니다..."

**After (200자 이내, 격식체 유지)**:
"김개발님은 3개 Task에 참여하여 평균 달성률 60.0%, 기여도 40.3점을 기록하였습니다. AI 제안서 개발(50%, 27점), 팀 가동률 모니터링(75%, 91점), 신규 고객 발굴(55%, 3점)로 팀 평균 80.9% 대비 낮은 성과를 보였습니다. 특정 분야 강점이 있으나 전반적 성과 개선이 필요합니다."

## 기본 톤 보정 규칙

### 1. 호칭 통일
* **통일 기준**: "[이름]님은"으로 일관성 있게 사용
* **수정 대상**: "해당 직원은" → "[이름]님은", "[이름]은" → "[이름]님은"

### 2. 문체 일관성
* **통일 기준**: 격식체(존댓말) 사용
* **수정 대상**: 평어체(~다, ~였다) → 격식체(~습니다, ~였습니다)

### 3. 반복 표현 개선
* 동일한 협업 표현의 과도한 반복 방지
* 다양한 표현으로 변경: "협업을 통해" → "협력하여", "팀 차원의 협력으로"

### 4. 부정적 피드백 전달 방식 개선
* 관리적 관점에서 객관적이고 전략적인 표현 사용
* "실수가 잦음" → "업무 정확성 향상 필요"

### 5. 시간 표현 구체화
* 팀_기본정보의 업무_수행_기간과 일치시키기

## 글자수 단축을 위한 표현 변경 가이드

### 압축 가능한 표현들 (격식체 유지)
* "현재 3개의 Task에 참여하여" → "3개 Task에 참여하여"
* "평균 달성률 60.0%와 평균 기여도 40.3점을 기록하였습니다" → "평균 달성률 60.0%, 기여도 40.3점을 기록하였습니다"
* "이는 팀 평균인 80.9%에 비해 낮은 수치입니다" → "팀 평균 80.9% 대비 낮은 성과입니다"
* "향후 성과 개선을 위해서는" → "성과 개선을 위해서는"
* "상대적으로 긍정적인 성과를 보였으나" → "긍정적 성과를 보였으나"

### ⚠️ 문체 일관성 주의사항
* **절대 금지**: "기록했다", "보였다", "필요하다" 등 평어체 사용 금지
* **반드시 사용**: "기록하였습니다", "보였습니다", "필요합니다" 등 격식체 유지
* **압축시에도**: 격식체를 반드시 유지하면서 단축

### 핵심 정보 우선순위
1. **1순위**: Task 수, 달성률, 기여도 점수
2. **2순위**: 팀 평균 대비 위치
3. **3순위**: 주요 Task명 1-2개
4. **4순위**: 간단한 평가 및 개선 방향

## ⚠️ 중요 지시사항

1. **절대 규칙**: 각 팀원의 `기여_내용`을 200자 이내로 반드시 맞춤
2. **문체 일관성**: 모든 문장을 격식체(~습니다, ~하였습니다)로 통일
3. **우선순위**: 글자수 준수 + 격식체 유지 > 다른 모든 규칙
4. **검증 필수**: 응답 전 각 팀원별 글자수 확인
5. **핵심 보존**: 수치 데이터와 핵심 평가는 유지
6. **JSON 구조**: 완전한 JSON 형태로 응답

### 문체 일관성 체크리스트
- [ ] 모든 문장이 "~습니다", "~하였습니다" 등 격식체로 끝나는가?
- [ ] "기록했다", "보였다", "필요하다" 등 평어체가 없는가?
- [ ] 압축 과정에서도 격식체가 유지되었는가?

## 글자수 확인 방법
응답하기 전에 반드시:
1. 각 팀원의 `기여_내용` 글자수를 확인
2. 200자 초과 시 추가 단축
3. 핵심 정보(수치, 순위, 주요 Task)는 보존하면서 단축

아래 JSON 형태의 팀장용 업무 평가 레포트의 톤을 위 규칙에 따라 보정해주세요.
**특히 중요**: 팀원_성과_분석 → 팀원별_기여도의 각 팀원별 기여_내용 필드가 반드시 200자 이내가 되도록 조정해주세요.
원본 JSON 구조와 데이터는 완전히 유지하되, 텍스트 내용만 보정하여 완전한 JSON 형태로 응답해주세요.
"""
        
        if str(report_type) == "ManagerReportType.MANAGER_QUARTERLY":
            base_prompt += "\n이 레포트는 팀장용 분기별 레포트입니다."
        else:
            base_prompt += "\n이 레포트는 팀장용 연말 레포트입니다."
        
        return base_prompt
    
    def correct_tone(self, report_json: Dict[str, Any], report_type) -> str:
        """LLM을 사용한 팀장용 톤 보정 (글자수 제한 강화)"""
        system_prompt = self.get_manager_tone_correction_prompt(report_type)
        
        # JSON을 문자열로 변환하여 전달
        report_text = json.dumps(report_json, ensure_ascii=False, indent=2)
        
        # 추가적인 글자수 제한 강조 메시지
        character_limit_reminder = """
⚠️ 글자수 제한 재확인:
- 각 팀원의 기여_내용이 정확히 200자 이내여야 합니다
- 200자를 1글자라도 초과하면 안됩니다
- 응답하기 전에 반드시 각 팀원별로 글자수를 확인하세요
- 핵심 수치(달성률, 기여도, Task 수)는 유지하면서 최대한 압축하세요
"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=character_limit_reminder + "\n\n" + report_text)
            ]
            
            response = self.client.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"LLM API 오류: {e}")
            return report_text  # 오류 시 원본 반환
    
    def validate_character_limits(self, corrected_json: Dict[str, Any]) -> Dict[str, int]:
        """글자수 제한 검증 및 위반 정보 반환"""
        violations = {}
        try:
            team_analysis = corrected_json.get('팀원_성과_분석', {})
            team_members = team_analysis.get('팀원별_기여도', [])
            
            for member in team_members:
                contribution = member.get('기여_내용', '')
                member_name = member.get('이름', 'Unknown')
                char_count = len(contribution)
                
                if char_count > 200:
                    violations[member_name] = char_count
            
        except Exception as e:
            print(f"글자수 검증 오류: {e}")
        
        return violations