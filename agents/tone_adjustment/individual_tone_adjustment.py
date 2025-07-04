
import json
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ================================================================
# 필드 매핑 설정 (간소화된 키 매핑으로 수정)
# ================================================================

# 톤 조정이 필요한 필드 (실제 추출되는 키로 변경)
INDIVIDUAL_TONE_ADJUSTMENT_FIELDS = {
    "feedback_reports": [
        "분석 코멘트",      # 업무표 배열 항목
        "종합_기여_코멘트",  # 단일 필드
        "총평",            # 단일 필드
        "강점",            # Peer_Talk 하위
        "우려",            # Peer_Talk 하위
        "협업_관찰",        # Peer_Talk 하위
        "종합_평가",        # 업무_실행_및_태도 하위
        "성장_포인트",      # 배열 항목
        "보완_영역",        # 배열 항목
        "추천_활동"         # 배열 항목
    ],
    
    "final_evaluation_reports": [
        "성과_요약",        # 최종_평가 하위
        "실적_요약",        # 분기별_업무_기여도 배열 항목
        "분석_코멘트",      # 업무표 배열 항목 (연말에는 분석_코멘트)
        "종합_기여_코멘트",  # 단일 필드
        "팀장_Comment",     # 단일 필드
        "종합_Comment",     # 단일 필드
        "강점",            # Peer_Talk 하위
        "우려",            # Peer_Talk 하위
        "협업_관찰",        # Peer_Talk 하위
        "성장_포인트",      # 배열 항목
        "보완_영역",        # 배열 항목
        "추천_활동"         # 배열 항목
    ]
}

# 길이 조정이 필요한 필드 (간소화된 키로 매핑)
INDIVIDUAL_LENGTH_ADJUSTMENT_TARGETS = {
    "feedback_reports": {
        "분석 코멘트": 200,      # 업무표 각 항목
        # "종합_기여_코멘트": 250,
        # "총평": 300,
        # "강점": 150,
        # "우려": 150,
        # "협업_관찰": 150,
        # "성장_포인트": 100,      # 각 배열 항목
        # "보완_영역": 100,
        # "추천_활동": 120
    },
    
    "final_evaluation_reports": {
        # "성과_요약": 300,
        "실적_요약": 200,        # 각 분기별 항목
        "분석_코멘트": 200,      # 업무표 각 항목
        # "종합_기여_코멘트": 250,
        # "팀장_Comment": 200,
        # "종합_Comment": 300,
        # "강점": 150,
        # "우려": 150,
        # "협업_관찰": 150,
        # "성장_포인트": 100,
        # "보완_영역": 100,
        # "추천_활동": 120
    }
}

# ================================================================
# Agent 클래스
# ================================================================

class IndividualToneAdjustmentAgent:
    """개인용 톤 조정 및 길이 조절 Agent (길이 제한 문제 해결)"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.tone_fields = INDIVIDUAL_TONE_ADJUSTMENT_FIELDS
        self.length_targets = INDIVIDUAL_LENGTH_ADJUSTMENT_TARGETS
    
    def process_report(self, report_json: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """개인용 레포트 톤 조정 및 길이 조절"""
        print(f"🎯 개인용 레포트 톤 조정 시작: {report_type}")
        
        # 1. 필드 분류 (간소화된 키로)
        tone_fields = set(self.tone_fields[report_type])
        length_fields = set(self.length_targets[report_type].keys())
        
        both_fields = tone_fields & length_fields          # 톤+길이 둘다
        tone_only_fields = tone_fields - length_fields     # 톤만
        length_only_fields = length_fields - tone_fields   # 길이만
        
        print(f"  📊 필드 분류:")
        print(f"    • 톤+길이 조정: {len(both_fields)}개")
        print(f"    • 톤만 조정: {len(tone_only_fields)}개")  
        print(f"    • 길이만 조정: {len(length_only_fields)}개")
        
        # 2. 배치별 처리
        adjusted_fields = {}
        
        if both_fields:
            print(f"  🔄 톤+길이 조정 배치 처리 중...")
            both_data = self.extract_fields(report_json, both_fields)
            if both_data:
                adjusted_fields.update(self.adjust_tone_and_length(both_data, report_type))
        
        if tone_only_fields:
            print(f"  🎨 톤만 조정 배치 처리 중...")
            tone_data = self.extract_fields(report_json, tone_only_fields)
            if tone_data:
                adjusted_fields.update(self.adjust_tone_only(tone_data))
        
        if length_only_fields:
            print(f"  📏 길이만 조정 배치 처리 중...")
            length_data = self.extract_fields(report_json, length_only_fields)
            if length_data:
                adjusted_fields.update(self.adjust_length_only(length_data, report_type))
        
        # 3. 원본 JSON에 조정된 내용 적용
        result_json = self.merge_back_to_json(report_json, adjusted_fields)
        
        print(f"✅ 개인용 톤 조정 완료: {len(adjusted_fields)}개 필드 처리")
        return result_json
    
    def extract_fields(self, report_json: Dict[str, Any], target_field_keys: set) -> Dict[str, str]:
        """간소화된 키로 필드들 추출"""
        extracted = {}
        
        def extract_recursive(data, path="", parent_key=""):
            """재귀적으로 모든 필드 추출"""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 타겟 키 직접 매칭
                    if key in target_field_keys and isinstance(value, str) and len(value.strip()) > 10:
                        field_id = f"{current_path}" if current_path not in extracted else f"{current_path}_{len(extracted)}"
                        extracted[field_id] = value
                        print(f"    ✅ 추출: {key} -> {field_id} ({len(value)}자)")
                    
                    # 재귀 탐색
                    extract_recursive(value, current_path, key)
                    
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 배열 항목의 각 필드 확인
                        for key, value in item.items():
                            if key in target_field_keys and isinstance(value, str) and len(value.strip()) > 10:
                                field_id = f"{path}[{idx}].{key}"
                                extracted[field_id] = value
                                print(f"    ✅ 배열 추출: {key}[{idx}] -> {field_id} ({len(value)}자)")
                    elif isinstance(item, str) and parent_key in target_field_keys and len(item.strip()) > 10:
                        # 문자열 배열 항목
                        field_id = f"{path}[{idx}]"
                        extracted[field_id] = item
                        print(f"    ✅ 문자열 배열 추출: {parent_key}[{idx}] -> {field_id} ({len(item)}자)")
        
        extract_recursive(report_json)
        return extracted
    
    def adjust_tone_and_length(self, fields_data: Dict[str, str], report_type: str) -> Dict[str, str]:
        """톤 조정 + 길이 조절 (수정된 버전)"""
        if not fields_data:
            return {}
        
        # 길이 제한 정보 수집 - 수정된 로직
        length_limits = {}
        for field_path, content in fields_data.items():
            # 필드 키 추출 (경로에서 마지막 부분)
            field_key = self.extract_field_key(field_path)
            
            if field_key in self.length_targets[report_type]:
                length_limits[field_path] = self.length_targets[report_type][field_key]
                print(f"    🔍 길이 제한 매칭: {field_path} ({field_key}) -> {self.length_targets[report_type][field_key]}자")
        
        print(f"    📏 길이 제한 설정: {length_limits}")
        
        if not length_limits:
            print("    ⚠️ 길이 제한이 매칭되지 않았습니다. 톤만 조정합니다.")
            return self.adjust_tone_only(fields_data)
        
        prompt = self.build_tone_and_length_prompt(fields_data, length_limits)
        
        try:
            response = self.llm_call(prompt)
            result = self.parse_llm_response(response, set(fields_data.keys()))
            
            # 길이 제한 확인 (강제 조정 제거)
            for field_path, content in result.items():
                if field_path in length_limits:
                    target_length = length_limits[field_path]
                    if len(content) > target_length:
                        print(f"      ⚠️ 길이 초과: {field_path} ({len(content)}자 > {target_length}자)")
                    else:
                        print(f"      ✅ 길이 적절: {field_path} ({len(content)}자 ≤ {target_length}자)")
            
            return result
        except Exception as e:
            print(f"    ❌ 톤+길이 조정 실패: {e}")
            return fields_data
    
    def adjust_tone_only(self, fields_data: Dict[str, str]) -> Dict[str, str]:
        """톤 조정만"""
        if not fields_data:
            return {}
        
        prompt = self.build_tone_only_prompt(fields_data)
        
        try:
            response = self.llm_call(prompt)
            return self.parse_llm_response(response, set(fields_data.keys()))
        except Exception as e:
            print(f"    ❌ 톤 조정 실패: {e}")
            return fields_data
    
    def adjust_length_only(self, fields_data: Dict[str, str], report_type: str) -> Dict[str, str]:
        """길이 조정만"""
        if not fields_data:
            return {}
        
        # 길이 제한 정보 수집
        length_limits = {}
        for field_path, content in fields_data.items():
            field_key = self.extract_field_key(field_path)
            if field_key in self.length_targets[report_type]:
                length_limits[field_path] = self.length_targets[report_type][field_key]
        
        prompt = self.build_length_only_prompt(fields_data, length_limits)
        
        try:
            response = self.llm_call(prompt)
            result = self.parse_llm_response(response, set(fields_data.keys()))
            
            # 길이 제한 확인 (강제 조정 제거)
            for field_path, content in result.items():
                if field_path in length_limits:
                    target_length = length_limits[field_path]
                    if len(content) > target_length:
                        print(f"      ⚠️ 길이 초과: {field_path} ({len(content)}자 > {target_length}자)")
                    else:
                        print(f"      ✅ 길이 적절: {field_path} ({len(content)}자 ≤ {target_length}자)")
            
            return result
        except Exception as e:
            print(f"    ❌ 길이 조정 실패: {e}")
            return fields_data
    
    def extract_field_key(self, field_path: str) -> str:
        """필드 경로에서 실제 키 추출"""
        # "팀_업무_목표_및_개인_달성률.업무표[0].분석 코멘트" -> "분석 코멘트"
        # "성장_제안_및_개선_피드백.성장_포인트[0]" -> "성장_포인트"
        # "총평" -> "총평"
        
        # 배열 인덱스 제거
        path = re.sub(r'\[\d+\]', '', field_path)
        
        # 마지막 점 이후의 키 반환
        if '.' in path:
            return path.split('.')[-1]
        else:
            return path
    
    def check_length_compliance(self, text: str, target_length: int) -> bool:
        """길이 준수 여부 확인"""
        return len(text) <= target_length
    
    def build_tone_and_length_prompt(self, fields_data: Dict[str, str], length_limits: Dict[str, int]) -> str:
        """톤+길이 조정 프롬프트 생성"""
        return f"""
# 개인용 레포트 톤 조정 + 길이 조절

## 톤 조정 규칙
1. **호칭 통일**: "[이름]님은"으로 일관성 있게 사용
   - "해당 직원은" → "[이름]님은" 또는 생략
   - "김개발은" → "김개발님은"

2. **격식체 사용**: 모든 문장을 격식체로 통일
   - ✅ "기록하였습니다", "보였습니다", "필요합니다"
   - ❌ "기록했다", "보였다", "필요하다"

3. **개인용 톤**: 격려적이고 동기부여적인 표현
   - "뛰어난 성과를 보여주셨습니다"
   - "더욱 발전하실 수 있는 영역입니다"
   - "지속적인 노력이 돋보입니다"

4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거

## 길이 제한 (절대 준수 - 매우 중요!)
{self.format_length_limits(length_limits)}

**⚠️ 매우 중요**: 각 필드는 반드시 지정된 글자수 이내로 작성해주세요. 
글자수를 초과하면 안 됩니다. 핵심 내용만 남기고 불필요한 부분은 제거하세요.

## 조정할 텍스트
{self.format_fields_data(fields_data)}

위 규칙에 따라 톤 조정 및 길이 조절해주세요.
각 필드별로 정확한 글자수를 맞춰주세요.

응답 형식:
```json
{{
  "field_path_1": "조정된 텍스트",
  "field_path_2": "조정된 텍스트"
}}
```
"""
    
    def build_tone_only_prompt(self, fields_data: Dict[str, str]) -> str:
        """톤 조정만 프롬프트"""
        return f"""
# 개인용 레포트 톤 조정

## 톤 조정 규칙
1. **호칭 통일**: "[이름]님은"으로 일관성 있게 사용
2. **격식체 사용**: ~습니다, ~하였습니다 등
3. **개인용 톤**: 격려적이고 동기부여적
4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거
5. **길이 유지**: 원본 길이를 최대한 유지

## 조정할 텍스트
{self.format_fields_data(fields_data)}

응답 형식:
```json
{{
  "field_path_1": "조정된 텍스트",
  "field_path_2": "조정된 텍스트"
}}
```
"""
    
    def build_length_only_prompt(self, fields_data: Dict[str, str], length_limits: Dict[str, int]) -> str:
        """길이 조정만 프롬프트"""
        return f"""
# 길이 조절 (톤 유지)

## 길이 제한 (절대 준수)
{self.format_length_limits(length_limits)}

## 조정 원칙
- 톤과 스타일은 그대로 유지
- 핵심 내용 보존하면서 압축
- 사번 제거로 글자수 절약
- 불필요한 연결어구 제거
- **⚠️ 글자수를 초과하면 안 됩니다**

## 조정할 텍스트
{self.format_fields_data(fields_data)}

응답 형식:
```json
{{
  "field_path_1": "조정된 텍스트",
  "field_path_2": "조정된 텍스트"
}}
```
"""
    
    def format_length_limits(self, length_limits: Dict[str, int]) -> str:
        """길이 제한 정보 포맷팅"""
        if not length_limits:
            return "- 길이 제한 없음"
        
        return '\n'.join([f"- {field}: {limit}자 이내 (현재: {len(self.get_current_content(field))}자)" 
                         for field, limit in length_limits.items()])
    
    def get_current_content(self, field_path: str) -> str:
        """현재 처리 중인 필드의 내용 반환 (길이 표시용)"""
        return ""  # 단순화를 위해 빈 문자열 반환
    
    def format_fields_data(self, fields_data: Dict[str, str]) -> str:
        """필드 데이터 포맷팅"""
        formatted = []
        for field_path, content in fields_data.items():
            formatted.append(f"[{field_path}] (현재: {len(content)}자)\n{content}\n")
        return '\n'.join(formatted)
    
    def llm_call(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            messages = [
                SystemMessage(content="당신은 개인용 성과 레포트의 톤과 길이를 조정하는 전문가입니다. 지정된 글자수 제한을 절대 준수해야 합니다. 글자수를 초과하면 안 됩니다."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_client.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"    ❌ LLM 호출 실패: {e}")
            raise
    
    def parse_llm_response(self, response: str, expected_fields: set) -> Dict[str, str]:
        """LLM 응답 파싱"""
        try:
            # JSON 블록 추출
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                json_text = response.strip()
            
            # JSON 파싱
            parsed = json.loads(json_text)
            
            # 예상 필드와 매칭
            result = {}
            for field in expected_fields:
                if field in parsed:
                    result[field] = parsed[field]
                    print(f"      ✅ 조정 완료: {field} ({len(parsed[field])}자)")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON 파싱 실패: {e}")
            print(f"    응답 내용: {response[:200]}...")
            raise
    
    def merge_back_to_json(self, original_json: Dict[str, Any], adjusted_fields: Dict[str, str]) -> Dict[str, Any]:
        """조정된 필드를 원본 JSON에 다시 적용"""
        result_json = json.loads(json.dumps(original_json))  # 깊은 복사
        
        for field_path, adjusted_content in adjusted_fields.items():
            self.set_field_value_by_path(result_json, field_path, adjusted_content)
        
        return result_json
    
    def set_field_value_by_path(self, data: Dict[str, Any], field_path: str, value: str):
        """경로를 사용해 필드 값 설정 (개선된 버전)"""
        try:
            print(f"      🔧 필드 설정 시도: {field_path}")
            
            # 경로를 점으로 분할
            parts = field_path.split('.')
            current = data
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # 마지막 부분 - 값 설정
                    if '[' in part and ']' in part:
                        # 배열 항목 처리
                        if '.' in part:
                            # "업무표[0].분석 코멘트" -> key="업무표", index=0, field="분석 코멘트"
                            array_match = re.match(r'([^\[]+)\[(\d+)\]\.(.+)', part)
                            if array_match:
                                array_key, index, field_name = array_match.groups()
                                if array_key in current and isinstance(current[array_key], list):
                                    idx = int(index)
                                    if idx < len(current[array_key]) and isinstance(current[array_key][idx], dict):
                                        current[array_key][idx][field_name] = value
                                        print(f"      ✅ 배열 필드 설정 성공: {field_path}")
                                    else:
                                        print(f"      ❌ 배열 항목 없음: {array_key}[{index}]")
                                else:
                                    print(f"      ❌ 배열 필드 없음: {array_key}")
                            else:
                                print(f"      ❌ 배열 패턴 매칭 실패: {part}")
                        else:
                            # "추천_활동[0]" -> key="추천_활동", index=0 (문자열 배열)
                            array_match = re.match(r'([^\[]+)\[(\d+)\]', part)
                            if array_match:
                                array_key, index = array_match.groups()
                                if array_key in current and isinstance(current[array_key], list):
                                    idx = int(index)
                                    if idx < len(current[array_key]):
                                        current[array_key][idx] = value
                                        print(f"      ✅ 문자열 배열 설정 성공: {field_path}")
                                    else:
                                        print(f"      ❌ 배열 인덱스 초과: {array_key}[{index}]")
                                else:
                                    print(f"      ❌ 배열 필드 없음: {array_key}")
                            else:
                                print(f"      ❌ 배열 패턴 매칭 실패: {part}")
                    else:
                        # 일반 필드
                        current[part] = value
                        print(f"      ✅ 일반 필드 설정 성공: {field_path}")
                else:
                    # 중간 경로 탐색
                    if '[' in part and ']' in part:
                        # 배열 탐색: "업무표[0]" -> key="업무표", index=0
                        array_match = re.match(r'([^\[]+)\[(\d+)\]', part)
                        if array_match:
                            array_key, index = array_match.groups()
                            if array_key in current and isinstance(current[array_key], list):
                                idx = int(index)
                                if idx < len(current[array_key]):
                                    current = current[array_key][idx]
                                else:
                                    print(f"      ❌ 배열 인덱스 초과: {array_key}[{index}]")
                                    return
                            else:
                                print(f"      ❌ 중간 배열 필드 없음: {array_key}")
                                return
                        else:
                            print(f"      ❌ 중간 배열 패턴 매칭 실패: {part}")
                            return
                    else:
                        # 일반 객체 탐색
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                        
        except Exception as e:
            print(f"      ❌ 필드 설정 실패 {field_path}: {e}")

# ================================================================
# 테스트 실행 함수
# ================================================================

def test_with_real_llm():
    """진짜 LLM을 사용한 테스트"""
    print("🚀 길이 제한 문제 해결된 톤 조정 Agent 테스트 시작")

    # 샘플 데이터 (분기 개인 피드백)
    sample_feedback_report = {
            "기본_정보": {
                "성명": "김개발",
                "직위": "CL3",
                "소속": "W1팀",
                "업무_수행_기간": "2024년 1분기"
            },
            "팀_업무_목표_및_개인_달성률": {
                "업무표": [
                    {
                        "Task명": "AI 제안서 기능 고도화 개발",
                        "핵심 Task": "AI 제안서 자동 생성 기능 요구사항 분석 완료. 기본 알고리즘 설계 20% 진행. 이설계님, 박DB님과의 협업을 통한 통합 설계 완성.",
                        "누적 달성률 (%)": 20,
                        "분석 코멘트": "현재 김개발(SK0002)님은 AI 제안서 자동 생성 기능의 고도화 개발을 담당하고 있으며, 목표 달성을 위해 요구사항 분석을 완료하고 기본 알고리즘 설계를 20% 진행하였습니다. 팀 평균 달성률이 85.3%인 가운데, 김개발님은 이설계님, 박DB님과의 협업을 통해 통합 설계를 완성하여 팀에 90점의 기여도를 보이고 있습니다. 현재 상태를 분석해보면, 초기 단계에서의 성과는 긍정적이나, 전체 목표에 비해 달성률이 20%로 낮은 편입니다. 이는 향후 진행 과정에서의 추가적인 노력과 협업이 필요함을 시사합니다."
                    },
                    {
                        "Task명": "팀 가동률 모니터링 및 개선",
                        "핵심 Task": "팀원별 업무 현황 파악 및 가동률 측정 시스템 구축. 현재 평균 가동률 78%. 이설계님과 측정 방법론, 박DB님과 데이터 시스템 협업 완료.",
                        "누적 달성률 (%)": 100,
                        "분석 코멘트": "현재 팀의 평균 가동률은 78%로, 목표인 85%에 미치지 못하고 있으나, 김개발(SK0002)님은 팀원별 업무 현황 파악 및 가동률 측정 시스템을 성공적으로 구축하여 팀의 성과를 높이는 데 기여하였습니다. 특히, 이설계님과의 협업을 통해 측정 방법론을 정립하고, 박DB님과의 협력을 통해 데이터 시스템을 완성함으로써 팀의 기여도를 98점으로 평가받았습니다. 이러한 노력에도 불구하고 현재 가동률은 목표에 미치지 못하고 있으며, 팀 평균 달성률은 85.3%로, 향후 개선이 필요한 상황입니다. 과거 성장 추이를 고려할 때, 팀의 성과는 긍정적인 방향으로 나아가고 있으나, 지속적인 모니터링과 개선이 요구됩니다."
                    },
                    {
                        "Task명": "신규 고객 발굴 및 영업 지원",
                        "핵심 Task": "타겟 고객 리스트 25개사 작성. 영업 전략 수립 및 초기 접촉 8개사 진행. 이설계님과 만족도 연계, 박DB님과 데이터 분석 협업 완료.",
                        "누적 달성률 (%)": 40,
                        "분석 코멘트": "김개발(SK0002)님은 신규 고객 발굴 및 영업 지원을 담당하며, 목표로 설정된 신규 계약 2건 이상 체결에 대해 현재까지 40%의 달성률을 기록하고 있습니다. 구체적으로, 타겟 고객 리스트 25개사를 작성하고, 영업 전략을 수립하여 8개사와 초기 접촉을 진행하였습니다. 또한, 이설계님과의 만족도 연계 및 박DB님과의 데이터 분석 협업을 통해 팀의 성과에 기여하였습니다. 팀 평균 달성률이 85.3%인 가운데, 김개발님은 50점의 기여도를 보이며 팀의 목표 달성에 중요한 역할을 하고 있습니다. 현재 상태를 분석할 때, 김개발님은 과거 성장 추세를 고려할 때 추가적인 성과를 기대할 수 있는 위치에 있습니다."
                    }
                ],
                "개인_종합_달성률": 55,
                "종합_기여_코멘트": "김개발(SK0002)님은 현재 3개의 Task에 참여하여 평균 달성률 53.3%와 평균 기여도 79.3점을 기록하고 있습니다. 특히, AI 제안서 기능 고도화 개발에서 20%의 달성률을 보였으나 기여도는 90점으로 높아, 해당 Task에 대한 전문성과 기여도가 두드러집니다. 반면, 신규 고객 발굴 및 영업 지원에서는 40%의 달성률과 50점의 기여도로 상대적으로 낮은 성과를 나타내고 있습니다. 팀 가동률 모니터링 및 개선 Task에서는 100%의 달성률과 98점의 기여도로 팀의 목표 달성에 크게 기여하였습니다. 이러한 성과를 통해 김개발님은 특정 분야에서의 강점을 보이고 있으나, 신규 고객 발굴과 같은 영역에서는 개선이 필요합니다. 현재 역량 평가를 통해 김개발님은 팀 평균 달성률 85.3%에 비해 낮은 성과를 보이고 있으며, 이는 향후 성장 포인트로 작용할 수 있습니다.",
                "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
            },
            "Peer_Talk": {
                "강점": "해당 직원은 AI 제안서 자동 생성 기능의 요구사항 분석을 성공적으로 완료하고, 기본 알고리즘 설계를 20% 진행하는 과정에서 열정과 문제해결력을 발휘했습니다. 특히, 능동적인 태도로 이설계님과 박DB님과의 협업을 통해 통합 설계를 완성하는 데 기여했습니다.",
                "우려": "해당 직원은 가끔 실수가 잦고 기술 이해력이 부족하여 무관심한 태도를 보이는 경향이 있습니다. 이러한 부분은 향후 프로젝트의 성공적인 진행을 위해 개선이 필요합니다.",
                "협업_관찰": "해당 직원은 이설계님과 박DB님과의 협업을 통해 통합 설계를 완성하는 과정에서 능동적으로 문제를 해결하고, 친절하게 협력하는 모습을 보였습니다. 그러나 가끔 산만한 모습을 보이기도 했습니다."
            },
            "업무_실행_및_태도": {
                "Passionate": "AI 제안서 자동 생성 시스템의 요구사항 분석을 통해 팀의 생산성 향상에 기여함.",
                "Proactive": "AI 제안서 자동 생성 시스템의 요구사항 분석을 통해 팀의 제안서 작성 시간을 대폭 단축할 수 있는 기초를 마련함.",
                "Professional": "AI 제안서 자동 생성 시스템의 요구사항 분석을 완료하고 기본 알고리즘 설계 문서를 작성하여 프로토타입 개발 진행률 20%를 달성함.",
                "People": "김개발은 이설계님과 박DB님과의 협업을 통해 통합 설계를 완성하는 과정에서 능동적으로 문제를 해결하고, 친절하게 협력하는 모습을 보였다.",
                "종합_평가": "우수 수준의 4P 역량을 보유하고 있으며, 4P 영역이 매우 균형있게 발달"
            },
            "성장_제안_및_개선_피드백": {
                "성장_포인트": [
                    "김개발님은 AI 제안서 자동 생성 기능의 요구사항 분석 경험을 바탕으로 복잡한 기술 문제를 해결하는 능력이 뛰어나며, 이를 통해 아키텍처 설계 역할을 더욱 확대할 수 있는 잠재력을 가지고 있습니다.",
                    "팀 내에서 기여도가 높은 달성률을 바탕으로 신규 프로젝트의 기술 리더 역할을 맡아 성과를 극대화할 수 있는 기회를 잘 활용하고 있습니다."
                ],
                "보완_영역": [
                    "코드 리뷰 시 설명 방식을 개선하여 동료들에게 기술적 지식을 효과적으로 전달하는 스킬을 향상시키면, 팀 전체의 기술 수준이 높아질 것입니다.",
                    "기술 이해력을 높이기 위해 정기적인 자기주도 학습 및 피드백 세션을 통해 실수를 줄이는 노력을 기울이면 더욱 성장할 수 있습니다."
                ],
                "추천_활동": [
                    "사내 기술 세미나에서 AI 제안서 자동 생성 시스템에 대한 발표를 진행하여 자신의 지식을 공유하고, 다른 동료들과의 소통을 강화해보세요.",
                    "타팀과의 API 설계 협업 프로젝트에 참여하여 실무 경험을 쌓고 기술적 이해도를 높이는 활동을 통해 다양한 시각을 얻을 수 있습니다.",
                    "주간 코드 리뷰 세션에서 동료와 함께 코드 품질을 점검하고 피드백을 주고받는 활동을 통해 서로의 성장에 기여할 수 있습니다."
                ]
            },
            "총평": "김개발님, 이번 분기 동안의 성과를 진심으로 축하드립니다. 53.33%의 달성률과 76%의 기여도로 팀 내 2위라는 우수한 성과를 기록하셨습니다. 특히 AI 제안서 자동 생성 시스템 개발에서의 핵심 개발자로서의 역할은 팀의 성공에 큰 기여를 하였습니다. 협업률 100%는 김개발님의 뛰어난 팀워크를 보여주며, 이설계님과 박DB님과의 협업을 통해 통합 설계를 완성한 점은 매우 인상적입니다.\n\n다만, 가끔 실수가 잦고 기술 이해력이 부족한 부분은 향후 프로젝트의 성공을 위해 개선이 필요합니다. 이를 위해 기술 교육이나 멘토링을 통해 전문성을 높이는 것을 추천드립니다. 앞으로도 김개발님의 열정과 문제해결력을 바탕으로 더욱 성장하는 모습을 기대합니다."
        }
    

    # 진짜 LLM 클라이언트 초기화
    print("🔗 OpenAI LLM 클라이언트 초기화 중...")
    llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Agent 생성
    agent = IndividualToneAdjustmentAgent(llm_client)
    
    result = agent.process_report(sample_feedback_report, "feedback_reports")
    
    print("\n📋 이전:")
    print(sample_feedback_report)

    print("\n📋 이후:")
    print(result)


if __name__ == "__main__":
    test_with_real_llm()