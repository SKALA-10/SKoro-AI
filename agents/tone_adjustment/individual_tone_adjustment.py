
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
        "분석_코멘트",      # 업무표 배열 항목
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
        "분석_코멘트": 200,      # 업무표 각 항목
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
        """깊은 depth를 고려한 필드 추출"""
        extracted = {}
        
        def extract_recursive(data, path=""):
            """재귀적으로 모든 필드 추출 (깊은 depth 지원)"""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 타겟 키 직접 매칭 (깊은 depth에서도 작동)
                    if key in target_field_keys and isinstance(value, str) and len(value.strip()) > 10:
                        field_id = current_path
                        extracted[field_id] = value
                        print(f"    ✅ 추출: {key} -> {field_id} ({len(value)}자)")
                    
                    # 재귀 탐색 (깊은 depth 지원)
                    extract_recursive(value, current_path)
                    
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 배열 항목의 각 필드 확인 (깊은 depth 지원)
                        for key, value in item.items():
                            if key in target_field_keys and isinstance(value, str) and len(value.strip()) > 10:
                                field_id = f"{path}[{idx}].{key}"
                                extracted[field_id] = value
                                print(f"    ✅ 배열 추출: {key}[{idx}] -> {field_id} ({len(value)}자)")
                            elif isinstance(value, dict):
                                # 배열 내 중첩 객체 재귀 탐색
                                extract_recursive(value, f"{path}[{idx}].{key}")
                            elif isinstance(value, list):
                                # 배열 내 중첩 배열 재귀 탐색
                                extract_recursive(value, f"{path}[{idx}].{key}")
                    elif isinstance(item, str) and len(item.strip()) > 10:
                        # 문자열 배열 항목 (성장_포인트, 보완_영역, 추천_활동 등)
                        field_id = f"{path}[{idx}]"
                        extracted[field_id] = item
                        print(f"    ✅ 문자열 배열 추출: {path}[{idx}] -> {field_id} ({len(item)}자)")
                    elif isinstance(item, dict):
                        # 배열 내 객체 재귀 탐색
                        extract_recursive(item, f"{path}[{idx}]")
        
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
- 기존의 \n 는 유지

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
- 기존의 \n 는 유지

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
        """깊은 depth를 고려한 필드 값 설정"""
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
                            # "업무표[0].분석_코멘트" -> key="업무표", index=0, field="분석_코멘트"
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
                    # 중간 경로 탐색 (깊은 depth 지원)
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
                        # 일반 객체 탐색 (깊은 depth 지원)
                        if part not in current:
                            # 깊은 depth에서 필요한 경우 객체 생성
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

    sample_quarterly_report = {
        "기본_정보": {
            "성명": "김개발",
            "직위": "CL3",
            "소속": "W1팀",
            "업무_수행_기간": "2024년 2분기"
        },
        "팀_업무_목표_및_개인_달성률": {
            "업무표": [
                {
                    "Task명": "팀 가동률 모니터링 및 개선",
                    "핵심_Task": "업무 효율성 개선 프로세스 도입. 팀 평균 가동률 82% 달성. 이설계님과 프로세스 개선, 박DB님과 자동화 도구 협업 완료.",
                    "누적_달성률_퍼센트": 75,
                    "분석_코멘트": "현재 팀의 평균 가동률은 82%로, 목표인 85%에 근접하고 있으나 여전히 3%의 차이가 존재한다. 김개발(SK0002)님은 업무 효율성 개선 프로세스를 도입하여 팀의 성과를 높이는 데 기여하였으며, 이설계님과의 협업을 통해 프로세스 개선을, 박DB님과의 협력을 통해 자동화 도구를 성공적으로 도입하였다. 이러한 노력에도 불구하고 팀의 달성률은 75%로, 팀 평균 달성률 80.9%에 미치지 못하고 있다. 그러나 김개발님은 91점의 기여도를 기록하며 팀 내에서 중요한 역할을 수행하고 있으며, 이는 팀의 전반적인 성과 향상에 긍정적인 영향을 미치고 있다. 현재 상태를 분석할 때, 팀의 지속적인 개선 노력과 협업이 필요하며, 이를 통해 목표 달성을 위한 기반을 마련할 수 있을 것으로 보인다."
                },
                {
                    "Task명": "신규 고객 발굴 및 영업 지원",
                    "핵심_Task": "신규 계약 1건 체결 완료(3.8억원 규모). 추가 6개사와 협상 진행 중. 이설계님과 제안서 작성, 박DB님과 기술 솔루션 협업 완료.",
                    "누적_달성률_퍼센트": 55,
                    "분석_코멘트": "김개발(SK0002)님은 신규 고객 발굴 및 영업 지원 업무에서 55%의 달성률을 기록하며, 목표인 신규 계약 2건 체결에 대해 1건(3.8억원 규모)을 완료했습니다. 현재 6개사와의 협상이 진행 중이며, 이설계님과의 제안서 작성 및 박DB님과의 기술 솔루션 협업을 통해 팀의 시너지를 극대화했습니다. 팀 평균 달성률인 80.9%에 비해 다소 낮은 성과를 보였으나, 협업을 통한 기여도는 3점으로 평가됩니다. 현재 상태를 분석할 때, 김개발님은 팀의 목표 달성을 위해 지속적으로 노력하고 있으며, 향후 성과 향상을 위한 기반을 마련하고 있습니다."
                },
                {
                    "Task명": "AI 제안서 기능 고도화 개발",
                    "핵심_Task": "AI 제안서 자동 생성 핵심 모듈 50% 개발 완료. 템플릿 엔진 구현 및 테스트. 이설계님과의 코드 리뷰, 박DB님과의 데이터 연동 협업 완료.",
                    "누적_달성률_퍼센트": 50,
                    "분석_코멘트": "현재 김개발(SK0002)님은 AI 제안서 자동 생성 기능의 고도화 개발을 담당하고 있으며, 목표 달성을 위해 핵심 모듈의 50%를 완료했습니다. 이 과정에서 템플릿 엔진을 성공적으로 구현하고 테스트를 마쳤으며, 이설계님과의 코드 리뷰 및 박DB님과의 데이터 연동 협업을 통해 팀의 시너지를 극대화했습니다. 팀 평균 달성률이 80.9%인 가운데, 김개발님은 50%의 달성률을 기록하며 팀 기여도 27점을 보였습니다. 현재 상태를 분석해보면, 과거 성장 추세에 비해 다소 낮은 성과를 보이고 있으나, 협업을 통한 기술적 진전을 이루어낸 점은 긍정적으로 평가할 수 있습니다."
                }
            ],
            "개인_종합_달성률": 60,
            "종합_기여_코멘트": "김개발(SK0002)님은 현재 3개의 Task에 참여하여 평균 달성률 60.0%와 평균 기여도 40.3점을 기록하였습니다. 특히, AI 제안서 기능 고도화 개발에서 50%의 달성률을 보이며 27점의 기여도를 나타냈으나, 이는 팀 평균인 80.9%에 비해 낮은 수치입니다. 팀 가동률 모니터링 및 개선 Task에서는 75%의 달성률과 91점의 기여도를 기록하여 상대적으로 긍정적인 성과를 보였으나, 신규 고객 발굴 및 영업 지원에서는 55%의 달성률과 3점의 기여도로 낮은 기여도를 나타냈습니다. 이러한 성과를 통해 김개발님은 특정 Task에서의 강점을 보이지만, 전반적인 성과는 팀 평균에 미치지 못하는 상황입니다. 향후 성과 개선을 위해서는 각 Task의 기여도를 균형 있게 높이는 것이 필요할 것으로 보입니다.",
            "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
        },
        "Peer_Talk": {
            "강점": "해당 직원은 AI 제안서 자동 생성 모듈 개발에서 열정과 능동성을 발휘하며, 빠른 실행력으로 템플릿 엔진 구현 및 테스트를 성공적으로 완료했습니다. 또한, 책임감 있게 프로젝트의 절반을 개발하며 신뢰할 수 있는 결과를 도출했습니다.",
            "우려": "현재 분석 데이터에서는 특별한 우려사항이 발견되지 않았습니다.",
            "협업_관찰": "해당 직원은 이설계님과의 코드 리뷰 및 박DB님과의 데이터 연동 협업을 통해 개발 속도를 향상시키고 문제해결력을 발휘했습니다. 이러한 협업은 프로젝트의 성공적인 진행에 기여했습니다."
        },
        "업무_실행_및_태도": {
            "Passionate": "AI 제안서 자동 생성 핵심 모듈 50% 개발 완료, 단위 테스트 커버리지 85% 달성.",
            "Proactive": "AI 제안서 자동 생성 핵심 모듈 50% 개발 완료, 단위 테스트 커버리지 85% 달성",
            "Professional": "AI 제안서 자동 생성 핵심 모듈 50% 개발 완료 및 단위 테스트 커버리지 85% 달성",
            "People": "김개발은 팀 효율성 개선을 위해 업무 프로세스 개선과 자동화 도구 도입을 주도하여 팀 평균 가동률을 82%로 향상시켰습니다.",
            "종합_평가": "우수 수준의 4P 역량을 보유하고 있으며, 4P 영역이 매우 균형있게 발달"
        },
        "성장_제안_및_개선_피드백": {
            "성장_포인트": [
                "김개발님은 AI 제안서 자동 생성 모듈 개발 경험을 통해 복잡한 시스템 아키텍처 설계 역할로의 확장을 기대할 수 있습니다. 이는 귀하의 기술적 역량을 더욱 발전시키는 기회가 될 것입니다.",
                "프로젝트의 절반을 책임감 있게 개발한 경험은 팀 내에서 기술 리더십을 강화하는 데 큰 도움이 될 것입니다. 귀하의 경험을 바탕으로 팀원들에게 긍정적인 영향을 미칠 수 있습니다."
            ],
            "보완_영역": [
                "코드 리뷰 시 동료에게 기술적 설명을 보다 명확하게 전달하는 커뮤니케이션 스킬을 향상시키는 것은 귀하의 성장에 큰 도움이 될 것입니다. 이를 통해 팀 내 협업이 더욱 원활해질 것입니다.",
                "과도한 협업을 줄이기 위해 독립적인 문제 해결 능력을 강화하는 것도 중요합니다. 이를 통해 귀하의 자율성과 자신감을 높일 수 있습니다."
            ],
            "추천_활동": [
                "사내 기술 세미나에서 AI 제안서 시스템 개발 경험을 발표해보세요. 이는 귀하의 경험을 공유하고, 다른 팀원들과의 소통을 강화하는 좋은 기회가 될 것입니다.",
                "타팀과의 API 설계 협업 프로젝트에 참여하여 독립적인 개발 경험을 쌓아보세요. 이는 귀하의 기술적 역량을 더욱 확장하는 데 도움이 될 것입니다.",
                "주간 코드 리뷰 세션에서 피드백을 주고받으며 설명 방식 개선을 위한 실습을 진행해보세요. 이는 귀하의 커뮤니케이션 스킬을 향상시키는 데 큰 도움이 될 것입니다."
            ]
        },
        "총평": "김개발님, 이번 분기 동안의 성과에 대해 진심으로 축하드립니다! 60%의 달성률과 팀 내 1위의 성과는 귀하의 뛰어난 기여도를 잘 보여줍니다. 특히 AI 제안서 시스템 개발에서의 핵심 역할과 템플릿 엔진 구현을 통해 팀의 효율성을 크게 향상시킨 점은 매우 인상적입니다. \n\n협업률 100%는 팀워크의 중요성을 잘 나타내며, 귀하의 책임감 있는 자세가 팀에 긍정적인 영향을 미쳤음을 알 수 있습니다. 다만, 협업 편중도가 높아 과의존의 위험이 있으니, 다양한 팀원들과의 협업을 통해 균형을 맞추는 것이 좋겠습니다. \n\n앞으로는 Passionate, Proactive, Professional, People 각 영역에서의 성장을 위해 새로운 기술 습득이나 멘토링 활동을 추천드립니다. 이러한 노력이 귀하의 전문성을 더욱 강화할 것입니다. 계속해서 멋진 성과를 기대합니다!"
    }

    # 샘플 데이터 (연말 개인 레포트 - 실제 구조)
    sample_final_report = {
        "기본_정보": {
            "성명": "김개발",
            "직위": "CL3",
            "소속": "W1팀",
            "업무_수행_기간": "2024년 연말"
        },
        "최종_평가": {
            "최종_점수": 3.22,
            "업적": 3.2,
            "SK_Values": {
                "Passionate": 4.5,
                "Proactive": 4.5,
                "Professional": 4.5,
                "People": 4.5
            },
            "성과_요약": "김개발(SK0002)님은 올해 총 3개의 주요 Task에 참여하여 평균 달성률 95.0%를 기록하며 팀의 성과에 기여하였습니다. 특히, AI 제안서 기능 고도화 개발과 팀 가동률 모니터링 및 개선 Task에서 각각 100%의 달성률을 보이며, 기여도는 35점과 83점으로 팀 내에서 중요한 역할을 수행하였습니다. 신규 고객 발굴 및 영업 지원 Task에서는 85%의 달성률을 기록하였으나, 기여도는 10점으로 상대적으로 낮은 편이었습니다. 이러한 성과를 바탕으로 팀 평균 달성률 104.8%를 초과 달성하는 데 기여하였으며, 김개발님은 팀의 핵심 기여 영역에서 두드러진 성과를 보였습니다. 종합역량평가에서 김개발님은 팀의 목표 달성에 있어 중요한 역할을 하였으며, 향후에도 지속적인 성과 향상을 기대할 수 있는 기반을 마련하였습니다."
        },
        "분기별_업무_기여도": [
            {
                "분기": "1분기",
                "순위": 2,
                "달성률": 55,
                "실적_요약": "김개발(SK0002)님은 현재 3개의 Task에 참여하여 평균 달성률 53.3%와 평균 기여도 79.3점을 기록하고 있습니다. 특히, AI 제안서 기능 고도화 개발에서 20%의 달성률을 보였으나 기여도는 90점으로 높아, 해당 Task에 대한 전문성과 기여도가 두드러집니다. 반면, 신규 고객 발굴 및 영업 지원에서는 40%의 달성률과 50점의 기여도로 상대적으로 낮은 성과를 나타내고 있습니다. 팀 가동률 모니터링 및 개선 Task에서는 100%의 달성률과 98점의 기여도로 팀의 목표 달성에 크게 기여하였습니다. 이러한 성과를 통해 김개발님은 특정 분야에서의 강점을 보이고 있으나, 신규 고객 발굴과 같은 영역에서는 개선이 필요합니다. 현재 역량 평가를 통해 김개발님은 팀 평균 달성률 85.3%에 비해 낮은 성과를 보이고 있으며, 이는 향후 성장 포인트로 작용할 수 있습니다."
            },
            {
                "분기": "2분기",
                "순위": 1,
                "달성률": 60,
                "실적_요약": "김개발(SK0002)님은 현재 3개의 Task에 참여하여 평균 달성률 60.0%와 평균 기여도 40.3점을 기록하였습니다. 특히, AI 제안서 기능 고도화 개발에서 50%의 달성률을 보이며 27점의 기여도를 나타냈으나, 이는 팀 평균인 80.9%에 비해 낮은 수치입니다. 팀 가동률 모니터링 및 개선 Task에서는 75%의 달성률과 91점의 기여도를 기록하여 상대적으로 긍정적인 성과를 보였으나, 신규 고객 발굴 및 영업 지원에서는 55%의 달성률과 3점의 기여도로 낮은 기여도를 나타냈습니다. 이러한 성과를 통해 김개발님은 특정 Task에서의 강점을 보이지만, 전반적인 성과는 팀 평균에 미치지 못하는 상황입니다. 향후 성과 개선을 위해서는 각 Task의 기여도를 균형 있게 높이는 것이 필요할 것으로 보입니다."
            },
            {
                "분기": "3분기",
                "순위": 1,
                "달성률": 76,
                "실적_요약": "김개발(SK0002)님은 올해 총 3개의 Task에 참여하여 평균 달성률 76.7%와 평균 기여도 44.0점을 기록하였습니다. 특히, AI 제안서 기능 고도화 개발에서 85%의 달성률을 보이며 33점의 기여도를 나타냈고, 팀 가동률 모니터링 및 개선 Task에서는 95%의 높은 달성률과 85점의 기여도로 팀에 중요한 기여를 하였습니다. 반면, 신규 고객 발굴 및 영업 지원 Task에서는 50%의 달성률과 14점의 기여도로 상대적으로 낮은 성과를 보였습니다. 이러한 성과를 통해 김개발님은 팀 평균 달성률 92.0%에 비해 다소 낮은 성과를 기록하였으나, 가동률 개선에 대한 기여는 두드러진 점으로 평가됩니다. 현재 역량은 특정 Task에서의 높은 성과와 낮은 성과가 혼재되어 있으며, 향후 성장 포인트로는 신규 고객 발굴 및 영업 지원 분야에서의 개선이 필요할 것으로 보입니다."
            },
            {
                "분기": "4분기",
                "순위": 1,
                "달성률": 94,
                "실적_요약": "김개발(SK0002)님은 올해 총 3개의 주요 Task에 참여하여 평균 달성률 95.0%를 기록하며 팀의 성과에 기여하였습니다. 특히, AI 제안서 기능 고도화 개발과 팀 가동률 모니터링 및 개선 Task에서 각각 100%의 달성률을 보이며, 기여도는 35점과 83점으로 팀 내에서 중요한 역할을 수행하였습니다. 신규 고객 발굴 및 영업 지원 Task에서는 85%의 달성률을 기록하였으나, 기여도는 10점으로 상대적으로 낮은 편이었습니다. 이러한 성과를 바탕으로 팀 평균 달성률 104.8%를 초과 달성하는 데 기여하였으며, 김개발님은 팀의 핵심 기여 영역에서 두드러진 성과를 보였습니다. 종합역량평가에서 김개발님은 팀의 목표 달성에 있어 중요한 역할을 하였으며, 향후에도 지속적인 성과 향상을 기대할 수 있는 기반을 마련하였습니다."
            }
        ],
        "팀_업무_목표_및_개인_달성률": {
            "업무표": [
                {
                    "Task명": "AI 제안서 기능 고도화 개발",
                    "핵심_Task": "AI 제안서 자동 생성 기능 100% 완성. 실제 업무에 적용하여 제안서 작성 시간 42% 단축. 이설계님과 운영 가이드, 박DB님과 데이터 관리 협업 완료.",
                    "누적_달성률_퍼센트": 100,
                    "분석_코멘트": "2023년 동안 팀은 평균 104.8%의 성과 달성률을 기록하며, 각 구성원이 중요한 기여를 하였습니다. 특히 김개발(SK0002)님은 AI 제안서 자동 생성 기능을 100% 완성하여, 실제 업무에 적용함으로써 제안서 작성 시간을 42% 단축하는 성과를 이루었습니다. 이설계님과의 운영 가이드 협업, 박DB님과의 데이터 관리 협업을 통해 팀의 시너지를 극대화하였으며, 이러한 협력은 팀 전체의 성장에 긍정적인 영향을 미쳤습니다. 김개발님은 35점의 기여도를 기록하며 팀의 목표 달성에 중대한 역할을 하였고, 이는 팀의 연간 성과와 성장 추이를 더욱 견고히 하는 데 기여하였습니다. 전체적으로, 팀은 각자의 역할을 충실히 수행하며, 지속적인 성장을 이루어낸 한 해였습니다."
                },
                {
                    "Task명": "신규 고객 발굴 및 영업 지원",
                    "핵심_Task": "신규 계약 총 1.8건 체결 완료(총 6.5억원 규모). 신규 고객 발굴 체계 정착. 이설계님과 추천 시스템, 박DB님과 성과 분석 협업 완료.",
                    "누적_달성률_퍼센트": 85,
                    "분석_코멘트": "2023년 동안 김개발(SK0002)님은 신규 고객 발굴 및 영업 지원 분야에서 총 1.8건의 신규 계약을 체결하며 6.5억원 규모의 성과를 달성했습니다. 이는 팀 평균 달성률 104.8%에 비해 85%의 달성률을 기록한 것으로, 목표에 근접한 성과를 보여줍니다. 특히, 이설계님과의 추천 시스템 구축 및 박DB님과의 성과 분석 협업을 통해 신규 고객 발굴 체계를 정착시키는 데 기여하였습니다. 이러한 팀 내 협업은 김개발님이 팀의 성장에 긍정적인 영향을 미쳤음을 나타내며, 종합적으로 평가할 때 그의 기여도는 10점으로 평가됩니다. 전체적으로, 김개발님은 연간 성과에서 중요한 역할을 수행하며 팀의 성장 추이에 기여한 바가 큽니다."
                },
                {
                    "Task명": "팀 가동률 모니터링 및 개선",
                    "핵심_Task": "팀원별 업무 현황 파악 및 가동률 측정 시스템 구축. 현재 평균 가동률 78%. 이설계님과 측정 방법론, 박DB님과 데이터 시스템 협업 완료.",
                    "누적_달성률_퍼센트": 100,
                    "분석_코멘트": "현재 팀의 평균 가동률은 78%로, 목표인 85%에 미치지 못하고 있으나, 김개발(SK0002)님은 팀원별 업무 현황 파악 및 가동률 측정 시스템을 성공적으로 구축하여 팀의 성과를 높이는 데 기여하였습니다. 특히, 이설계님과의 협업을 통해 측정 방법론을 정립하고, 박DB님과의 협력을 통해 데이터 시스템을 완성함으로써 팀의 기여도를 98점으로 평가받았습니다. 이러한 노력에도 불구하고 현재 가동률은 목표에 미치지 못하고 있으며, 팀 평균 달성률은 85.3%로, 향후 개선이 필요한 상황입니다. 과거 성장 추이를 고려할 때, 팀의 성과는 긍정적인 방향으로 나아가고 있으나, 지속적인 모니터링과 개선이 요구됩니다."
                }
            ],
            "개인_종합_달성률": 94,
            "종합_기여_코멘트": "김개발(SK0002)님은 올해 총 3개의 주요 Task에 참여하여 평균 달성률 95.0%를 기록하며 팀의 성과에 기여하였습니다. 특히, AI 제안서 기능 고도화 개발과 팀 가동률 모니터링 및 개선 Task에서 각각 100%의 달성률을 보이며, 기여도는 35점과 83점으로 팀 내에서 중요한 역할을 수행하였습니다. 신규 고객 발굴 및 영업 지원 Task에서는 85%의 달성률을 기록하였으나, 기여도는 10점으로 상대적으로 낮은 편이었습니다. 이러한 성과를 바탕으로 팀 평균 달성률 104.8%를 초과 달성하는 데 기여하였으며, 김개발님은 팀의 핵심 기여 영역에서 두드러진 성과를 보였습니다. 종합역량평가에서 김개발님은 팀의 목표 달성에 있어 중요한 역할을 하였으며, 향후에도 지속적인 성과 향상을 기대할 수 있는 기반을 마련하였습니다."
        },
        "Peer_Talk": {
            "강점": "해당 직원은 AI 제안서 자동 생성 기능을 100% 완성하여 제안서 작성 시간을 42% 단축시키는 데 기여했습니다. 열정적이고 주도적인 리더십을 발휘하여 프로젝트 성공에 크게 기여했으며, 책임감을 바탕으로 신뢰할 수 있는 문제해결력을 보여주었습니다.",
            "우려": "현재로서는 특별히 우려할 만한 부정적인 피드백이 없습니다.",
            "협업_관찰": "해당 직원은 이설계님과의 운영 가이드 및 박DB님과의 데이터 관리 협업을 통해 열정적이고 주도적인 리더십을 발휘했습니다. 또한, 열린 소통을 통해 팀 내에서 멘토링 역할을 수행하며 협업을 촉진했습니다."
        },
        "성장_제안_및_개선_피드백": {
            "성장_포인트": [
                "김개발님은 AI 제안서 자동 생성 기능을 활용하여 복잡한 시스템 아키텍처 설계 역할을 확대할 수 있는 뛰어난 능력을 보여주셨습니다. 이는 팀 내에서의 귀하의 기여도를 더욱 높이는 데 큰 도움이 될 것입니다.",
                "팀 내 기여도를 바탕으로 신규 프로젝트의 기술 리더 역할을 수행하신 점은 정말 인상적입니다. 귀하의 리더십이 팀의 성공에 큰 영향을 미치고 있습니다."
            ],
            "보완_영역": [
                "코드리뷰 시 설명 방식을 개선하면 귀하의 지식 전달 스킬이 더욱 향상될 것입니다. 이는 팀원들에게 더 많은 가치를 제공할 수 있는 기회가 될 것입니다.",
                "과도한 협업을 줄이기 위해 독립적인 문제 해결 능력을 강화하는 것이 중요합니다. 이를 통해 귀하의 기술적 자립성이 더욱 높아질 것입니다."
            ],
            "추천_활동": [
                "사내 기술 세미나에서 AI 제안서 자동 생성 기능에 대한 발표를 1회 진행해 보세요. 이는 귀하의 전문성을 더욱 부각시킬 수 있는 좋은 기회입니다.",
                "타팀과의 API 설계 협업 프로젝트에 참여하여 독립적인 기술적 기여 경험을 쌓아보세요. 이는 귀하의 성장에 큰 도움이 될 것입니다.",
                "주간 코드리뷰 세션에서 피드백을 통해 설명 방식 개선을 위한 실습을 진행해 보세요. 이는 귀하의 지식 전달 능력을 한층 더 발전시킬 수 있는 좋은 방법입니다."
            ]
        },
        "팀장_Comment": "김개발님, 올해 정말 수고 많으셨습니다! 최종 점수 3.72점은 매우 우수한 성과를 나타내며, 특히 AI 제안서 기능 고도화와 팀 가동률 모니터링에서의 100% 달성률은 정말 대단한 기여입니다. 팀의 목표 달성에 중요한 역할을 해주셔서 감사드리며, 앞으로도 지속적인 성과 향상을 기대합니다. 신규 고객 발굴에서도 더 많은 성과를 이루실 수 있도록 함께 노력해보아요. 계속해서 발전하는 모습 기대하겠습니다!",
        "종합_Comment": "김개발님, 이번 연간 평가에서 뛰어난 성과를 보여주셔서 진심으로 축하드립니다. 94%의 달성률과 팀 내 1위의 성과는 귀하의 헌신과 전문성을 잘 나타냅니다. 특히, AI 제안서 자동 생성 기능을 완성하여 제안서 작성 시간을 42% 단축시킨 것은 팀의 효율성을 크게 향상시켰습니다. \n\n협업에서도 100%의 협업률을 기록하며 팀의 조율자로서 중요한 역할을 수행하셨습니다. 다만, 협업 편중도가 높아 과의존의 위험이 있으니, 다양한 팀원들과의 협업을 통해 균형을 맞추는 것이 향후 성장에 도움이 될 것입니다. \n\n귀하의 Passionate, Proactive, Professional, People 각 항목에서 4.5점을 기록한 것은 팀에 긍정적인 영향을 미치고 있음을 보여줍니다. 앞으로도 이러한 강점을 바탕으로 새로운 도전과 기회를 모색하시길 바랍니다."
    }


    # 진짜 LLM 클라이언트 초기화
    print("🔗 OpenAI LLM 클라이언트 초기화 중...")
    llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Agent 생성
    agent = IndividualToneAdjustmentAgent(llm_client)
    
    # 연말 개인 레포트 테스트
    # result = agent.process_report(sample_quarterly_report, "feedback_reports")
    result = agent.process_report(sample_final_report, "final_evaluation_reports")
    
    print("\n📋 처리 전")
    print(result)

    print("\n📋 연말 개인 레포트 톤 조정 완료!")
    print(result)


if __name__ == "__main__":
    test_with_real_llm()