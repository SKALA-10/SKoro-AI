import json
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ================================================================
# 필드 매핑 설정 (경로 기반으로 수정)
# ================================================================

# 톤 조정이 필요한 필드 (경로 기반)
INDIVIDUAL_TONE_ADJUSTMENT_FIELDS = {
    "feedback_reports": [  # 분기 개인 피드백 - 경로 기반
        "팀_업무_목표_및_개인_달성률.업무표.분석_코멘트",  # 업무표 배열 항목
        # "팀_업무_목표_및_개인_달성률.종합_기여_코멘트",  # 단일 필드
        # "Peer_Talk.강점",                  # Peer_Talk 하위
        # "Peer_Talk.우려",                  # Peer_Talk 하위
        # "Peer_Talk.협업_관찰",              # Peer_Talk 하위
        # "업무_실행_및_태도.종합_평가",        # 업무_실행_및_태도 하위
        # "성장_제안_및_개선_피드백.성장_포인트",  # 배열 항목
        # "성장_제안_및_개선_피드백.보완_영역",    # 배열 항목
        # "성장_제안_및_개선_피드백.추천_활동",    # 배열 항목
        # "총평"                             # 단일 필드
    ],
    
    "final_evaluation_reports": [  # 연말 개인 레포트 - 경로 기반
        # "최종_평가.성과_요약",                    # 최종_평가 하위
        "분기별_업무_기여도.실적_요약",            # 분기별_업무_기여도 배열 항목
        "팀_업무_목표_및_개인_달성률.업무표.분석_코멘트",  # 업무표 배열 항목
        # "팀_업무_목표_및_개인_달성률.종합_기여_코멘트",  # 단일 필드
        # "Peer_Talk.강점",                  # Peer_Talk 하위
        # "Peer_Talk.우려",                  # Peer_Talk 하위
        # "Peer_Talk.협업_관찰",              # Peer_Talk 하위
        # "성장_제안_및_개선_피드백.성장_포인트",  # 배열 항목
        # "성장_제안_및_개선_피드백.보완_영역",    # 배열 항목
        # "성장_제안_및_개선_피드백.추천_활동",    # 배열 항목
        # "팀장_Comment",                    # 단일 필드
        # "종합_Comment"                     # 단일 필드
    ]
}

# 길이 조정이 필요한 필드 (경로 기반)
INDIVIDUAL_LENGTH_ADJUSTMENT_TARGETS = {
    "feedback_reports": {  # 분기 개인 피드백 - 경로 기반
        "팀_업무_목표_및_개인_달성률.업무표.분석_코멘트": 200,  # 업무표 각 항목
        # "팀_업무_목표_및_개인_달성률.종합_기여_코멘트": 250,  # 단일 필드
        # "총평": 300,                                       # 단일 필드
        # "Peer_Talk.강점": 150,                             # Peer_Talk 하위
        # "Peer_Talk.우려": 150,                             # Peer_Talk 하위
        # "Peer_Talk.협업_관찰": 150,                         # Peer_Talk 하위
        # "성장_제안_및_개선_피드백.성장_포인트": 100,          # 각 배열 항목
        # "성장_제안_및_개선_피드백.보완_영역": 100,            # 각 배열 항목
        # "성장_제안_및_개선_피드백.추천_활동": 120             # 각 배열 항목
    },
    
    "final_evaluation_reports": {  # 연말 개인 레포트 - 경로 기반
        # "최종_평가.성과_요약": 300,                          # 최종_평가 하위
        "분기별_업무_기여도.실적_요약": 200,                  # 각 분기별 항목
        "팀_업무_목표_및_개인_달성률.업무표.분석_코멘트": 200,  # 업무표 각 항목
        # "팀_업무_목표_및_개인_달성률.종합_기여_코멘트": 250,  # 단일 필드
        # "Peer_Talk.강점": 150,                             # Peer_Talk 하위
        # "Peer_Talk.우려": 150,                             # Peer_Talk 하위
        # "Peer_Talk.협업_관찰": 150,                         # Peer_Talk 하위
        # "성장_제안_및_개선_피드백.성장_포인트": 100,          # 각 배열 항목
        # "성장_제안_및_개선_피드백.보완_영역": 100,            # 각 배열 항목
        # "성장_제안_및_개선_피드백.추천_활동": 120,            # 각 배열 항목
        # "팀장_Comment": 200,                               # 단일 필드
        # "종합_Comment": 300                                 # 단일 필드
    }
}

# ================================================================
# 개인용 Agent 클래스
# ================================================================

class IndividualToneAdjustmentAgent:
    """개인용 톤 조정 및 길이 조절 Agent"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.tone_fields = INDIVIDUAL_TONE_ADJUSTMENT_FIELDS
        self.length_targets = INDIVIDUAL_LENGTH_ADJUSTMENT_TARGETS
    
    def process_report(self, report_json: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """개인용 레포트 톤 조정 및 길이 조절"""
        print(f"🎯 개인용 레포트 톤 조정 시작: {report_type}")
        
        # 1. 필드 분류
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
        """개인용 필드들 추출 (실제 JSON 구조에 맞춰 개선)"""
        extracted = {}
        
        def extract_recursive(data, path="", parent_key=""):
            """재귀적으로 모든 필드 추출 (경로 기반 매칭)"""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 경로 기반 매칭 (배열 인덱스 제거 후 비교)
                    path_for_matching = re.sub(r'\[\d+\]', '', current_path)
                    if path_for_matching in target_field_keys and isinstance(value, str) and len(value.strip()) > 20:
                        field_id = f"{current_path}" if current_path not in extracted else f"{current_path}_{len(extracted)}"
                        extracted[field_id] = value
                        print(f"    ✅ 경로 매칭 추출: {path_for_matching} -> {field_id} ({len(value)}자)")
                    
                    # 재귀 탐색
                    extract_recursive(value, current_path, key)
                    
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 배열 항목의 각 필드 확인 (경로 기반 매칭)
                        for key, value in item.items():
                            current_path = f"{path}[{idx}].{key}"
                            path_for_matching = re.sub(r'\[\d+\]', '', current_path)
                            
                            if path_for_matching in target_field_keys and isinstance(value, str) and len(value.strip()) > 20:
                                field_id = f"{current_path}"
                                extracted[field_id] = value
                                print(f"    ✅ 배열 경로 매칭 추출: {path_for_matching} -> {field_id} ({len(value)}자)")
                    elif isinstance(item, str) and parent_key in target_field_keys and len(item.strip()) > 20:
                        # 문자열 배열 항목
                        field_id = f"{path}[{idx}]"
                        extracted[field_id] = item
                        print(f"    ✅ 문자열 배열 추출: {parent_key}[{idx}] -> {field_id} ({len(item)}자)")
        
        # 일반적인 재귀 추출
        extract_recursive(report_json)
        
        return extracted
    
    def adjust_tone_and_length(self, fields_data: Dict[str, str], report_type: str) -> Dict[str, str]:
        """톤 조정 + 길이 조절"""
        if not fields_data:
            return {}
        
        # 길이 제한 정보 수집
        length_limits = {}
        for field_path, content in fields_data.items():
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
            
            # 길이 제한 확인
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
            
            # 길이 제한 확인
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
        """필드 경로에서 실제 키 추출 (배열 인덱스만 제거)"""
        # "팀_업무_목표_및_개인_달성률.업무표[0].분석_코멘트" -> "팀_업무_목표_및_개인_달성률.업무표.분석_코멘트"
        # "성장_제안_및_개선_피드백.성장_포인트[0]" -> "성장_제안_및_개선_피드백.성장_포인트"
        # "총평" -> "총평"
        
        # 배열 인덱스만 제거하고 전체 경로 반환
        return re.sub(r'\[\d+\]', '', field_path)
    
    def build_tone_and_length_prompt(self, fields_data: Dict[str, str], length_limits: Dict[str, int]) -> str:
        """개인용 톤+길이 조정 프롬프트 생성"""
        return f"""
# 개인용 레포트 톤 조정 + 길이 조절

## 톤 조정 규칙
1. **개인용 톤**: 격려적이고 동기부여적인 표현
   - "뛰어난 성과를 보여주셨습니다"
   - "더욱 발전하실 수 있는 영역입니다"
   - "지속적인 노력이 돋보입니다"

2. **호칭 통일**: "[이름]님은"으로 일관성 있게 사용
   - "해당 직원은" → "[이름]님은" 또는 생략
   - "김개발은" → "김개발님은"

3. **격식체 사용**: 모든 문장을 격식체로 통일
   - ✅ "기록하였습니다", "보였습니다", "필요합니다"
   - ❌ "기록했다", "보였다", "필요하다"

4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거
   - "김개발(SK0002)" → "김개발님"
   - "이설계(SK0003)" → "이설계님"  
   - "박DB(SK0004)" → "박DB님"
   - 모든 (SK0002), (SK0003), (SK0004) 패턴 완전 제거

5. **데이터 기반 분석**: 구체적인 수치와 근거 제시
   - "달성률 95%를 기록하여"
   - "팀 평균 대비 10% 높은 성과"

## 길이 제한 (절대 준수)
{self.format_length_limits(length_limits)}

**⚠️ 매우 중요**: 각 필드는 반드시 지정된 글자수 이내로 작성해주세요.
- 기존의 \n 는 유지

## 조정할 텍스트
{self.format_fields_data(fields_data)}

위 규칙에 따라 개인용 톤 조정 및 길이 조절해주세요.

응답 형식:
```json
{{
  "field_path_1": "조정된 텍스트",
  "field_path_2": "조정된 텍스트"
}}
```
"""
    
    def build_tone_only_prompt(self, fields_data: Dict[str, str]) -> str:
        """개인용 톤 조정만 프롬프트"""
        return f"""
# 개인용 레포트 톤 조정

## 톤 조정 규칙
1. **개인용 톤**: 격려적이고 동기부여적인 표현
2. **호칭 통일**: "[이름]님은"으로 일관성 있게 사용
3. **격식체 사용**: ~습니다, ~하였습니다 등
4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거
5. **데이터 기반**: 구체적인 수치와 근거 제시
6. **길이 유지**: 원본 길이를 최대한 유지
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
        """개인용 길이 조정만 프롬프트"""
        return f"""
# 길이 조절 (개인용 톤 유지)

## 길이 제한 (절대 준수)
{self.format_length_limits(length_limits)}

## 조정 원칙
- 개인용 톤과 스타일은 그대로 유지
- 핵심 내용과 데이터 보존하면서 압축 또는 확장
- 사번 제거로 글자수 절약
- 불필요한 연결어구 제거
- 기존의 \n 는 유지 
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
        
        return '\n'.join([f"- {field}: {limit}자 이내" for field, limit in length_limits.items()])
    
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
            
            # response.content가 str | list 타입일 수 있으므로 str로 변환
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                # list인 경우 첫 번째 텍스트 요소 반환
                for item in response.content:
                    if isinstance(item, str):
                        return item
                    elif isinstance(item, dict) and 'text' in item:
                        return item['text']
                return str(response.content)  # fallback
            else:
                return str(response.content)  # fallback
            
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
        """경로를 사용해 필드 값 설정"""
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
                            # "업무표[0].분석_코멘트" 형태
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
                            # "성장_포인트[0]" 형태 (문자열 배열)
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
                            print(f"      ❌ 중간 필드 없음: {part}")
                            return
                        current = current[part]
                        
        except Exception as e:
            print(f"      ❌ 필드 설정 실패 {field_path}: {e}")
