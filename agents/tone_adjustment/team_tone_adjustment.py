import json
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ================================================================
# 팀장용 필드 매핑 설정 (실제 JSON 구조에 맞춰 수정)
# ================================================================

# 톤 조정이 필요한 필드 (팀장용 3개 타입)
TEAM_LEADER_TONE_ADJUSTMENT_FIELDS = {
    "team_feedback_reports": [  # 분기 팀장 피드백 - 경로 기반
        # "팀_종합_평가.팀_성과_분석_코멘트",           # 팀_종합_평가 하위
        # "팀_업무_목표_및_달성률.kpi_목록.kpi_분석_코멘트",              # kpi_목록 배열 항목
        # "팀_업무_목표_및_달성률.전사_유사팀_비교분석_코멘트",    # 단일 필드
        "팀원_성과_분석.팀원별_기여도.기여_내용",                    # 팀원별_기여도 배열 항목
        # "협업_네트워크.팀_협업_요약",                 # 협업_네트워크 하위
        # "협업_네트워크.협업_매트릭스.종합_평가",                    # 협업_매트릭스 배열 항목
        # "팀원별_코칭_제안.일반_코칭.다음_분기_코칭_제안",          # 일반_코칭 배열 항목
        # "팀원별_코칭_제안.집중_코칭.코칭_제안",                    # 집중_코칭 배열 항목
        # "리스크_및_향후_운영_제안.주요_리스크.리스크_설명",                 # 주요_리스크_목록 배열 항목
        # "리스크_및_향후_운영_제안.주요_리스크.운영_개선_전략_제안.전략_설명",          # 주요_리스크 하위
        # "총평.주요_인사이트"                 # 총평 하위
    ],
    
    
    "team_interim_evaluation": [  # 최종 전 중간 평가 - 경로 기반 (실제 구조에 맞춰 수정)
    #     "팀원_평가_요약표.팀_협업_요약",                    # 팀원_평가_요약표 하위
    #     "팀원_평가_요약표.표.종합_평가",                    # 표 배열 항목 (중간평가)
        "팀원별_평가_근거.AI_점수_산출_기준.업적.실적_요약",  # AI_점수_산출_기준 하위
        "팀원별_평가_근거.AI_점수_산출_기준.평가_근거_요약",  # AI_점수_산출_기준 하위
        "팀원별_평가_근거.연간_핵심_성과_기여도.Task_표.분석_코멘트",  # Task_표 배열 항목 (중간평가)
        # "팀원별_평가_근거.연간_핵심_성과_기여도.종합_기여_코멘트",  # 연간_핵심_성과_기여도 하위
        # "팀원별_평가_근거.Peer_Talk.강점",                  # Peer_Talk 하위
        # "팀원별_평가_근거.Peer_Talk.우려",                  # Peer_Talk 하위
        # "팀원별_평가_근거.Peer_Talk.협업_관찰"              # Peer_Talk 하위
    ],
    
    "team_final_reports": [  # 연말 팀 레포트 - 경로 기반 (실제 구조에 맞춰 수정)
        # "팀_종합_평가.팀_성과_분석_코멘트",          # 팀_종합_평가 하위
        # "팀_업무_목표_및_달성률.업무목표표.kpi_분석_코멘트",             # 업무목표표 배열 항목 (연말)
        # "팀_업무_목표_및_달성률.전사_유사팀_비교분석_코멘트",   # 단일 필드
        "팀_성과_요약.팀원별_성과_표.성과_요약",                    # 팀원별_성과_표 배열 항목 (연말)
        # "팀_조직력_및_리스크_요인.주요_리스크_목록.리스크_설명",                 # 주요_리스크_목록 배열 항목 (연말)
        # "팀_조직력_및_리스크_요인.주요_리스크_목록.운영_개선_전략_제안",          # 운영_개선_전략_제안 배열 항목 (연말)
        # "총평.종합_의견"                 # 총평 하위 (연말)
    ]
}

# 길이 조정이 필요한 필드 (팀장용) - 경로 기반 매칭으로 수정
TEAM_LEADER_LENGTH_ADJUSTMENT_TARGETS = {  # 분기 팀장 피드백 - 경로 기반
    "team_feedback_reports": {
        # "팀_종합_평가.팀_성과_분석_코멘트": 400,
        # "팀_업무_목표_및_달성률.kpi_목록.kpi_분석_코멘트": 250,
        # "팀_업무_목표_및_달성률.전사_유사팀_비교분석_코멘트": 300,
        "팀원_성과_분석.팀원별_기여도.기여_내용": 200,
        # "협업_네트워크.팀_협업_요약": 350,
        # "협업_네트워크.협업_매트릭스.종합_평가": 200,
        # "팀원별_코칭_제안.일반_코칭.다음_분기_코칭_제안": 200,
        # "팀원별_코칭_제안.집중_코칭.코칭_제안": 250,
        # "리스크_및_향후_운영_제안.주요_리스크.리스크_설명": 200,
        # "리스크_및_향후_운영_제안.주요_리스크.운영_개선_전략_제안.전략_설명": 200,
        # "총평.주요_인사이트": 500
    },
    
    "team_interim_evaluation": {  # 최종 전 중간 평가 - 경로 기반 (실제 구조에 맞춰 수정)
        # "팀원_평가_요약표.팀_협업_요약": 300,                    # 팀원_평가_요약표 하위
        # "팀원_평가_요약표.표.종합_평가": 250,                    # 표 배열 항목 (중간평가)
        "팀원별_평가_근거.AI_점수_산출_기준.업적.실적_요약": 200,  # AI_점수_산출_기준 하위
        "팀원별_평가_근거.AI_점수_산출_기준.평가_근거_요약": 200,  # AI_점수_산출_기준 하위
        "팀원별_평가_근거.연간_핵심_성과_기여도.Task_표.분석_코멘트": 200,  # Task_표 배열 항목 (중간평가)
        # "팀원별_평가_근거.연간_핵심_성과_기여도.종합_기여_코멘트": 300,  # 연간_핵심_성과_기여도 하위
        # "팀원별_평가_근거.Peer_Talk.강점": 150,                  # Peer_Talk 하위
        # "팀원별_평가_근거.Peer_Talk.우려": 150,                  # Peer_Talk 하위
        # "팀원별_평가_근거.Peer_Talk.협업_관찰": 150              # Peer_Talk 하위
    },
    
    "team_final_reports": {  # 연말 팀 레포트 - 경로 기반 (실제 구조에 맞춰 수정)
        # "팀_종합_평가.팀_성과_분석_코멘트": 400,
        # "팀_업무_목표_및_달성률.업무목표표.kpi_분석_코멘트": 200,  # 연말은 업무목표표
        # "팀_업무_목표_및_달성률.전사_유사팀_비교분석_코멘트": 300,
        "팀_성과_요약.팀원별_성과_표.성과_요약": 200,  # 연말은 성과_요약
        # "팀_조직력_및_리스크_요인.주요_리스크_목록.리스크_설명": 200,  # 연말은 팀_조직력_및_리스크_요인
        # "팀_조직력_및_리스크_요인.주요_리스크_목록.운영_개선_전략_제안": 200,  # 운영_개선_전략_제안 배열 항목 (연말)
        # "총평.종합_의견": 500  # 연말은 종합_의견
    }
}

# ================================================================
# 팀장용 Agent 클래스
# ================================================================

class TeamLeaderToneAdjustmentAgent:
    """팀장용 톤 조정 및 길이 조절 Agent"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.tone_fields = TEAM_LEADER_TONE_ADJUSTMENT_FIELDS
        self.length_targets = TEAM_LEADER_LENGTH_ADJUSTMENT_TARGETS
    
    def process_report(self, report_json: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """팀장용 레포트 톤 조정 및 길이 조절"""
        print(f"🎯 팀장용 레포트 톤 조정 시작: {report_type}")
        
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
        
        print(f"✅ 팀장용 톤 조정 완료: {len(adjusted_fields)}개 필드 처리")
        return result_json
    
    def extract_fields(self, report_json: Dict[str, Any], target_field_keys: set) -> Dict[str, str]:
        """팀장용 필드들 추출 (실제 JSON 구조에 맞춰 개선)"""
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
                            
                            # 배열 내부의 중첩된 구조도 재귀 탐색
                            extract_recursive(value, current_path, key)
                    elif isinstance(item, str) and parent_key in target_field_keys and len(item.strip()) > 20:
                        # 문자열 배열 항목
                        field_id = f"{path}[{idx}]"
                        extracted[field_id] = item
                        print(f"    ✅ 문자열 배열 추출: {parent_key}[{idx}] -> {field_id} ({len(item)}자)")
        
        # 일반적인 재귀 추출
        extract_recursive(report_json)
        
        # 디버깅: 실제 JSON 구조 출력
        print(f"    🔍 디버깅: target_field_keys = {target_field_keys}")
        print(f"    🔍 디버깅: 실제 JSON 키들 = {list(self._get_all_paths(report_json))}")
        
        return extracted
    
    def _get_all_paths(self, data, path="") -> set:
        """JSON의 모든 경로를 수집하는 헬퍼 함수"""
        paths = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                paths.add(current_path)
                paths.update(self._get_all_paths(value, current_path))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                current_path = f"{path}[{idx}]"
                paths.add(current_path)
                paths.update(self._get_all_paths(item, current_path))
        
        return paths
    
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
        """필드 경로에서 매칭 키 추출 (경로 기반 매칭 지원)"""
        # 배열 인덱스 제거 (예: kpi_목록[0] -> kpi_목록)
        path = re.sub(r'\[\d+\]', '', field_path)
        
        # 경로 기반 매칭을 위해 전체 경로 반환
        # 예: "팀_업무_목표_및_달성률.kpi_목록.kpi_분석_코멘트"
        return path
    
    def build_tone_and_length_prompt(self, fields_data: Dict[str, str], length_limits: Dict[str, int]) -> str:
        """팀장용 톤+길이 조정 프롬프트 생성"""
        return f"""
# 팀장용 레포트 톤 조정 + 길이 조절

## 톤 조정 규칙
1. **팀장 관점**: 팀을 이끄는 리더의 시각에서 작성
   - "팀원들이 성과를 달성했습니다"
   - "팀의 협업 성과가 우수합니다"
   - "향후 팀 운영 방향을 제시합니다"

2. **전문적이고 객관적인 톤**: 상급자에게 보고하는 격식체
   - ✅ "분석하였습니다", "평가됩니다", "제안드립니다"
   - ❌ "생각해요", "좋아요", "나쁘지 않네요"

3. **팀장용 호칭**: 팀원에 대한 적절한 호칭 사용
   - "김개발님", "이설계님" (존칭 유지)
   - "해당 직원은" → "김개발님은" 또는 "팀원명"

5. **데이터 기반 분석**: 구체적인 수치와 근거 제시
   - "달성률 95%를 기록하여"
   - "팀 평균 대비 10% 높은 성과"

## 길이 제한 (절대 준수)
{self.format_length_limits(length_limits)}

**⚠️ 매우 중요**: 각 필드는 반드시 지정된 글자수 이내로 작성해주세요.
- 기존의 \n 는 유지

## 조정할 텍스트
{self.format_fields_data(fields_data)}

위 규칙에 따라 팀장용 톤 조정 및 길이 조절해주세요.

응답 형식:
```json
{{
  "field_path_1": "조정된 텍스트",
  "field_path_2": "조정된 텍스트"
}}
```
"""

# 4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거
    
    def build_tone_only_prompt(self, fields_data: Dict[str, str]) -> str:
        """팀장용 톤 조정만 프롬프트"""
        return f"""
# 팀장용 레포트 톤 조정

## 톤 조정 규칙
1. **팀장 관점**: 팀을 이끄는 리더의 시각에서 작성
2. **전문적이고 객관적인 톤**: 상급자 보고용 격식체
3. **팀장용 호칭**: 팀원에 대한 적절한 호칭 사용
4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거
5. **데이터 기반**: 구체적인 수치와 근거 제시
6. **길이 유지**: 원본 길이를 최대한 유지

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
        """팀장용 길이 조정만 프롬프트"""
        return f"""
# 길이 조절 (팀장용 톤 유지)

## 길이 제한 (절대 준수)
{self.format_length_limits(length_limits)}

## 조정 원칙
- 팀장용 톤과 스타일은 그대로 유지
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
                SystemMessage(content="당신은 팀장용 성과 레포트의 톤과 길이를 조정하는 전문가입니다. 팀장의 관점에서 전문적이고 객관적인 톤으로 작성해야 하며, 지정된 글자수 제한을 절대 준수해야 합니다."),
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
                            # "kpi_목록[0].kpi_분석_코멘트" 형태
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
                            # "주요_리스크[0]" 형태 (문자열 배열)
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
                        # 배열 탐색: "kpi_목록[0]" -> key="kpi_목록", index=0
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
