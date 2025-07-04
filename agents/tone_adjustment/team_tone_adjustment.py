import json
import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ================================================================
# 팀장용 필드 매핑 설정
# ================================================================

# 톤 조정이 필요한 필드 (팀장용 3개 타입)
TEAM_LEADER_TONE_ADJUSTMENT_FIELDS = {
    "team_feedback_reports": [  # 분기 팀장 피드백
        "팀_성과_분석_코멘트",           # 팀_종합_평가 하위
        "kpi_분석_코멘트",              # 업무목표표 배열 항목
        "전사_유사팀_비교분석_코멘트",    # 단일 필드
        "기여_내용",                   # 팀원별_기여도 배열 항목
        "종합_평가",                   # 협업_매트릭스 배열 항목
        "팀_협업_요약",                # 단일 필드
        "다음_분기_코칭_제안",          # 일반_코칭 배열 항목
        "핵심_이슈",                   # 집중_코칭 배열 항목
        "상세_분석",                   # 집중_코칭 배열 항목
        "코칭_제안",                   # 집중_코칭 배열 항목
        "리스크_설명",                 # 주요_리스크 배열 항목
        "운영_개선_전략_제안",          # 주요_리스크 하위
        "주요_인사이트"                # 총평 하위
    ],
    
    "team_interim_evaluation": [  # 최종 전 중간 평가
        "팀_협업_요약",                # 단일 필드
        "종합_평가",                   # 표 배열 항목
        "실적_요약",                   # AI_점수_산출_기준 하위
        "평가_근거_요약",              # 기본 내용 하위
        "분석_코멘트",                 # Task_표 배열 항목
        "종합_기여_코멘트",            # 단일 필드
        "강점",                       # Peer_Talk 하위
        "우려",                       # Peer_Talk 하위
        "협업_관찰"                    # Peer_Talk 하위
    ],
    
    "team_final_reports": [  # 연말 팀 레포트
        "팀_성과_분석_코멘트",          # 팀_종합_평가 하위
        "kpi_분석_코멘트",             # 업무목표표 배열 항목
        "전사_유사팀_비교분석_코멘트",   # 단일 필드
        "성과요약",                    # 팀원별_성과_표 배열 항목
        "리스크_설명",                 # 주요_리스크_목록 배열 항목
        "운영_개선_전략_제안",          # 주요_리스크 하위
        "실행_방안",                   # 핵심_인력_운용_전략 배열 항목
        "개선_방안",                   # 협업_구조_개선_방향 배열 항목
        "종합_의견"                    # 총평 하위
    ]
}

# 길이 조정이 필요한 필드 (팀장용)
TEAM_LEADER_LENGTH_ADJUSTMENT_TARGETS = {
    "team_feedback_reports": {
        # "팀_성과_분석_코멘트": 400,
        # "kpi_분석_코멘트": 250,
        # "전사_유사팀_비교분석_코멘트": 300,
        "기여_내용": 200,
        # "종합_평가": 200,
        # "팀_협업_요약": 300,
        # "다음_분기_코칭_제안": 150,
        # "핵심_이슈": 200,
        # "상세_분석": 150,
        # "코칭_제안": 250,
        # "리스크_설명": 200,
        # "운영_개선_전략_제안": 200,
        # "주요_인사이트": 400
    },
    
    "team_interim_evaluation": {
        # "팀_협업_요약": 300,
        # "종합_평가": 250,
        # "실적_요약": 300,
        "평가_근거_요약": 350,
        # "분석_코멘트": 300,
        # "종합_기여_코멘트": 300,
        # "강점": 150,
        # "우려": 150,
        # "협업_관찰": 150
    },
    
    "team_final_reports": {
        # "팀_성과_분석_코멘트": 400,
        "kpi_분석_코멘트": 200,
        # "전사_유사팀_비교분석_코멘트": 300,
        "성과요약": 200,
        # "리스크_설명": 200,
        # "운영_개선_전략_제안": 200,
        # "실행_방안": 150,
        # "개선_방안": 200,
        # "종합_의견": 500
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
        """팀장용 필드들 추출"""
        extracted = {}
        
        def extract_recursive(data, path="", parent_key=""):
            """재귀적으로 모든 필드 추출"""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 타겟 키 직접 매칭
                    if key in target_field_keys and isinstance(value, str) and len(value.strip()) > 20:
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
                            if key in target_field_keys and isinstance(value, str) and len(value.strip()) > 20:
                                field_id = f"{path}[{idx}].{key}"
                                extracted[field_id] = value
                                print(f"    ✅ 배열 추출: {key}[{idx}] -> {field_id} ({len(value)}자)")
                    elif isinstance(item, str) and parent_key in target_field_keys and len(item.strip()) > 20:
                        # 문자열 배열 항목
                        field_id = f"{path}[{idx}]"
                        extracted[field_id] = item
                        print(f"    ✅ 문자열 배열 추출: {parent_key}[{idx}] -> {field_id} ({len(item)}자)")
        
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
        """필드 경로에서 실제 키 추출"""
        # 배열 인덱스 제거
        path = re.sub(r'\[\d+\]', '', field_path)
        
        # 마지막 점 이후의 키 반환
        if '.' in path:
            return path.split('.')[-1]
        else:
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

4. **사번 제거**: (SK0002) 등 모든 사번 완전 제거

5. **데이터 기반 분석**: 구체적인 수치와 근거 제시
   - "달성률 95%를 기록하여"
   - "팀 평균 대비 10% 높은 성과"

## 길이 제한 (절대 준수)
{self.format_length_limits(length_limits)}

**⚠️ 매우 중요**: 각 필드는 반드시 지정된 글자수 이내로 작성해주세요.

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

# ================================================================
# 테스트 실행 함수
# ================================================================

def test_team_leader_tone_agent():
    """팀장용 톤 조정 Agent 테스트"""
    print("🚀 팀장용 톤 조정 Agent 테스트 시작")
    
    # 환경 변수 확인
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    # 샘플 데이터 (분기 팀장 피드백)
    sample_team_feedback = {
        "기본_정보": {
            "팀명": "W4팀",
            "팀장명": "김도현",
            "업무 수행 기간": "2025년 연말"
        },
        "팀원_평가_요약표": {
            "표": [
                {
                    "이름": "이지민(SK0028)",
                    "AI 추천 점수": 4.12,
                    "총 Task 수": 2,
                    "협업률": "100.0%",
                    "핵심 협업자": [
                        "최현진(SK0031)",
                        "박성우(SK0029)",
                        "정수빈(SK0030)"
                    ],
                    "팀 내 역할": "Cloud 아키텍처 설계 및 최적화 | 핵심 개발자",
                    "Peer Talk 평가": "리더십과 주도적인 자세, 지속적인 자기 개발 필요",
                    "협업 편중도": "높음(과의존 위험)",
                    "종합 평가": "이지민(SK0028)은 팀 내에서 핵심 개발자로서 리더형 협업 스타일을 보이며, 협업률이 100%로 매우 높은 기여도를 나타냅니다. 그러나 협업 편중도가 높아 과의존 위험이 있으므로, 독립적인 작업을 통해 자신의 역량을 더욱 강화할 필요가 있습니다."
                },
                {
                    "이름": "박성우(SK0029)",
                    "AI 추천 점수": 4.09,
                    "총 Task 수": 2,
                    "협업률": "100.0%",
                    "핵심 협업자": [
                        "정수빈(SK0030)",
                        "이지민(SK0028)",
                        "한민수(SK0032)"
                    ],
                    "팀 내 역할": "Cloud 보안 아키텍처 설계 및 운영 | 핵심 개발자",
                    "Peer Talk 평가": "리더십과 문제해결력 강함, 새로운 도전 과제 필요",
                    "협업 편중도": "높음(과의존 위험)",
                    "종합 평가": "박성우(SK0029)는 팀 내에서 핵심 개발자로서 리더형 협업 스타일을 보이며, 100%의 협업률을 기록하고 있습니다. 그러나 협업 편중도가 높아 과의존 위험이 있으므로, 다양한 팀원과의 협업을 통해 균형을 맞추는 것이 필요합니다. 팀 기여도는 높지만, 독립적인 작업을 통해 더 넓은 시각을 갖는 것이 개선 방향으로 제안됩니다."
                },
                {
                    "이름": "정수빈(SK0030)",
                    "AI 추천 점수": 3.94,
                    "총 Task 수": 2,
                    "협업률": "100.0%",
                    "핵심 협업자": [
                        "박성우(SK0029)",
                        "한민수(SK0032)",
                        "이지민(SK0028)"
                    ],
                    "팀 내 역할": "Cloud 데이터 보안 및 성능 최적화 | 핵심 개발자",
                    "Peer Talk 평가": "문제해결력과 열정 강함, 새로운 도전 과제 탐색 필요",
                    "협업 편중도": "높음(과의존 위험)",
                    "종합 평가": "정수빈(SK0030)은 팀 내에서 핵심 개발자로서 리더형 협업 스타일을 보이며, 협업률이 100%로 매우 높은 기여도를 나타냅니다. 그러나 협업 편중도가 높아 과의존의 위험이 있으므로, 독립적인 작업을 통해 다양한 관점을 수용하고 팀원들과의 균형 잡힌 협업을 강화하는 것이 필요합니다."
                },
                {
                    "이름": "최현진(SK0031)",
                    "AI 추천 점수": 3.87,
                    "총 Task 수": 2,
                    "협업률": "100.0%",
                    "핵심 협업자": [
                        "이지민(SK0028)",
                        "한민수(SK0032)",
                        "정수빈(SK0030)"
                    ],
                    "팀 내 역할": "Cloud 모니터링 및 운영 최적화 | 핵심 개발자",
                    "Peer Talk 평가": "모니터링 시스템 완성으로 운영 효율성 향상, 구체적 개선 사항 부족",
                    "협업 편중도": "높음(과의존 위험)",
                    "종합 평가": "최현진(SK0031)은 리더형 협업 스타일을 보이며, 팀 내에서 핵심 개발자로서 높은 기여도를 보여주고 있습니다. 그러나 협업 편중도가 높아 과의존 위험이 있으므로, 다양한 팀원과의 협업을 통해 균형 잡힌 협업을 지향하는 것이 필요합니다."
                },
                {
                    "이름": "한민수(SK0032)",
                    "AI 추천 점수": 4.03,
                    "총 Task 수": 2,
                    "협업률": "100.0%",
                    "핵심 협업자": [
                        "박성우(SK0029)",
                        "정수빈(SK0030)",
                        "최현진(SK0031)"
                    ],
                    "팀 내 역할": "Cloud 시스템 보안 및 운영 자동화 | 핵심 개발자",
                    "Peer Talk 평가": "문제해결력과 성실함, 기술 습득 필요성",
                    "협업 편중도": "높음(과의존 위험)",
                    "종합 평가": "한민수(SK0032)는 팀 내에서 핵심 개발자로서 리더형 협업 스타일을 보이며, 100%의 협업률을 기록하고 있습니다. 그러나 협업 편중도가 높아 과의존 위험이 있으므로, 독립적인 작업을 통해 자신의 역량을 더욱 강화할 필요가 있습니다. 팀 내 다양한 의견을 수렴하고, 자율성을 높이는 방향으로 개선을 제안합니다."
                },
                {
                    "이름": "윤지영(SK0033)",
                    "AI 추천 점수": 3.61,
                    "총 Task 수": 2,
                    "협업률": "50.0%",
                    "핵심 협업자": [
                        "한민수(SK0032)"
                    ],
                    "팀 내 역할": "Cloud 운영 프로세스 자동화 | 핵심 개발자",
                    "Peer Talk 평가": "업무 효율성 향상, 지속적인 기술 학습 필요",
                    "협업 편중도": "높음(과의존 위험)",
                    "종합 평가": "윤지영(SK0033) 직원은 핵심 개발자로서 Cloud 운영 프로세스 자동화에 기여하고 있으며, 협업 스타일은 서포터형으로 평가됩니다. 그러나 협업률이 50%로 낮고, 한민수(SK0032)에게 과도하게 의존하는 경향이 있어 팀 기여도는 보통 수준입니다. 다양한 팀원과의 협업을 통해 의존도를 줄이고, 협업률을 높이는 것이 필요합니다."
                }
            ],
            "팀_협업_요약": "팀의 전체 평균 협업률이 91.7%로 매우 높은 수준을 유지하고 있으며, 이는 팀원들이 효과적으로 소통하고 협력하고 있음을 나타냅니다. 그러나, 기여도가 평균 41.3점으로 상대적으로 낮고, 특히 이지민, 박성우, 정수빈, 최현진, 한민수, 윤지영 등 6명의 팀원이 과의존 위험에 처해 있어, 특정 멤버에게 의존하는 경향이 강하다는 점은 우려스럽습니다. 이러한 편중된 협업 패턴은 팀의 지속 가능성을 저해할 수 있으므로, 팀원 간의 역할 분담을 재조정하고, 윤지영과 같은 기여도가 높은 멤버의 협업률을 높이는 방안을 모색하는 것이 필요합니다.",
            "협업률_설명": "개인이 수행한 전체 업무 중, 다른 팀원과 실제로 협업한 업무가 차지하는 비율입니다.",
            "협업_편중도_설명": "특정 동료에게만 협업이 쏠려있는지, 혹은 여러 동료와 고르게 협업하는지를 나타내는 지표입니다."
        },
        "팀원별_평가_근거": [
            {
                "기본_내용": {
                    "이름": "이지민",
                    "직무": "Cloud Architect",
                    "CL레벨": "CL3"
                },
                "AI_점수_산출_기준": {
                    "업적": {
                        "점수": 3.7,
                        "실적 요약": "이지민(SK0028)님은 Cloud Architect로서 연간 성과에서 평균 달성률 105.5%를 기록하며 팀의 성과 중심 목표에 기여하였습니다. 특히, 고가용성 설계와 성능 최적화 두 가지 주요 Task에서 각각 100%와 111%의 달성률을 보이며, 각 Task에서 33점의 기여도를 기록하였습니다. 이러한 성과는 팀 전체 평균 달성률 139.4%에 비추어 볼 때, 이지민님이 팀의 핵심 기여 영역에서 중요한 역할을 수행했음을 나타냅니다. 분기별 성장 추이를 살펴보면, 이지민님은 지속적으로 높은 성과를 유지하며, 팀의 목표 달성에 기여한 바가 큽니다. 종합적으로 평가할 때, 이지민님은 Cloud 아키텍처 분야에서 뛰어난 역량을 발휘하며, 팀의 성과에 긍정적인 영향을 미친 것으로 판단됩니다. 이러한 성과는 향후에도 지속적인 발전 가능성을 보여주며, 팀의 전반적인 역량 강화에 기여할 것으로 기대됩니다."
                    },
                    "SK_Values": {
                        "Passive": {
                            "점수": 5.0,
                            "평가 근거": "이지민은 고가용성 아키텍처 설계와 성능 최적화 프로젝트에서 각각 100% 완료를 달성하며, 목표를 초과하는 성과를 지속적으로 보여주었습니다. 또한, 업무 외에도 혁신적인 성과를 도출하기 위해 특허 출원과 인증을 추진하는 등 압도적인 몰입과 헌신을 보여주었습니다."
                        },
                        "Proactive": {
                            "점수": 5.0,
                            "평가 근거": "이지민은 Cloud 아키텍처의 고가용성과 성능 최적화를 위해 주도적으로 프로젝트를 이끌며, 혁신적인 솔루션을 제안하고 실행했습니다. 특히, 고가용성 아키텍처 설계와 성능 최적화에서 뛰어난 성과를 달성하여 조직의 방향성을 제시하고, 새로운 비즈니스 기회를 창출하는 데 기여했습니다."
                        },
                        "Professional": {
                            "점수": 4.5,
                            "평가 근거": "이지민은 Cloud 아키텍처 분야에서 뛰어난 전문성을 발휘하며, 고가용성 아키텍처 설계와 성능 최적화 프로젝트를 성공적으로 수행하였습니다. 특히, 99.97%의 가용성을 달성하고, 업계 컨퍼런스에서 발표하여 외부 전문가로서의 입지를 다졌습니다. 또한, 팀원들과의 협업을 통해 다양한 기술적 문제를 해결하며 팀의 전문성을 높이는 데 기여하였습니다."
                        },
                        "People": {
                            "점수": 4.5,
                            "평가 근거": "이지민은 팀 내에서 핵심 개발자로서 높은 협업률을 기록하며, 팀 목표 달성에 기여하고 있습니다. 고가용성 아키텍처 설계와 성능 최적화 프로젝트에서 동료들과의 협업을 통해 성공적인 결과를 도출하였고, 문서화 및 특허 출원 등의 성과를 이룬 점에서 팀워크와 시너지 창출에 기여한 것으로 평가됩니다. 그러나 동료 평가에서 구체적인 개선 사항이 부족하다는 점은 아쉬운 부분입니다."
                        }
                    },
                    "종합 원점수": 4.12,
                    "AI 추천 점수 (CL 정규화)": 4.12,
                    "평가 근거 요약": "이지민(SK0028)님은 Cloud Architect로서 연간 달성률 104%를 기록하여 A등급에 해당하며, 원시 하이브리드 점수는 4.12점으로 평가되었습니다. 이지민님은 고가용성 설계와 성능 최적화 두 가지 주요 Task에서 각각 100%와 111%의 달성률을 보이며, 팀의 성과 중심 목표에 기여하였습니다. 특히, 팀 전체 평균 달성률 139.4%에 비추어 볼 때, 이지민님은 팀의 핵심 기여 영역에서 중요한 역할을 수행했음을 나타냅니다. 4P 평가 결과에서 Passionate와 Proactive 부문에서 각각 5.0점을 기록하였고, Professional과 People 부문에서도 높은 점수를 받았습니다. 이러한 성과는 이지민님이 팀의 목표 달성에 긍정적인 영향을 미쳤음을 보여주며, 향후에도 지속적인 발전 가능성을 기대할 수 있습니다."
                },
                "연간_핵심_성과_기여도": {
                    "Task_표": [
                        {
                            "Task명": "Cloud 아키텍처 고가용성 설계",
                            "핵심 Task": "고가용성 아키텍처 설계 100% 완료, 99.97% 가용성 달성",
                            "누적 달성률 (%)": 100,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히, 이지민(SK0028)님은 핵심 시스템의 가용성을 99.95%로 설정한 목표를 초과 달성하여 99.97%의 가용성을 실현하였고, 고가용성 아키텍처 설계를 100% 완료함으로써 팀의 성과에 크게 기여하였습니다. 이지민님은 팀 내에서 33점의 기여도를 기록하며, 전체 목표 달성에 중요한 역할을 하였습니다. 이러한 성과는 팀의 협력과 각 구성원의 전문성이 결합된 결과로, 향후에도 지속적인 성장을 기대할 수 있는 기반이 마련되었습니다."
                        },
                        {
                            "Task명": "Cloud 아키텍처 성능 최적화",
                            "핵심 Task": "아키텍처 성능 최적화 100% 완료, 응답시간 8.9% 단축",
                            "누적 달성률 (%)": 111,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히, 이지민(SK0028)님은 Cloud 아키텍처 성능 최적화 프로젝트에서 100%의 목표 달성을 이루어내며 시스템 응답시간을 8.9% 단축시키는 성과를 거두었습니다. 이로 인해 이지민님은 팀 내에서 33점의 기여도를 기록하며 중요한 역할을 수행하였습니다. 이러한 성과는 팀 전체의 협력과 각 구성원의 전문성이 결합된 결과로, 향후에도 지속적인 성장을 기대할 수 있는 기반이 마련되었습니다. 종합적으로, 팀의 연간 성과는 각 개인의 기여가 집합되어 이루어진 결과로, 향후에도 이러한 성과를 유지하고 발전시킬 수 있는 가능성을 보여줍니다."
                        }
                    ],
                    "개인 종합 달성률": 47,
                    "종합 기여 코멘트": "이지민(SK0028)님은 Cloud Architect로서 두 가지 주요 Task에 참여하였으며, 전체 평균 달성률은 47.5%로 팀 평균 86.4%에 비해 낮은 수치를 기록하였습니다. 특히, Cloud 아키텍처 고가용성 설계에서 48%의 달성률과 34점의 기여도를 보였고, 성능 최적화 Task에서는 47%의 달성률과 41점의 기여도를 기록하였습니다. 이러한 성과는 이지민님이 맡은 역할의 복잡성과 도전성을 반영하며, 현재의 역량은 향후 개선이 필요한 상황임을 나타냅니다. 전반적으로, 이지민님은 두 개의 Task에서 고르게 낮은 성과를 보였으나, 성능 최적화에서 상대적으로 높은 기여도를 보여주어 향후 성장 가능성을 엿볼 수 있습니다. 현재의 성과를 바탕으로 향후 더 나은 결과를 도출하기 위한 전략적 접근이 필요할 것으로 보입니다."
                },
                "Peer_Talk": {
                    "강점": "해당 직원은 고가용성 아키텍처 설계의 48%를 성공적으로 완료하고 이중화 구성을 완성하여 프로젝트의 안정성을 크게 향상시켰습니다. 이는 프로젝트의 안정성에 중요한 기여를 한 것으로 평가됩니다.",
                    "우려": "현재 제공된 데이터에서는 구체적인 개선 사항이 언급되지 않았습니다. 추가적인 피드백이 필요할 수 있습니다.",
                    "협업 관찰": "해당 직원은 프로젝트의 중요한 부분을 책임지고 성공적으로 수행하여 팀의 목표 달성에 기여하였습니다. 이는 협업과 책임감 있는 태도를 보여주는 사례입니다."
                }
            },
            {
                "기본_내용": {
                    "이름": "박성우",
                    "직무": "Security Engineer",
                    "CL레벨": "CL3"
                },
                "AI_점수_산출_기준": {
                    "업적": {
                        "점수": 3.9,
                        "실적 요약": "박성우(SK0029)님은 Security Engineer로서 연간 성과에서 평균 달성률 107.0%를 기록하며 팀의 목표 달성에 기여하였습니다. 특히, Cloud 보안 아키텍처 안정성 강화 작업에서 99%의 높은 달성률을 보였고, 이는 팀의 핵심 기여 영역 중 하나로 평가됩니다. 또한, Cloud 보안 취약점 분석 및 대응 Task에서는 115%의 달성률을 기록하며 41점의 기여도를 나타내어, 박성우님이 맡은 두 개의 Task에서 모두 긍정적인 성과를 도출하였습니다. 이러한 성과는 팀 평균 달성률 139.4%에 기여하며, 박성우님 개인의 기여도가 평균 37.0점에 달하는 것을 보여줍니다. 분기별 성장 추이를 살펴보면, 박성우님은 지속적으로 안정적인 성과를 유지하며 팀의 전반적인 역량 강화에 기여하고 있습니다. 종합적으로, 박성우(SK0029)님은 팀의 목표 달성에 중요한 역할을 하였으며, 그의 전문성과 헌신은 향후에도 팀의 성과에 긍정적인 영향을 미칠 것으로 기대됩니다."
                    },
                    "SK_Values": {
                        "Passive": {
                            "점수": 4.5,
                            "평가 근거": "박성우는 Cloud 보안 아키텍처의 안정성을 강화하기 위해 다양한 프로젝트에 적극적으로 참여하였으며, 목표를 초과 달성하는 성과를 지속적으로 보여주었습니다. 특히, 보안 사고를 제로로 만드는 성과는 그의 열정과 헌신을 잘 나타냅니다. 또한, 팀원들과의 협업을 통해 성과를 극대화하고, 보안 취약점 대응에서 우수성을 인정받은 점은 그의 주인의식을 잘 보여줍니다."
                        },
                        "Proactive": {
                            "점수": 4.5,
                            "평가 근거": "박성우는 Cloud 보안 아키텍처의 안정성을 강화하기 위해 다양한 혁신적인 아이디어를 제안하고 실행하였으며, 특히 Zero Trust 아키텍처 설계와 AI 기반 위협 예측 시스템 구축을 통해 조직의 보안 수준을 크게 향상시켰습니다. 이러한 성과는 연간 1억원 이상의 비용 절감 및 수익 창출에 기여할 수 있는 잠재력을 가지고 있습니다."
                        },
                        "Professional": {
                            "점수": 4.5,
                            "평가 근거": "박성우는 Cloud 보안 아키텍처와 취약점 분석 체계를 성공적으로 구축하여 조직의 보안 수준을 크게 향상시켰습니다. 특히, 99.95%의 보안 가용성을 달성하고 Zero-Day 취약점에 대한 완벽한 대응 체계를 구축한 점은 그의 전문성을 잘 보여줍니다. 또한, 다양한 최신 기술을 활용하여 팀의 생산성을 높이고, 동료들과의 협업을 통해 효과적인 보안 체계를 마련한 점에서 높은 평가를 받을 수 있습니다."
                        },
                        "People": {
                            "점수": 4.0,
                            "평가 근거": "박성우는 팀 내에서 핵심 개발자로서 뛰어난 협업 능력을 보여주었으며, 팀 목표 달성에 적극적으로 기여했습니다. 협업률이 100%로 매우 높고, 다양한 프로젝트에서 동료들과의 협업을 통해 성과를 이끌어냈습니다. 그러나 협업 편중도가 높아 과의존 위험이 있으므로, 다양한 팀원과의 협업을 통해 균형 잡힌 협업을 지향하는 것이 필요합니다."
                        }
                    },
                    "종합 원점수": 4.09,
                    "AI 추천 점수 (CL 정규화)": 4.09,
                    "평가 근거 요약": "박성우(SK0029)님은 Security Engineer로서 연간 달성률 108%를 기록하여 A등급에 해당하며, 최종 정규화 점수는 4.09점입니다. 이는 팀 내 CL3에서의 원시 하이브리드 점수 4.09점이 유지된 결과로, 달성률 점수 3.9점과 4P 점수 4.38점이 결합된 것입니다. 특히, Cloud 보안 아키텍처 안정성 강화 작업에서 99%의 높은 달성률을 보였고, Cloud 보안 취약점 분석 및 대응 Task에서는 115%의 달성률을 기록하여 팀의 목표 달성에 기여하였습니다. 박성우님은 팀 평균 달성률 139.4%에 기여하며, 개인 기여도는 평균 37.0점에 달합니다. 이러한 성과는 팀의 전반적인 역량 강화에 긍정적인 영향을 미쳤으며, 향후에도 지속적인 성과 향상이 기대됩니다."
                },
                "연간_핵심_성과_기여도": {
                    "Task_표": [
                        {
                            "Task명": "Cloud 보안 아키텍처 안정성 강화",
                            "핵심 Task": "보안 아키텍처 100% 완료, 99.95% 보안 가용성 달성",
                            "누적 달성률 (%)": 99,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히, 박성우(SK0029)님은 Cloud 보안 아키텍처의 안정성 강화를 통해 보안 시스템 가용성 99.9% 달성을 목표로 하였고, 실제로 100%의 아키텍처 완료와 99.95%의 보안 가용성을 달성하여 팀에 중요한 기여를 하였습니다. 그의 기여도는 33점으로 평가되며, 이는 팀 전체 성과에 긍정적인 영향을 미쳤습니다. 이러한 성과는 팀의 지속적인 성장과 발전을 뒷받침하는 중요한 요소로 작용하였으며, 향후에도 이러한 성과를 기반으로 한 안정적인 운영이 기대됩니다."
                        },
                        {
                            "Task명": "Cloud 보안 취약점 분석 및 대응",
                            "핵심 Task": "보안 취약점 대응 46% 완료, Zero-Day 완벽 대응 체계 구축",
                            "누적 달성률 (%)": 115,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히 박성우(SK0029)님은 보안 취약점 분석 및 대응 업무에서 40%의 목표를 설정하고, 실제로는 46%의 보안 취약점 대응을 완료하여 115%의 달성률을 기록했습니다. 또한, Zero-Day 대응 체계를 완벽하게 구축함으로써 팀의 전반적인 기여도에 41점을 부여받았습니다. 이러한 성과는 팀의 목표 달성에 크게 기여하였으며, 각 구성원의 전문성과 협업이 돋보인 한 해였습니다. 종합적으로, 팀은 지속적인 성장과 성과 향상을 이루어내며, 향후에도 이러한 추세를 유지할 것으로 기대됩니다."
                        }
                    ],
                    "개인 종합 달성률": 46,
                    "종합 기여 코멘트": "박성우(SK0029)님은 Security Engineer로서 두 개의 주요 Task에 참여하여 평균 달성률 46.5%와 평균 기여도 34.0점을 기록했습니다. 특히, Cloud 보안 아키텍처 안정성 강화와 Cloud 보안 취약점 분석 및 대응 Task에서 각각 48%와 45%의 달성률을 보이며, 팀 평균 달성률 86.4%에 비해 다소 낮은 성과를 나타냈습니다. 이러한 결과는 박성우님이 맡은 Task의 복잡성과 도전적인 성격을 반영할 수 있으며, 향후 안정성 강화와 취약점 분석에 대한 보다 심층적인 접근이 필요할 것으로 보입니다. 현재 역량 평가에 따르면, 박성우님은 기여도 면에서 팀 내에서 평균적인 성과를 보이고 있으나, Task 수행의 효율성을 높이기 위한 추가적인 전략이 요구됩니다. 전반적으로, 박성우님은 향후 성장을 위한 기회가 있는 상황이며, 현재의 성과를 바탕으로 지속적인 발전이 기대됩니다."
                },
                "Peer_Talk": {
                    "강점": "해당 직원은 보안 아키텍처의 48%를 성공적으로 완료하며 다층 보안 체계 구축에 기여하여 프로젝트의 중요한 부분을 책임감 있게 수행하고 있습니다. 이는 프로젝트의 중요한 진전을 이루는 데 큰 역할을 했습니다.",
                    "우려": "현재 분석 데이터에서 부정적인 키워드가 명시되지 않았습니다. 추가적인 피드백이 필요할 수 있습니다.",
                    "협업 관찰": "해당 직원은 프로젝트의 중요한 부분을 책임감 있게 수행하며, 다층 보안 체계 구축에 기여하여 팀의 목표 달성에 긍정적인 영향을 미치고 있습니다."
                }
            },
            {
                "기본_내용": {
                    "이름": "한민수",
                    "직무": "System Engineer",
                    "CL레벨": "CL1"
                },
                "AI_점수_산출_기준": {
                    "업적": {
                        "점수": 3.5,
                        "실적 요약": "한민수(SK0032)님은 2023년 동안 두 가지 주요 Task에 참여하여 평균 달성률 100.5%와 평균 기여도 37.0점을 기록하였습니다. 특히, Cloud 시스템 보안 모니터링 Task에서는 108%의 달성률을 보이며 24점의 기여도를 나타냈고, Cloud 운영 자동화 시스템 구축 Task에서는 93%의 달성률과 50점의 기여도를 기록하여 팀의 핵심 기여 영역에서 중요한 역할을 수행하였습니다. 이러한 성과는 팀 전체 평균 달성률 139.4%에 기여하며, 팀원 7명 중에서 한민수님이 보여준 성과는 팀의 종합 역량 평가에 긍정적인 영향을 미쳤습니다. 과거 분기별 성장 추이를 살펴보면, 한민수님은 지속적으로 안정적인 성과를 유지하며 팀의 목표 달성에 기여해왔습니다. 이러한 성과는 향후에도 팀의 발전에 중요한 기반이 될 것으로 평가됩니다."
                    },
                    "SK_Values": {
                        "Passive": {
                            "점수": 4.5,
                            "평가 근거": "한민수는 보안 및 운영 자동화 시스템 구축에 있어 높은 주인의식을 가지고 있으며, 목표를 초과 달성하기 위해 지속적으로 노력한 점이 두드러집니다. 특히, 보안 모니터링과 운영 자동화에서 각각 27%와 42%의 성과를 달성하며, 관련 분야에서 우수상을 수상한 것은 그의 열정과 헌신을 잘 보여줍니다. 또한, 팀원들과의 협업을 통해 복잡한 문제를 해결하고, 주기적인 검토를 통해 지속적인 개선을 도모한 점도 긍정적입니다."
                        },
                        "Proactive": {
                            "점수": 4.5,
                            "평가 근거": "한민수 직원은 보안 모니터링과 운영 자동화 시스템 구축에 있어 주도적으로 혁신적인 아이디어를 실행하였으며, 이를 통해 조직의 효율성을 크게 향상시켰습니다. 특히, 보안 모니터링 체계와 운영 자동화 시스템의 성공적인 구축은 연간 5,000만원 이상의 비용 절감 및 수익 창출에 기여한 것으로 평가됩니다. 또한, 문제를 사전에 예측하고 선제적으로 대응 방안을 마련하여 팀과 조직의 변화를 주도한 점이 높이 평가됩니다."
                        },
                        "Professional": {
                            "점수": 4.5,
                            "평가 근거": "한민수는 Cloud 시스템의 보안 및 운영 자동화 분야에서 뛰어난 성과를 보여주었으며, 특히 보안 모니터링과 운영 자동화 시스템 구축에서 높은 수준의 전문성을 발휘했습니다. 다양한 기술을 활용하여 복잡한 문제를 해결하고, 팀원들과의 협업을 통해 성과를 극대화했습니다. 또한, 수상 경력과 자격증 취득은 그의 전문성을 더욱 강화하는 요소입니다."
                        },
                        "People": {
                            "점수": 4.0,
                            "평가 근거": "한민수는 팀 내에서 핵심 개발자로서 높은 협업률을 기록하며, 동료들과의 협업을 통해 중요한 프로젝트를 성공적으로 수행했습니다. 그러나 동료 평가에서 구체적인 개선 사항이 부족하다는 점이 지적되었으며, 이는 다양한 팀원과의 협업을 통해 더욱 균형 잡힌 협업 환경을 조성할 필요가 있음을 나타냅니다."
                        }
                    },
                    "종합 원점수": 4.03,
                    "AI 추천 점수 (CL 정규화)": 4.03,
                    "평가 근거 요약": "한민수(SK0032)님은 2023년 동안 두 가지 주요 Task에 참여하여 평균 달성률 100.5%를 기록하였으며, 이는 A등급에 해당하는 3.5점으로 평가됩니다. 특히, Cloud 시스템 보안 모니터링 Task에서 108%의 달성률을 보이며 24점의 기여도를 나타냈고, Cloud 운영 자동화 시스템 구축 Task에서는 93%의 달성률과 50점의 기여도를 기록하였습니다. 이러한 성과는 팀 전체 평균 달성률 139.4%에 긍정적인 영향을 미쳤습니다. 4P 평가 결과에서 Passionate, Proactive, Professional 항목에서 각각 4.5점을 기록하였고, People 항목에서 4.0점을 기록하여 평균 4.38점으로 평가되었습니다. 최종 정규화 점수는 4.03점으로, 이는 팀 내 CL1에서 원시 점수를 유지한 결과입니다. 한민수님은 지속적으로 안정적인 성과를 유지하며 팀의 목표 달성에 기여해왔으며, 향후에도 팀의 발전에 중요한 기반이 될 것으로 평가됩니다."
                },
                "연간_핵심_성과_기여도": {
                    "Task_표": [
                        {
                            "Task명": "Cloud 시스템 보안 모니터링",
                            "핵심 Task": "보안 모니터링 27% 완료, 100% 실시간 탐지 달성",
                            "누적 달성률 (%)": 108,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 달성률을 기록하며 뛰어난 성과를 보였습니다. 특히 한민수(SK0032)님은 시스템 보안 모니터링 분야에서 25%의 목표를 맡아 Zero-Day 대응에 기여하였으며, 누적 성과로는 보안 모니터링 27% 완료와 100% 실시간 탐지를 달성하여 108%의 달성률을 기록했습니다. 이러한 성과는 팀 전체의 성장 추이에 긍정적인 영향을 미쳤으며, 한민수님은 팀 기여도에서 24점을 기록하여 팀의 목표 달성에 중요한 역할을 하였습니다. 종합적으로, 팀의 성장은 각 구성원의 노력과 기여가 결합된 결과로, 향후에도 이러한 성과를 지속적으로 이어갈 수 있는 기반을 마련하였습니다."
                        },
                        {
                            "Task명": "Cloud 운영 자동화 시스템 구축",
                            "핵심 Task": "운영 자동화 시스템 100% 구축, 자동화율 42% 달성",
                            "누적 달성률 (%)": 93,
                            "분석 코멘트": "2023년 동안 팀은 연간 목표를 초과 달성하며 눈에 띄는 성과를 기록했습니다. 팀 평균 달성률은 139.4%에 달하며, 7명의 구성원이 각자의 역할에서 기여한 결과입니다. 특히, 한민수(SK0032)님은 운영 자동화 시스템을 100% 구축하고, 목표 자동화율 45%에 근접한 42%를 달성하여 팀의 성장에 중요한 역할을 했습니다. 그의 기여도는 50점으로 평가되며, 이는 팀 전체 성과에 긍정적인 영향을 미쳤습니다. 이러한 성과는 과거 성장 추이를 반영하며, 팀의 지속적인 발전 가능성을 보여줍니다. 종합적으로, 올해의 성과는 팀의 협력과 개인의 노력이 결합된 결과로, 향후에도 이러한 성과를 지속적으로 이어갈 수 있는 기반을 마련했습니다."
                        }
                    ],
                    "개인 종합 달성률": 58,
                    "종합 기여 코멘트": "한민수(SK0032)님은 현재 두 개의 주요 Task에 참여하여 평균 달성률 58.0%와 평균 기여도 44.5점을 기록하였습니다. Cloud 시스템 보안 모니터링 Task에서는 72%의 달성률을 보이며 34점을 기여하였고, Cloud 운영 자동화 시스템 구축 Task에서는 44%의 달성률로 55점을 기여하였습니다. 이러한 성과는 팀 평균 달성률 86.4%에 비해 다소 낮은 편이나, 각 Task에서의 기여도는 상대적으로 균형을 이루고 있습니다. 특히, 보안 모니터링 Task에서의 성과는 향후 보안 강화에 중요한 역할을 할 것으로 기대됩니다. 그러나 운영 자동화 시스템 구축에서의 낮은 달성률은 향후 개선이 필요한 성장 포인트로 지적될 수 있습니다. 현재 한민수님은 두 개의 Task에 집중하고 있으며, 이를 통해 향후 더 높은 성과를 달성할 수 있는 기반을 마련하고 있습니다."
                },
                "Peer_Talk": {
                    "강점": "해당 직원은 보안 모니터링 작업의 18%를 성공적으로 완료하고 실시간 위협 탐지 시스템 구축에 기여하여 보안 강화에 중요한 역할을 수행하고 있습니다. 이는 보안 분야에서의 전문성과 책임감을 보여주는 중요한 성과입니다.",
                    "우려": "현재 제공된 데이터에서는 구체적인 개선 사항이 명시되지 않았습니다. 추가적인 피드백이 필요할 수 있습니다.",
                    "협업 관찰": "해당 직원은 보안 모니터링과 실시간 위협 탐지 시스템 구축에 기여하며 팀의 보안 강화에 중요한 역할을 수행하고 있습니다. 이는 팀 내에서의 협업과 기여도를 잘 보여줍니다."
                }
            },
            {
                "기본_내용": {
                    "이름": "정수빈",
                    "직무": "Data Engineer",
                    "CL레벨": "CL2"
                },
                "AI_점수_산출_기준": {
                    "업적": {
                        "점수": 3.5,
                        "실적 요약": "정수빈(SK0030)님은 올해 두 가지 주요 Task에 참여하여 평균 달성률 100.5%를 기록하며 팀의 성과에 기여하였습니다. 특히, Cloud 데이터 보안 파이프라인 구축에서 105%의 달성률을 보이며 33점의 기여도를 기록하였고, Cloud 데이터 처리 성능 개선 Task에서도 96%의 달성률을 달성하여 동일한 기여도 33점을 유지했습니다. 이러한 성과는 팀 평균 달성률 139.4%에 비해 다소 낮은 수치이나, 정수빈님이 맡은 Task의 특성과 난이도를 고려할 때 긍정적인 평가를 받을 만합니다. 분기별 성장 추이를 살펴보면, 정수빈님은 지속적으로 안정적인 성과를 유지하며 팀의 핵심 기여 영역에서 중요한 역할을 수행하고 있습니다. 종합적으로 평가할 때, 정수빈님은 팀 내에서 중요한 데이터 엔지니어로 자리매김하고 있으며, 향후에도 이러한 성과를 지속적으로 이어갈 가능성이 높습니다."
                    },
                    "SK_Values": {
                        "Passive": {
                            "점수": 4.5,
                            "평가 근거": "정수빈은 데이터 보안 및 처리 성능 개선을 위해 상당한 시간과 노력을 투자하였으며, 그 결과로 우수한 성과를 달성했습니다. 특히, 컴플라이언스 요구사항을 100% 충족시키고, 성능 개선으로 우수상을 수상한 점은 그의 열정과 헌신을 잘 보여줍니다. 또한, 팀원들과의 협업을 통해 지속적으로 개선 기회를 발굴하고 실행한 점도 긍정적입니다."
                        },
                        "Proactive": {
                            "점수": 4.5,
                            "평가 근거": "정수빈은 데이터 보안 및 처리 성능 개선을 위해 적극적으로 새로운 기술을 도입하고, 팀원들과 협력하여 혁신적인 솔루션을 실행했습니다. 특히, 데이터 보안 파이프라인 구축과 관련하여 100% 컴플라이언스를 달성한 점은 매우 긍정적이며, 성과가 조직에 미친 영향이 큽니다. 그러나 연간 5,000만원 이상의 비용 절감이나 수익 창출에 대한 구체적인 수치가 부족하여 4.5점을 부여합니다."
                        },
                        "Professional": {
                            "점수": 4.5,
                            "평가 근거": "정수빈은 데이터 엔지니어로서 클라우드 환경에서 데이터 보안 및 성능 개선을 위한 다양한 프로젝트를 성공적으로 수행하였으며, 특히 데이터 보안 파이프라인 구축과 관련하여 100% 컴플라이언스를 달성한 점은 그의 전문성을 잘 보여줍니다. 또한, 성능 개선을 통해 응답시간을 단축시키고, 관련 인증을 취득하여 자신의 전문성을 지속적으로 확장하고 있습니다. 이러한 성과들은 그가 팀 내에서 선진 기술을 도입하고 전파하는 데 기여하고 있음을 나타냅니다."
                        },
                        "People": {
                            "점수": 4.0,
                            "평가 근거": "정수빈은 팀 내에서 핵심 개발자로서 100%의 협업률을 기록하며 팀 목표 달성에 기여하고 있습니다. 동료들과의 협업을 통해 성과를 문서화하고, 데이터 보안 및 성능 개선 프로젝트에서 우수한 결과를 도출하였습니다. 그러나 동료 평가에서 구체적인 개선 사항이 부족하다는 점이 지적되었으며, 이는 팀워크를 더욱 강화하기 위한 추가적인 소통과 협력이 필요함을 시사합니다."
                        }
                    },
                    "종합 원점수": 3.94,
                    "AI 추천 점수 (CL 정규화)": 3.94,
                    "평가 근거 요약": "정수빈(SK0030)님은 2023년 동안 두 가지 주요 Task에 참여하여 평균 달성률 100.5%를 기록하며 팀의 성과에 기여하였습니다. 달성률 점수는 3.5점으로 A등급에 해당하며, 이는 목표를 달성한 것을 의미합니다. 4P 평가에서는 평균 4.38점을 기록하였으며, Passionate, Proactive, Professional 항목에서 각각 4.5점을, People 항목에서 4.0점을 받았습니다. 최종 정규화 점수는 3.94점으로, 팀 내 CL2에서 원시 점수를 유지하였습니다. 정수빈님은 Cloud 데이터 보안 파이프라인 구축에서 105%의 달성률을 기록하며 33점의 기여도를 보였고, Cloud 데이터 처리 성능 개선 Task에서도 96%의 달성률을 달성하여 동일한 기여도를 유지했습니다. 팀 평균 달성률 139.4%에 비해 다소 낮은 수치이나, 맡은 Task의 특성과 난이도를 고려할 때 긍정적인 평가를 받을 만합니다. 지속적으로 안정적인 성과를 유지하며 팀의 핵심 기여 영역에서 중요한 역할을 수행하고 있습니다."
                },
                "연간_핵심_성과_기여도": {
                    "Task_표": [
                        {
                            "Task명": "Cloud 데이터 보안 파이프라인 구축",
                            "핵심 Task": "데이터 보안 파이프라인 37% 완료, 100% 컴플라이언스 달성",
                            "누적 달성률 (%)": 105,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히 정수빈(SK0030)님은 데이터 보안 파이프라인 구축 프로젝트에서 35%의 목표를 맡아 Zero-Day 대응에 기여하였으며, 누적 성과로는 데이터 보안 파이프라인을 37% 완료하고 100%의 컴플라이언스를 달성하여 105%의 달성률을 기록했습니다. 이러한 성과는 팀 전체의 기여도와 협업의 결과로, 정수빈님은 33점의 기여도를 평가받았습니다. 종합적으로, 팀의 성장은 각 구성원의 노력과 성과가 결합된 결과로, 향후에도 지속적인 발전이 기대됩니다."
                        },
                        {
                            "Task명": "Cloud 데이터 처리 성능 개선",
                            "핵심 Task": "데이터 처리 성능 개선 100% 완료, 응답시간 6.8% 단축",
                            "누적 달성률 (%)": 96,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히 정수빈(SK0030)님은 클라우드 데이터 처리 성능 개선 프로젝트를 통해 목표했던 데이터 처리 응답시간 7% 단축을 목표로 하였으나, 최종적으로 6.8% 단축을 달성하여 96%의 목표 달성률을 기록했습니다. 이 과정에서 정수빈님은 팀 내에서 33점의 기여도를 보이며, 팀 전체 성과에 중요한 역할을 하였습니다. 이러한 성과는 팀의 종합적인 성장에 기여하였으며, 향후에도 지속적인 발전이 기대됩니다."
                        }
                    ],
                    "개인 종합 달성률": 45,
                    "종합 기여 코멘트": "정수빈(SK0030)님은 Data Engineer로서 두 개의 주요 Task에 참여하여 평균 달성률 45.0%와 평균 기여도 29.5점을 기록하였습니다. Cloud 데이터 보안 파이프라인 구축과 Cloud 데이터 처리 성능 개선 Task에서 각각 45%의 달성률을 보이며, 기여도는 30점과 29점으로 나타났습니다. 이는 팀 평균 달성률 86.4%에 비해 상대적으로 낮은 수치로, 향후 성과 개선이 필요한 상황입니다. 정수빈님은 현재 두 개의 Task에 집중하고 있으며, 이로 인해 전체적인 성과에 기여하고 있으나, 보다 높은 성과를 위해 추가적인 성장 포인트를 모색할 필요가 있습니다. 현재 역량 평가에 따르면, 정수빈님은 특정 Task에서 기여도가 다소 낮은 편이므로, 향후 데이터 처리 및 보안 관련 기술 향상에 집중할 경우 성과 개선이 기대됩니다."
                },
                "Peer_Talk": {
                    "강점": "해당 직원은 데이터 보안 파이프라인의 16%를 성공적으로 완료하고 암호화 시스템 구축에 기여하여 프로젝트의 초기 단계에서 중요한 역할을 수행했습니다. 이는 프로젝트의 성공에 기여하는 중요한 성과로 평가됩니다.",
                    "우려": "현재 제공된 데이터에서는 구체적인 개선 사항이 언급되지 않았습니다. 추가적인 피드백이 필요할 수 있습니다.",
                    "협업 관찰": "해당 직원은 프로젝트의 초기 단계에서 중요한 역할을 수행하며 팀의 목표 달성에 기여하고 있습니다. 이는 협업과 팀워크에 긍정적인 영향을 미친 것으로 보입니다."
                }
            },
            {
                "기본_내용": {
                    "이름": "최현진",
                    "직무": "DevOps Engineer",
                    "CL레벨": "CL2"
                },
                "AI_점수_산출_기준": {
                    "업적": {
                        "점수": 3.35,
                        "실적 요약": "최현진(SK0031)님은 DevOps Engineer로서 연간 성과에서 평균 달성률 97.0%를 기록하며 팀의 목표 달성에 기여하였습니다. 특히, Cloud 모니터링 및 장애 대응 분야에서 90%의 달성률을 보였고, 이는 팀의 안정적인 운영에 중요한 역할을 했습니다. 또한, Cloud 운영 성능 최적화 작업에서는 104%의 달성률을 기록하여 성과를 더욱 높였습니다. 이러한 성과는 팀 평균 달성률 139.4%에 기여하며, 최현진님은 두 개의 주요 Task에 참여하여 평균 기여도 33.0점을 기록했습니다. 이는 팀 내에서 핵심 기여 영역으로 평가되며, 최현진님은 팀의 종합 역량 평가에서도 긍정적인 영향을 미쳤습니다. 전반적으로, 최현진님은 팀의 목표 달성에 있어 중요한 역할을 수행하며, 향후에도 지속적인 성과를 기대할 수 있는 인재로 평가됩니다."
                    },
                    "SK_Values": {
                        "Passive": {
                            "점수": 4.5,
                            "평가 근거": "최현진은 Cloud 시스템의 복잡성 문제를 해결하기 위해 다양한 프로젝트에 적극적으로 참여하였으며, 각 프로젝트에서 목표를 초과 달성하는 성과를 보여주었습니다. 특히, 평균 장애 대응 시간을 단축하고 가용성을 높이는 데 기여한 점은 그의 열정과 헌신을 잘 보여줍니다. 또한, 동료들과의 협업을 통해 성과를 문서화하고 공유하는 모습은 높은 주인의식을 나타냅니다."
                        },
                        "Proactive": {
                            "점수": 4.5,
                            "평가 근거": "최현진은 Cloud 시스템의 복잡성 문제를 해결하기 위해 통합 모니터링 체계를 구축하고, 운영 성능 최적화 프로젝트를 통해 응답 시간을 단축하는 등 능동적으로 문제를 해결하고 있습니다. 또한, 새로운 기술을 도입하여 시스템의 효율성을 높이고, 팀과 협력하여 성과를 도출하는 모습이 인상적입니다. 이러한 점에서 혁신적 아이디어를 실행하는 데 기여하고 있습니다."
                        },
                        "Professional": {
                            "점수": 4.5,
                            "평가 근거": "최현진은 Cloud 시스템의 복잡성을 해결하기 위해 다양한 모니터링 및 최적화 프로젝트를 성공적으로 수행하였으며, 이를 통해 팀의 생산성과 효율성을 크게 향상시켰습니다. 또한, 최신 기술을 도입하고 동료들과의 협업을 통해 지식을 공유하며 팀의 전문성을 높이는 데 기여하였습니다. 그러나 외부 전문가 세미나 발표 횟수나 Q&A 해결률에 대한 정보가 부족하여 5.0점에는 미치지 못합니다."
                        },
                        "People": {
                            "점수": 4.0,
                            "평가 근거": "최현진은 팀 내에서 핵심 개발자로서 높은 협업률을 기록하며, 동료들과의 협업을 통해 중요한 성과를 달성했습니다. 그러나 협업 편중도가 높아 과의존 위험이 있다는 점에서, 다양한 팀원과의 협업을 통해 균형 잡힌 협업을 지향할 필요가 있습니다."
                        }
                    },
                    "종합 원점수": 3.87,
                    "AI 추천 점수 (CL 정규화)": 3.87,
                    "평가 근거 요약": "최현진(SK0031)님은 DevOps Engineer로서 연간 성과에서 평균 달성률 97.0%를 기록하여 B등급(목표 근접)에 해당하는 3.35점을 부여받았습니다. 4P 평가에서는 Passionate, Proactive, Professional에서 각각 4.5점을, People에서 4.0점을 기록하여 평균 4.38점을 달성하였습니다. 최종 정규화 점수는 3.87점으로, 팀 내 CL2에서 원시 점수를 유지하며 팀의 목표 달성에 기여한 바가 큽니다. 특히, Cloud 모니터링 및 장애 대응 분야에서 90%의 달성률을 보였고, Cloud 운영 성능 최적화 작업에서는 104%의 달성률을 기록하여 팀의 안정적인 운영에 중요한 역할을 했습니다. 이러한 성과는 팀 평균 달성률 139.4%에 기여하며, 최현진님은 두 개의 주요 Task에 참여하여 평균 기여도 33.0점을 기록했습니다. 전반적으로, 최현진님은 팀의 목표 달성에 있어 중요한 역할을 수행하며, 향후에도 지속적인 성과를 기대할 수 있는 인재로 평가됩니다."
                },
                "연간_핵심_성과_기여도": {
                    "Task_표": [
                        {
                            "Task명": "Cloud 모니터링 및 장애 대응",
                            "핵심 Task": "모니터링 시스템 100% 완성, 평균 장애 대응 시간 1.8시간",
                            "누적 달성률 (%)": 90,
                            "분석 코멘트": "2023년 동안 최현진(SK0031)님은 클라우드 모니터링 및 장애 대응 분야에서 두드러진 성과를 달성하였습니다. 연간 목표인 99.9% 가용성에 기여하기 위해 설정된 장애 대응 시간 2시간 이내를 초과 달성하여 평균 1.8시간의 장애 대응 시간을 기록하였고, 이는 팀 평균 달성률 139.4%에 기여하는 중요한 요소로 작용하였습니다. 최현진님은 모니터링 시스템을 100% 완성하여 팀의 전반적인 성과 향상에 기여하였으며, 이로 인해 팀 내 기여도는 33점으로 평가되었습니다. 이러한 성과는 과거 성장 추이를 반영하며, 최현진님은 팀의 목표 달성에 있어 중요한 역할을 수행하였습니다. 종합적으로 볼 때, 최현진님은 뛰어난 성과를 통해 팀의 성장에 기여한 바가 크며, 향후에도 이러한 성과가 지속될 것으로 기대됩니다."
                        },
                        {
                            "Task명": "Cloud 운영 성능 최적화",
                            "핵심 Task": "운영 성능 최적화 100% 완료, 응답시간 5.2% 단축",
                            "누적 달성률 (%)": 104,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 달성률을 기록하며 뛰어난 성과를 보였습니다. 특히 최현진(SK0031)님은 운영 프로세스 응답시간을 5% 단축하는 목표를 설정하고, 이를 초과 달성하여 5.2%의 응답시간 단축을 이뤄냈습니다. 이로 인해 운영 성능 최적화 작업이 100% 완료되었으며, 최현진님은 팀 내에서 33점의 기여도를 기록하여 팀의 성장에 중요한 역할을 했습니다. 이러한 성과는 팀 전체의 성장 추이를 반영하며, 각 구성원의 기여가 팀 목표 달성에 긍정적인 영향을 미쳤음을 보여줍니다. 종합적으로, 올해의 성과는 팀의 협력과 개인의 노력으로 이루어진 결과로, 향후에도 지속적인 성장을 기대할 수 있는 기반이 마련되었습니다."
                        }
                    ],
                    "개인 종합 달성률": 68,
                    "종합 기여 코멘트": "최현진(SK0031)님은 DevOps Engineer로서 두 가지 주요 Task에 참여하여 전체적으로 68.0%의 평균 달성률을 기록했습니다. 특히 Cloud 모니터링 및 장애 대응 Task에서는 90%의 높은 달성률을 보이며 31점의 기여도를 기록, 팀 내에서 중요한 역할을 수행했습니다. 반면, Cloud 운영 성능 최적화 Task에서는 46%의 달성률로 다소 낮은 성과를 나타냈으며, 이 부분은 향후 개선이 필요한 성장 포인트로 평가됩니다. 최현진님은 총 2개의 Task에 참여하여 평균 기여도 30.0점을 기록하였으며, 이는 팀 평균 기여도와 유사한 수준입니다. 현재 역량 평가를 통해 볼 때, 강점은 장애 대응 능력에 있으며, 운영 성능 최적화 분야에서의 추가적인 역량 개발이 필요할 것으로 보입니다. 이러한 분석을 통해 최현진님은 팀의 전반적인 성과에 긍정적인 영향을 미치고 있으나, 특정 Task에서의 성과 향상이 요구됩니다."
                },
                "Peer_Talk": {
                    "강점": "해당 직원은 모니터링 시스템을 45% 완성하는 과정에서 평균 장애 대응 시간을 2.3시간으로 유지하며 신속하고 효율적인 문제 해결 능력을 보여주었습니다. 이는 문제 발생 시 빠르게 대응하고 해결하는 데 있어 뛰어난 역량을 가지고 있음을 나타냅니다.",
                    "우려": "현재 제공된 데이터에서는 구체적인 개선 사항이 언급되지 않았습니다. 추가적인 피드백이 필요할 수 있습니다.",
                    "협업 관찰": "해당 직원은 모니터링 시스템 개발 과정에서 신속한 문제 해결 능력을 발휘하며 팀의 목표 달성에 기여했습니다. 이는 팀 내에서의 협업과 조율 능력이 뛰어남을 시사합니다."
                }
            },
            {
                "기본_내용": {
                    "이름": "윤지영",
                    "직무": "Junior Engineer",
                    "CL레벨": "CL1"
                },
                "AI_점수_산출_기준": {
                    "업적": {
                        "점수": 2.85,
                        "실적 요약": "윤지영(SK0033)님은 올해 두 가지 주요 과제에 참여하여 평균 달성률 87.5%와 평균 기여도 75.0점을 기록하였습니다. 특히, Cloud 신기술 연구 및 도입에서 95%의 높은 달성률을 보이며 100점의 기여도를 달성한 것은 팀의 기술적 발전에 크게 기여한 것으로 평가됩니다. 반면, Cloud 운영 프로세스 자동화에서는 80%의 달성률과 50점의 기여도를 기록하였으나, 이는 향후 개선의 여지가 있는 부분으로 분석됩니다. 팀 전체 평균 달성률이 139.4%에 달하는 가운데, 윤지영님은 두 개의 Task에 참여하여 팀의 성과에 긍정적인 영향을 미쳤습니다. 종합적으로 볼 때, 윤지영님은 핵심 기여 영역에서 뛰어난 성과를 보였으며, 팀의 목표 달성에 중요한 역할을 한 것으로 평가됩니다. 이러한 성과는 향후 더욱 발전할 수 있는 기반이 될 것으로 기대됩니다."
                    },
                    "SK_Values": {
                        "Passive": {
                            "점수": 4.5,
                            "평가 근거": "윤지영은 Cloud 운영 프로세스의 복잡성을 해결하기 위해 스스로 프로세스 자동화에 착수하고, 주어진 업무 외에도 주기적인 스터디를 통해 동료들과 협력하여 성과를 도출했습니다. 또한, 신기술 도입과 관련하여 5.7건의 신기술을 실제 서비스에 적용하여 기술 혁신을 이끌어냈습니다. 이러한 성과는 높은 주인의식과 지속적인 노력의 결과로, 4.5점의 평가를 부여합니다."
                        },
                        "Proactive": {
                            "점수": 4.5,
                            "평가 근거": "윤지영은 Cloud 운영 프로세스의 복잡성을 해결하기 위해 능동적으로 프로세스 자동화에 착수하고, 다양한 신기술을 연구 및 도입하여 팀의 경쟁력을 높였습니다. 특히, 자동화율을 28%까지 끌어올리고, 신기술 도입을 통해 기술 혁신상을 수상하는 등 뛰어난 성과를 보여주었습니다. 이러한 점에서 혁신적 아이디어를 실행하고, 팀 및 조직의 변화를 주도하는 모습이 두드러집니다."
                        },
                        "Professional": {
                            "점수": 4.5,
                            "평가 근거": "윤지영은 Cloud 운영 프로세스 자동화와 신기술 도입에 있어 뛰어난 성과를 보여주었으며, 특히 신기술 5.7건을 도입하고 특허를 출원하는 등 기술 혁신을 이끌었습니다. 또한, 프로세스 자동화율 28%를 달성하고 RPA 기초 인증을 취득하여 전문성을 입증했습니다. 동료와의 협업 및 주기적인 기술 스터디를 통해 팀의 전문성을 높이는 데 기여하고 있습니다."
                        },
                        "People": {
                            "점수": 3.0,
                            "평가 근거": "윤지영은 팀 내에서 독립적으로 업무를 수행하며 높은 기여도를 보이고 있지만, 협업률이 50%로 낮아 다양한 팀원과의 협업이 부족한 상황입니다. 팀워크를 극대화하기 위한 소통이나 협력의 적극성이 부족하여, 기본적인 협업 수준에 머물러 있습니다."
                        }
                    },
                    "종합 원점수": 3.61,
                    "AI 추천 점수 (CL 정규화)": 3.61,
                    "평가 근거 요약": "윤지영(SK0033)님은 2023년 동안 두 가지 주요 과제에 참여하여 평균 달성률 87.5%를 기록하였으며, 이는 B등급에 해당합니다. 특히, Cloud 신기술 연구 및 도입에서 95%의 높은 달성률을 보이며 100점의 기여도를 달성한 것은 팀의 기술적 발전에 크게 기여한 것으로 평가됩니다. 반면, Cloud 운영 프로세스 자동화에서는 80%의 달성률과 50점의 기여도를 기록하였으나, 이는 향후 개선의 여지가 있는 부분으로 분석됩니다. 4P 평가 결과에서 Passionate, Proactive, Professional 항목에서 각각 4.5점을 기록하였고, People 항목에서 3.0점을 기록하여 평균 4.12점을 달성하였습니다. 최종 정규화 점수는 3.61점으로, 팀 내 CL1에서 원시 점수를 유지하며 팀의 성과에 긍정적인 영향을 미쳤습니다. 종합적으로 볼 때, 윤지영님은 핵심 기여 영역에서 뛰어난 성과를 보였으며, 팀의 목표 달성에 중요한 역할을 한 것으로 평가됩니다."
                },
                "연간_핵심_성과_기여도": {
                    "Task_표": [
                        {
                            "Task명": "Cloud 운영 프로세스 자동화",
                            "핵심 Task": "프로세스 자동화 100% 구축, 자동화율 28% 달성",
                            "누적 달성률 (%)": 80,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히, 윤지영(SK0033)님은 운영 프로세스 자동화 목표를 설정하고, 누적 성과로 100%의 프로세스 자동화 구축을 이루어냈습니다. 그러나 자동화율은 28%에 그쳐 목표인 35%에는 미치지 못했으며, 이로 인해 달성률은 80%로 평가되었습니다. 윤지영님은 팀 내에서 50점의 기여도를 기록하며, 팀의 전반적인 성과에 긍정적인 영향을 미쳤습니다. 이러한 성과는 팀의 지속적인 성장과 발전에 기여하였으며, 향후 더 높은 목표 달성을 위한 기반을 마련하였습니다."
                        },
                        {
                            "Task명": "Cloud 신기술 연구 및 도입",
                            "핵심 Task": "신기술 도입 5.7건 완료, 특허 2건 출원",
                            "누적 달성률 (%)": 95,
                            "분석 코멘트": "2023년 동안 팀은 평균 139.4%의 성과 달성률을 기록하며 눈에 띄는 성장 추이를 보였습니다. 특히, 윤지영(SK0033)님은 신기술 연구 및 도입을 담당하며 목표인 6건 중 5.7건을 완료하고 2건의 특허를 출원하여 기여도 100점을 달성했습니다. 이는 팀 전체의 성과에 긍정적인 영향을 미쳤으며, 각 팀원들이 협력하여 높은 목표를 달성하는 데 기여한 결과입니다. 이러한 성과는 팀의 전문성과 헌신을 반영하며, 향후에도 지속적인 성장을 기대할 수 있는 기반이 마련되었습니다. 종합적으로, 2023년은 팀의 역량을 강화하고 성과를 극대화한 해로 평가됩니다."
                        }
                    ],
                    "개인 종합 달성률": 45,
                    "종합 기여 코멘트": "윤지영(SK0033)님은 현재 Cloud 운영 프로세스 자동화와 Cloud 신기술 연구 및 도입 두 가지 주요 Task에 참여하고 있으며, 전체 평균 달성률은 45.0%로 팀 평균 86.4%에 비해 낮은 수치를 보이고 있습니다. 특히, Cloud 운영 프로세스 자동화에서 45%의 달성률과 44점의 기여도를 기록하였고, Cloud 신기술 연구 및 도입에서는 45%의 달성률과 100점의 기여도를 달성하여, 후자의 Task에서 두드러진 성과를 나타냈습니다. 이러한 성과는 윤지영님이 신기술 도입에 대한 높은 기여도를 보였음을 시사하지만, 전체적인 Task 수행에서 개선이 필요함을 보여줍니다. 현재까지의 참여 Task 수는 2개로, 이는 향후 더 많은 프로젝트에 참여하여 경험을 쌓고 성과를 향상시킬 수 있는 성장 포인트로 작용할 수 있습니다. 전반적으로 윤지영님은 특정 분야에서 강점을 보이고 있으나, 전체적인 성과 향상을 위해서는 추가적인 노력이 요구됩니다."
                },
                "Peer_Talk": {
                    "강점": "해당 직원은 프로세스 자동화 프로젝트에서 16%의 자동화율을 성공적으로 구축하여 팀의 효율성을 크게 향상시켰습니다. 이는 팀의 업무 효율성을 높이는 데 중요한 기여를 했으며, 평가 비중 4로 반영될 만한 성과입니다.",
                    "우려": "현재 분석 데이터에서는 구체적인 우려사항이 드러나지 않았습니다. 추가적인 피드백이 필요할 수 있습니다.",
                    "협업 관찰": "해당 직원은 팀의 효율성을 높이는 데 기여하였으며, 이는 팀 내 협업에 긍정적인 영향을 미쳤습니다. 자동화 프로젝트를 통해 팀의 업무 효율성을 향상시키는 데 중요한 역할을 했습니다."
                }
            }
        ]
    }

    # LLM 클라이언트 초기화
    print("🔗 OpenAI LLM 클라이언트 초기화 중...")
    llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Agent 생성
    agent = TeamLeaderToneAdjustmentAgent(llm_client)
    
    # 테스트 실행
    result = agent.process_report(sample_team_feedback, "team_feedback_reports")
    # result = agent.process_report(sample_interim_feedback, "team_interim_evaluation")
    # result = agent.process_report(sample_final_feedback, "team_final_reports")
    
    print("\n📋 이전 결과:")
    # print(json.dumps(sample_team_feedback, ensure_ascii=False, indent=2))
    print(sample_team_feedback)


    print("\n📋 처리 결과:")
    print(result)

if __name__ == "__main__":
    test_team_leader_tone_agent()