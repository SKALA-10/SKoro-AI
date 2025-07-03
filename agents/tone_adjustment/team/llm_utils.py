"""
팀장용 LLM 처리 및 JSON 파싱 유틸리티
Team manager LLM processing and JSON parsing utilities
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

def safe_json_loads_manager(raw_text: str) -> dict:
    """팀장용 레포트 안전한 JSON 로드"""
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
    
    # 3차: 팀장용 레포트 재구성
    print("⚠️ 팀장용 JSON 파싱 실패, 데이터 재구성 중...")
    return reconstruct_manager_json_from_text(raw_text)

def reconstruct_manager_json_from_text(raw_text: str) -> dict:
    """팀장용 레포트 텍스트에서 JSON 데이터 재구성"""
    try:
        # 팀 기본 정보 추출
        team_name = extract_value(r'"팀명":\s*"([^"]+)"', raw_text, "알 수 없는 팀")
        team_leader = extract_value(r'"팀장명":\s*"([^"]+)"', raw_text, "알 수 없음")
        period = extract_value(r'"업무_수행_기간":\s*"([^"]+)"', raw_text, "")
        
        # 팀 성과 데이터 추출
        team_achievement = extract_number(r'"팀_종합_달성률":\s*(\d+)', raw_text, 0)
        team_efficiency = extract_number(r'"팀_효율성":\s*(\d+)', raw_text, 0)
        
        # 팀원별 성과 추출 (간단화)
        member_performances = []
        member_patterns = list(re.finditer(r'"성명":\s*"([^"]+)"', raw_text))
        
        for i, member_match in enumerate(member_patterns[:5]):  # 최대 5명까지
            member_name = member_match.group(1)
            
            # 해당 팀원의 섹션 범위 결정
            start_pos = member_match.start()
            if i + 1 < len(member_patterns):
                end_pos = member_patterns[i + 1].start()
            else:
                end_pos = raw_text.find('"팀_운영_분석"', start_pos)
                if end_pos == -1:
                    end_pos = len(raw_text)
            
            member_section = raw_text[start_pos:end_pos]
            
            position = extract_value(r'"직위":\s*"([^"]*)"', member_section, "")
            achievement_rate = extract_number(r'"개인_달성률":\s*(\d+)', member_section, 0)
            contribution = extract_number(r'"팀_기여도":\s*(\d+)', member_section, 0)
            performance_comment = extract_value(r'"성과_분석":\s*"([^"]*(?:\\.[^"]*)*)"', member_section, f"{member_name}님의 성과")
            
            # 긴 텍스트 자르기
            if len(performance_comment) > 800:
                performance_comment = performance_comment[:800] + "..."
            
            member_performances.append({
                "성명": member_name,
                "직위": position,
                "개인_달성률": achievement_rate,
                "팀_기여도": contribution,
                "성과_분석": performance_comment
            })
        
        # 팀 운영 분석 추출
        team_strength = extract_value(r'"팀_강점":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "팀워크 우수")
        team_weakness = extract_value(r'"개선_필요사항":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "지속적인 개선 필요")
        risk_factors = extract_value(r'"리스크_요인":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "특별한 리스크 없음")
        
        # 향후 전략 추출
        short_term = extract_value(r'"단기_전략":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "현재 방향 유지")
        mid_term = extract_value(r'"중기_전략":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "역량 강화")
        long_term = extract_value(r'"장기_전략":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "지속적 성장")
        
        # 총평 추출
        total_comment = extract_value(r'"총평":\s*"([^"]*(?:\\.[^"]*)*)"', raw_text, "")
        if len(total_comment) > 1000:
            total_comment = total_comment[:1000] + "..."
        if not total_comment:
            total_comment = f"{team_name}의 이번 기간 성과는 전반적으로 양호합니다."
        
        # 팀장용 JSON 구조 생성
        reconstructed_json = {
            "팀_기본정보": {
                "팀명": team_name,
                "팀장명": team_leader,
                "업무_수행_기간": period
            },
            "팀_전체_성과": {
                "팀_종합_달성률": team_achievement,
                "팀_효율성": team_efficiency,
                "성과_해석": f"{team_name}의 종합적인 성과 분석 결과입니다.",
                "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
            },
            "팀원별_성과_분석": {
                "팀원_성과표": member_performances,
                "팀원_관리_총평": f"{team_name} 팀원들의 개별 성과 분석입니다."
            },
            "팀_운영_분석": {
                "팀_강점": team_strength,
                "개선_필요사항": team_weakness,
                "리스크_요인": risk_factors
            },
            "향후_운영_전략": {
                "단기_전략": short_term,
                "중기_전략": mid_term,
                "장기_전략": long_term
            },
            "총평": total_comment
        }
        
        print(f"✅ 팀장용 JSON 재구성 완료: {team_name}")
        return reconstructed_json
        
    except Exception as e:
        print(f"❌ 팀장용 JSON 재구성 실패: {e}")
        # 최소한의 안전한 데이터 반환
        return {
            "팀_기본정보": {
                "팀명": "데이터 오류",
                "팀장명": "데이터 오류",
                "업무_수행_기간": ""
            },
            "팀_전체_성과": {
                "팀_종합_달성률": 0,
                "팀_효율성": 0,
                "성과_해석": "데이터 처리 중 오류 발생",
                "해석_기준": "달성률 90% 이상: 우수, 80-89%: 양호, 70-79%: 보통, 70% 미만: 개선 필요"
            },
            "팀원별_성과_분석": {
                "팀원_성과표": [],
                "팀원_관리_총평": "데이터 오류"
            },
            "팀_운영_분석": {
                "팀_강점": "데이터 오류",
                "개선_필요사항": "데이터 오류",
                "리스크_요인": "데이터 오류"
            },
            "향후_운영_전략": {
                "단기_전략": "데이터 오류",
                "중기_전략": "데이터 오류",
                "장기_전략": "데이터 오류"
            },
            "총평": "데이터 처리 중 오류가 발생했습니다."
        }

class ManagerLLMClient:
    """팀장용 LLM 클라이언트"""
    
    def __init__(self, llm_client: ChatOpenAI):
        self.client = llm_client
    
    def get_manager_tone_correction_prompt(self, report_type) -> str:
        """팀장용 레포트 톤 보정 프롬프트 반환"""
        base_prompt = """
# 팀장용 레포트 톤 보정 프롬프트

## 대상 레포트
* **팀장용 분기별 레포트**: 팀원들의(팀의) 분기 성과 평가
* **팀장용 연말 레포트**: 팀원들의(팀의) 연말 성과 평가

## 레포트 식별 기준
* 팀 기본정보 포함 (팀명, 팀장명)
* 팀 전체 성과 분석 및 팀원별 순위 정보
* 팀 운영, 리스크 관리, 향후 전략 등 관리 관련 섹션

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
     * "팀원들 간 협업으로"
     * "팀 차원의 협력을 통해"
     * "조직적 협업으로"

### 4. 부정적 피드백 전달 방식 개선 (팀장용 특화)
* **원칙**: 관리적 관점에서 객관적이고 전략적인 표현 사용
* **수정 예시**:
  * "실수가 잦고 기술 이해력이 부족" → "업무 정확성 향상을 위한 관리적 지원이 필요한 상황입니다"
  * "무관심한 태도" → "업무 몰입도 제고를 위한 동기부여 방안이 필요합니다"
  * "산만한 모습" → "집중력 향상을 위한 환경 조성이 필요합니다"

### 5. 시간 표현 구체화
* **원칙**: 기본_정보의 업무_수행_기간과 일치시키기
* **수정 예시**:
  * "이번 분기 동안" → JSON의 업무_수행_기간 값 사용
  * "2024년 1분기 동안의 성과" → "2024년 1분기 성과"

## 글자수 제한 규칙

### 팀원별 성과 분석 글자수 제한
* **대상**: `팀원_성과_분석` > `팀원별_기여도` > `기여_내용` 필드
* **제한**: 각 팀원의 기여_내용은 **200자 이내**로 작성
* **적용 방법**:
 - 핵심 성과와 주요 Task 중심으로 요약
 - 구체적인 수치(달성률, 기여도 점수)는 유지
 - 반복적인 설명이나 부연 설명 최소화
 - 향후 개선 방향은 간단히 1-2문장으로 정리

### 글자수 단축 가이드라인
* **유지 우선순위**: 
 1. 구체적 수치 (달성률, 기여도 점수)
 2. 주요 Task명과 핵심 성과
 3. 팀 내 순위 및 상대적 위치
* **단축 대상**:
 - "이러한 분석을 통해", "향후 성과 향상을 위해서는" 등 연결어구
 - 중복되는 수식어나 부연 설명
 - 과도한 상세 분석 내용

## 팀장용 레포트 특화 톤

### 핵심 톤 특성
* **관리적·전략적**: 팀 전체 성과 관리 관점
* **의사결정 지원**: 팀장의 의사결정에 도움이 되는 정보 제공
* **객관적·분석적**: 데이터 기반의 객관적 분석
* **리스크 중심**: 문제점과 개선 방안에 초점

### 특화 표현 가이드라인

**1. 팀 성과 분석 표현**
* "팀의 전반적인 성과 분석 결과"
* "조직 운영 관점에서 볼 때"
* "팀 목표 달성을 위한 전략적 관점에서"
* "효율적인 팀 운영을 위해"

**2. 팀원 관리 표현**
* "[이름]님에 대한 관리적 지원 방안"
* "팀원 역량 강화를 위한 전략"
* "인력 운용 최적화 관점에서"
* "팀 리더십 발휘를 위해"

**3. 리스크 관리 표현**
* "팀 운영상 주의가 필요한 사항"
* "조직 효율성 저해 요소"
* "사전 관리가 필요한 리스크"
* "팀 성과 최적화를 위한 개선점"

**4. 전략 제안 표현**
* "향후 팀 운영 전략으로"
* "조직 성과 향상을 위한 방안"
* "팀 역량 강화 전략"
* "효과적인 인력 관리 방안"

**5. 객관적 분석 표현**
* "데이터 분석 결과에 따르면"
* "성과 지표 기준으로 평가할 때"
* "정량적 분석을 통해"
* "객관적 관점에서 판단하면"

## 유지해야 할 요소

### 1. 수식어는 그대로 유지
* "진심으로", "매우", "크게", "성공적으로" 등의 수식어는 제거하지 않음
* 감정적 표현과 강조 표현은 보존

### 2. 구체적 수치와 데이터
* 달성률, 기여도, 점수 등의 수치 정보는 정확히 유지
* 분석 내용의 실질적 정보는 변경하지 않음

### 3. 평가 기준과 구조
* JSON의 원본 구조와 항목 구성 유지
* 평가 기준과 해석 방식은 그대로 보존

## 팀장용 레포트 특별 고려사항
* 팀장의 의사결정과 팀 운영에 실질적 도움이 되는 정보 제공
* 개인 감정보다는 조직적 관점에서 객관적 서술
* 문제 상황에 대한 구체적이고 실행 가능한 해결책 제시
* 팀 전체의 성과 최적화를 위한 전략적 관점 유지

## 지시사항
아래 JSON 형태의 팀장용 업무 평가 레포트의 톤을 위 규칙에 따라 보정해주세요.
원본 JSON 구조와 데이터는 유지하되, 텍스트 내용만 보정하여 완전한 JSON 형태로 응답해주세요.
특히 팀원별_기여도의 기여_내용은 반드시 200자 이내로 작성해주세요.
"""
        
        if str(report_type) == "ManagerReportType.MANAGER_QUARTERLY":
            base_prompt += "\n이 레포트는 팀장용 분기별 레포트입니다."
        else:
            base_prompt += "\n이 레포트는 팀장용 연말 레포트입니다."
        
        return base_prompt
    
    def correct_tone(self, report_json: Dict[str, Any], report_type) -> str:
        """LLM을 사용한 팀장용 톤 보정"""
        system_prompt = self.get_manager_tone_correction_prompt(report_type)
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