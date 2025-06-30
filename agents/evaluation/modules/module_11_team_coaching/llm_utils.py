# llm_utils.py
#  LLM 처리 및 JSON 파싱 유틸리티
import logging
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
import json
import re


# 로깅 설정
logger = logging.getLogger(__name__)


# ====================================
# LLM 클라이언트 초기화
# ====================================

def init_llm_client():
    """OpenAI 클라이언트 초기화"""
    # LLM 클라이언트 설정 (환경변수는 자동으로 로드됨)
    llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    logger.info("LLM 클라이언트 초기화 완료")
    
    return llm_client

# ====================================
# JSON 파싱 유틸리티
# ====================================

def parse_json_field(json_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """JSON 필드 안전 파싱"""
    if not json_str:
        return None
    
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"JSON 파싱 실패: {str(e)[:100]}...")
        return None

def extract_json_from_llm_response(text: str) -> str:
    """개선된 LLM 응답에서 JSON 블록 추출"""
    
    # 1. 마크다운 코드 블록에서 JSON 추출 (가장 일반적)
    json_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        json_content = match.group(1).strip()
        if is_valid_json_string(json_content):
            return json_content
    
    # 2. 중괄호로 둘러싸인 JSON 객체 찾기 (중첩 고려)
    json_objects = _extract_nested_json_objects(text)
    for json_obj in json_objects:
        if is_valid_json_string(json_obj):
            return json_obj
    
    # 3. 전체 텍스트가 JSON인지 확인
    stripped_text = text.strip()
    if stripped_text.startswith('{') and stripped_text.endswith('}'):
        if is_valid_json_string(stripped_text):
            return stripped_text
    
    # 4. 최후의 수단: 첫 번째 { 부터 마지막 } 까지
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        potential_json = text[first_brace:last_brace + 1]
        if is_valid_json_string(potential_json):
            return potential_json
    
    logger.warning(f"JSON 추출 실패, 원본 텍스트 반환: {text[:200]}...")
    return text.strip()

def _extract_nested_json_objects(text: str) -> List[str]:
    """중첩된 JSON 객체들을 추출"""
    json_objects = []
    brace_count = 0
    start_pos = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                json_candidate = text[start_pos:i + 1]
                json_objects.append(json_candidate)
                start_pos = -1
    
    # 길이순으로 정렬 (더 완전한 JSON이 뒤에 오도록)
    return sorted(json_objects, key=len, reverse=True)

def is_valid_json_string(text: str) -> bool:
    """JSON 문자열 유효성 검사"""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

# ====================================
# JSON 구조 검증
# ====================================

def validate_risk_json_structure(result: Dict) -> bool:
    """리스크 JSON 구조 검증"""
    try:
        risk_analysis = result.get('risk_analysis', {})
        major_risks = risk_analysis.get('major_risks', [])
        
        if not isinstance(major_risks, list) or len(major_risks) == 0:
            return False
        
        # 첫 번째 리스크의 필수 필드 검증
        first_risk = major_risks[0]
        required_fields = ['risk_name', 'severity', 'description', 'causes', 'impacts', 'strategies']
        
        for field in required_fields:
            if field not in first_risk:
                return False
        
        return True
        
    except Exception:
        return False

def validate_plan_json_structure(result: Dict) -> bool:
    """연말 계획 JSON 구조 검증"""
    try:
        annual_plans = result.get('annual_plans', [])
        if not isinstance(annual_plans, list) or len(annual_plans) == 0:
            return False
        
        plan = annual_plans[0]
        if 'personnel_strategies' not in plan or 'collaboration_improvements' not in plan:
            return False
        
        return True
        
    except Exception:
        return False

# ====================================
# 폴백 데이터 생성 함수들
# ====================================

def get_year_from_period(period_id: int) -> int:
    """period_id로 연도 조회"""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import DatabaseConfig
        
        db_config = DatabaseConfig()
        DATABASE_URL = db_config.DATABASE_URL
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        with engine.connect() as connection:
            query = text("SELECT year FROM periods WHERE period_id = :period_id")
            result = connection.execute(query, {"period_id": period_id}).scalar_one_or_none()
            return result if result else 2025  # fallback
    except Exception as e:
        logger.warning(f"연도 조회 실패: {e}, fallback 사용")
        return 2025

def create_fallback_risk_analysis() -> Dict[str, Any]:
    """개선된 폴백 리스크 분석"""
    from datetime import datetime
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return {
        "risk_analysis": {
            "major_risks": [
                {
                    "risk_name": "시스템 분석 한계로 인한 리스크 식별 제약",
                    "severity": "medium",
                    "description": f"LLM 분석 프로세스에서 기술적 한계가 발생하여 정확한 리스크 식별이 제한됨 (발생시간: {current_time})",
                    "causes": [
                        "자동화된 분석 시스템의 기술적 한계",
                        "복잡한 조직 데이터 처리 과정의 예외 상황"
                    ],
                    "impacts": [
                        {
                            "impact_scope": "team",
                            "timeline": "immediate",
                            "description": "정확한 리스크 대응 계획 수립의 어려움 및 의사결정 지연 가능성"
                        }
                    ],
                    "strategies": [
                        {
                            "description": "팀장 주도의 수동 리스크 검토 프로세스 즉시 시행 및 주요 리스크 요소 직접 식별"
                        }
                    ]
                }
            ]
        }
    }

def create_fallback_annual_plan(period_info: Dict) -> Dict[str, Any]:
    """개선된 폴백 연말 계획"""
    
    # period_id가 있으면 동적으로 연도 계산, 없으면 period_info에서 가져오기
    if 'period_id' in period_info:
        current_year = get_year_from_period(period_info['period_id'])
    else:
        current_year = period_info.get('year', 2025)
    
    next_year = current_year + 1
    
    return {
        "annual_plans": [
            {
                "personnel_strategies": [
                    {
                        "target": "팀 전체 구성원",
                        "action": f"{next_year}년 상반기 중 팀장 주도의 개별 면담을 통한 맞춤형 성장 계획 수립 및 실행"
                    },
                    {
                        "target": "시스템 분석 프로세스",
                        "action": "자동화 분석 시스템의 한계 보완을 위한 수동 검토 체계 구축 및 정기적 모니터링 실시"
                    }
                ],
                "collaboration_improvements": [
                    {
                        "current_issue": "자동화된 협업 분석의 기술적 한계",
                        "improvement": "팀 내 정기적 협업 효과성 점검 미팅 및 피드백 수집 체계 구축",
                        "expected_benefit": "실제 협업 이슈의 신속한 발견 및 해결을 통한 팀 효율성 향상",
                        "target": "월 1회 협업 점검 미팅 실시 및 분기별 협업 만족도 80% 이상 달성"
                    }
                ]
            }
        ]
    }

def create_fallback_annual_comment_text() -> str:
    """개선된 폴백 연말 총평"""
    return """**[팀 성과 방향]**
시스템 분석의 기술적 한계로 인해 정확한 연간 성과 분석이 제한되었으나, 팀의 지속적인 성장과 발전을 위해서는 체계적인 성과 관리와 목표 달성 모니터링이 필요합니다. 자동화된 분석 시스템을 보완하는 수동 검토 프로세스를 통해 더욱 정확한 성과 평가 체계를 구축해야 합니다.

**[구조적 인식]**
현재 조직의 데이터 기반 의사결정 시스템이 완전하지 않음을 인식하고, 이를 보완할 수 있는 다층적 검토 체계가 필요합니다. 팀장의 직접적인 관찰과 판단을 통한 정성적 평가와 시스템 분석을 결합하여 보다 균형잡힌 조직 운영이 가능할 것입니다.

**[향후 운영 전략]**  
차년도에는 시스템 의존도를 줄이고 팀장 주도의 능동적 팀 관리 체계를 강화하는 것이 우선순위입니다. 정기적인 개별 면담, 팀 내 소통 활성화, 그리고 구체적이고 측정 가능한 목표 설정을 통해 팀 성과를 지속적으로 향상시켜 나가야 합니다."""

def create_fallback_quarterly_comment_text() -> str:
    """개선된 폴백 분기 총평"""
    return """**[전분기 대비 변화]**
시스템 분석 한계로 인해 정확한 전분기 대비 변화를 수치적으로 파악하기 어려우나, 지속적인 성과 관리를 위해서는 팀장의 직접적인 관찰과 평가가 필요합니다.

**[유사조직 대비 현황]**
비교 데이터의 한계로 유사조직과의 정확한 벤치마킹이 제한적이지만, 팀 고유의 강점을 활용한 차별화된 성과 창출에 집중해야 합니다.

**[종합 평가]**
다음 분기에는 시스템에 의존하지 않는 독립적인 성과 모니터링 체계를 구축하고, 팀원들과의 직접적인 소통을 통해 실질적인 성과 개선 방안을 도출해야 합니다."""