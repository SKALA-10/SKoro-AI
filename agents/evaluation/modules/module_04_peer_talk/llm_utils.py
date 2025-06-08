# agents/evaluation/modules/module_04_peer_talk/llm_utils.py

from db_utils import *
from llm_utils import *

import re
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv() 

# LangChain LLM 관련 임포트
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# --- LLM 클라이언트 인스턴스 (전역 설정) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
print(f"LLM Client initialized with model: {llm.model_name}, temperature: {llm.temperature}")


def _extract_work_keywords(work_content: str) -> List[str]:
    """업무 내용에서 핵심 업무/프로세스 키워드 추출 (전 분야 대응)"""
    work_keywords = []
    
    # 전 분야 업무 패턴
    work_patterns = [
        # IT/개발
        r'(AI|ML|NLP|TensorFlow|Spring Boot|React|API|REST|UI/UX|시스템|데이터베이스|아키텍처)',
        # 인사/HR
        r'(채용|면접|교육|평가|복리후생|노사관계|인력관리|조직문화|성과관리)',
        # 재무/회계
        r'(예산|결산|회계|세무|자금|투자|원가|손익|재무제표|현금흐름)',
        # 마케팅/홍보
        r'(브랜딩|광고|캠페인|시장조사|고객분석|SNS|콘텐츠|프로모션|브랜드)',
        # 영업/고객관리
        r'(영업|고객|계약|제안|협상|수주|매출|고객만족|CRM|B2B|B2C)',
        # 제조/생산
        r'(생산|제조|품질|공정|설비|안전|물류|재고|납기|효율성)',
        # 연구개발
        r'(연구|개발|실험|특허|혁신|기술|프로토타입|테스트|검증|분석)',
        # 법무/컴플라이언스
        r'(계약|법무|컴플라이언스|리스크|감사|규정|정책|승인|검토)',
        # 기획/전략
        r'(기획|전략|계획|목표|성과|KPI|로드맵|분석|보고|의사결정)',
        # 일반 업무
        r'(회의|보고서|문서|프로젝트|협업|커뮤니케이션|조율|관리|실행|점검)',
        # 숫자/성과 관련
        r'(\d+%|\d+건|\d+명|\d+시간|\d+일|\d+개월|\d+년|\d+억|\d+만)',
        # 업무 동사
        r'(완료|달성|수행|진행|개선|해결|구축|운영|관리|지원)'
    ]
    
    for pattern in work_patterns:
        matches = re.findall(pattern, work_content, re.IGNORECASE)
        work_keywords.extend([match for match in matches if isinstance(match, str)])
    
    return list(set(work_keywords))[:4]  # 상위 4개로 확장


def _process_work_content_for_context(work_content: str, work_keywords: List[str]) -> str:
    """업무 내용을 문맥 생성용으로 압축 (전 분야 대응)"""
    lines = work_content.split('.')
    key_lines = []
    
    # 업무 관련 핵심 단어들 (전 분야)
    key_terms = [
        # 일반 업무 동사
        '개발', '설계', '구현', '분석', '관리', '운영', '기획', '실행', '완료', '달성',
        # 커뮤니케이션/협업
        '인터뷰', '회의', '협업', '소통', '조율', '협상', '보고',
        # 성과/결과
        '개선', '해결', '성과', '목표', '효율', '품질', '만족',
        # 문서/정보
        '문서', '자료', '데이터', '정보', '계획', '전략'
    ]
    
    for line in lines:
        # 추출한 키워드가 포함된 라인 우선
        if any(keyword.lower() in line.lower() for keyword in work_keywords):
            key_lines.append(line.strip())
        # 일반 업무 용어가 포함된 라인
        elif any(term in line for term in key_terms):
            key_lines.append(line.strip())
    
    return ' '.join(key_lines[:2])  # 상위 2개 라인만


def _validate_and_enhance_sentence(sentence: str, keywords: str, work_keywords: List[str], weight: int) -> str:
    """생성된 문장의 품질 검증 및 보완"""
    # 길이 체크 및 조정
    if len(sentence) > 150:
        sentence = sentence[:147] + "..."
    
    # 업무 키워드가 포함되지 않은 경우 보완
    has_work_context = any(keyword in sentence for keyword in work_keywords)
    if not has_work_context and work_keywords:
        sentence = f"{work_keywords[0]} 업무에서 {sentence}"
    
    return sentence


def _create_enhanced_fallback_sentence(employee_id: str, keywords: str, work_content: str, weight: int) -> str:
    """고도화된 기본 문장 생성 (개인 이름 없이)"""
    work_keywords = _extract_work_keywords(work_content)
    main_work = work_keywords[0] if work_keywords else "프로젝트"
    
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    main_keyword = keyword_list[0] if keyword_list else "협업"
    
    if weight >= 3:
        return f"{main_work} 진행 과정에서 동료가 {main_keyword}한 모습을 지속적으로 보여주며 팀 성과에 기여함"
    else:
        return f"{main_work} 업무에서 해당 직원이 {main_keyword}한 특성을 나타냄"


def call_llm_for_context_generation(keywords: str, work_situation: str, weight: int) -> str:
    """LLM을 호출하여 맥락 문장 생성"""
    
    # 상세한 프롬프트 (요청하신 내용 포함)
    detailed_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 동료평가 분석 전문가입니다. 개인 이름을 절대 사용하지 말고 '동료' 또는 '해당 직원'이라는 표현만 사용하세요."),
        ("human", """다음 키워드들을 바탕으로 업무 상황에서의 평가 문장을 한 문장으로 작성하세요.

키워드: {keywords}
업무 상황: {work_situation}
평가 비중: {weight}

=== 문장 생성 가이드라인 ===
1. 업무 맥락을 구체적으로 포함 
   - IT: "시스템 구축 과정에서", "API 설계 회의 중에"
   - 영업: "고객 프레젠테이션 중에", "계약 협상 과정에서"
   - 인사: "채용 면접 진행 시", "교육 프로그램 기획 중에"
   - 마케팅: "캠페인 기획 단계에서", "시장 조사 진행 중에"
   - 재무: "예산 수립 과정에서", "재무제표 분석 중에"
2. 키워드를 실제 행동이나 결과와 연결
3. 비중이 높을수록(3-4) 더 구체적이고 상세한 상황 묘사
4. 비중이 낮을수록(1-2) 간결하지만 핵심적인 특징 언급
5. 관찰 가능한 구체적 행동, 결과, 영향에 초점

=== 작성 스타일 예시 (다양한 분야) ===
❌ 피해야 할 표현: "동료가 리더십을 발휘함"
✅ IT 분야: "시스템 요구사항 분석 단계에서 동료가 팀원들의 다양한 의견을 조율하며 배려심을 보였으나, 때때로 회피형 태도로 기술적 결정을 미루는 경향을 보임"
✅ 영업 분야: "고객 프레젠테이션 준비 과정에서 해당 직원이 꼼꼼한 자료 분석을 통해 책임감을 보였지만, 무의욕한 태도로 팀워크에 부정적 영향을 미침"
✅ 인사 분야: "신입사원 교육 프로그램 기획 시 동료가 창의적 아이디어를 제시하며 긍정마인드를 발휘했으나, 실행 단계에서 소극적인 모습을 보임"

중요: 절대로 개인 이름이나 사번을 사용하지 마세요. 반드시 '동료', '해당 직원', '팀원' 등의 표현만 사용하세요.

출력: 구체적인 업무 상황과 연결된 평가 문장만 반환 (150자 이내)
""")
    ])
    
    try:
        # 메시지 생성
        messages = detailed_prompt.format_messages(
            keywords=keywords,
            work_situation=work_situation,
            weight=weight
        )
        
        response = llm(messages)
        요약문장 = response.content.strip()
        
        # 혹시 모를 템플릿 변수나 개인 이름 제거
        요약문장 = 요약문장.replace("{evaluated_name}", "동료")
        요약문장 = 요약문장.replace("{evaluator_name}", "동료")
        요약문장 = 요약문장.replace("evaluated_name", "동료")
        요약문장 = 요약문장.replace("evaluator_name", "동료")
        
        return 요약문장
        
    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {str(e)}")
        return "업무 진행 과정에서 동료가 다양한 특성을 보임"


def call_llm_for_keyword_sentiment_analysis(keyword: str) -> str:
    """새로운 키워드의 감정 분석"""
    
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 동료평가 키워드 분석 전문가입니다.
        주어진 키워드가 직장에서의 동료평가 맥락에서 긍정적인지 부정적인지 판단해주세요.
        """),
        ("human", """
        다음 키워드가 직장 동료평가에서 긍정적인지 부정적인지 판단해주세요.

        키워드: {keyword}

        판단 기준:
        - 긍정적: 업무 능력, 협업 태도, 성과 등에서 좋은 평가를 나타내는 키워드
        - 부정적: 업무 능력, 협업 태도, 성과 등에서 개선이 필요한 평가를 나타내는 키워드  
        - 중립적: 긍정도 부정도 아닌 객관적 특성을 나타내는 키워드

        다음 중 하나로만 답해주세요: "긍정", "부정", "중립"
        """)
    ])
    
    try:
        messages = sentiment_prompt.format_messages(keyword=keyword)
        response = llm(messages)
        sentiment = response.content.strip().lower()
        
        if "긍정" in sentiment:
            return "긍정"
        elif "부정" in sentiment:
            return "부정"
        else:
            return "중립"
            
    except Exception as e:
        print(f"키워드 감정 분석 오류: {str(e)}")
        return "중립"


def call_llm_for_feedback_generation(employee_id: str, total_evaluations: int, positive_keywords: str, negative_keywords: str, context_sentences: List[str]) -> Dict[str, str]:
    """최종 피드백 생성 LLM 호출"""
    
    # 더 강화된 프롬프트 템플릿
    forced_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        당신은 전문 HR 컨설턴트입니다. 
        주어진 정보가 제한적이더라도 반드시 의미 있는 피드백을 생성해야 합니다.
        데이터가 부족하다는 이유로 피드백을 거부하지 마세요.
        '평가 대상자'라는 표현을 사용하지 마세요.
        "긍정 키워드로 인해", "부정 키워드가 언급된"와 같이 분석 용어를 사용하지 마세요.
        """),
        HumanMessage(content="""
        다음 동료평가 정보를 바탕으로 피드백을 생성해주세요.

        평가 건수: {total_evaluations}
        
        긍정 키워드: {positive_keywords}
        부정 키워드: {negative_keywords}
        
        평가 상황들:
        {context_sentences}

        **중요**: 정보가 제한적이더라도 반드시 다음 JSON 형식으로 출력하세요:
        {{
          "강점": "실제 업무 상황에서 관찰된 긍정적 행동과 결과 (50-80자)",
          "우려": "업무 맥락에서 나타난 개선점을 건설적으로 표현 (50-80자)",  
          "협업관찰": "팀 내에서 보여준 협업 스타일과 소통 방식 (50-80자)"
        }}

        작성 규칙:
        1. 반드시 JSON 형식으로 출력
        2. 각 항목은 완전한 문장으로 작성
        3. 상황 맥락을 포함한 자연스러운 서술
        4. "데이터 부족"이나 "추가 분석 필요" 같은 표현 금지
        5. 주어진 정보에서 최선의 해석으로 의미 있는 피드백 생성

        데이터가 적어도 창의적으로 해석하여 반드시 피드백을 완성하세요.
        """)
    ])
    
    try:
        # 메시지 생성
        messages = forced_prompt.format_messages(
            total_evaluations=total_evaluations,
            positive_keywords=positive_keywords if positive_keywords else "협업, 책임감",
            negative_keywords=negative_keywords if negative_keywords else "일부 개선 영역",
            context_sentences="\n".join([f"- {s}" for s in context_sentences]) if context_sentences else "- 기본적인 업무 협력 상황"
        )
        
        response = llm(messages)
        
        # JSON 파싱 시도
        content = response.content.strip()
        
        # JSON 추출 시도
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            feedback = json.loads(json_str)
            
            # 필수 키 확인
            required_keys = ["강점", "우려", "협업관찰"]
            if all(key in feedback for key in required_keys):
                return feedback
            else:
                raise ValueError("필수 키 누락")
        else:
            raise ValueError("JSON 형식 불일치")
            
    except Exception as e:
        print(f"LLM 피드백 생성 오류: {str(e)}")
        # 기본값 반환
        return {
            "강점": "동료 평가에서 나타난 긍정적 특성을 바탕으로 팀에 기여하고 있습니다",
            "우려": "지속적인 성장을 위해 일부 영역에서 추가적인 개발과 개선이 필요한 상황으로 보입니다",
            "협업관찰": "팀 내에서의 역할 수행과 동료들과의 관계에서 전반적으로 안정적인 모습을 보여주고 있습니다"
        }