from agents.evaluation.modules.module_03_peer_talk.db_utils import *

import re
import json 
from typing import List, Dict
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv() 

# LangChain LLM 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# --- LLM 클라이언트 인스턴스 (전역 설정) ---
llm_client = ChatOpenAI(model="gpt-4o", temperature=0) 
print(f"LLM Client initialized with model: {llm_client.model_name}, temperature: {llm_client.temperature}")

# --- LLM 응답에서 JSON 코드 블록 추출 도우미 함수 ---
def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답 텍스트에서 ```json ... ``` 블록만 추출합니다."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip() # JSON 내용만 반환하고 양쪽 공백 제거
    return text.strip()


# --- 키워드 감정 분석을 위한 사전 정의 ---
def get_predefined_keywords():
    """미리 정의된 긍정/부정 키워드 리스트 반환"""
    positive_keywords = {
        '리더십', '책임감', '창의적', '적극적', '협력적', '소통능력', '문제해결', 
        '성실함', '전문성', '혁신적', '팀워크', '긍정마인드', '배려심', '열정적',
        '신뢰성', '효율성', '분석력', '추진력', '유연성', '성장지향', '도전정신',
        '세심함', '논리적', '객관적', '체계적', '꼼꼼함', '친화력', '겸손함',
        '배려', '책임감 있는', '밝은', '성실한', '문제해결력', '긍정적', '열정',
        '추진력 있는', '주도적인', '목표지향적', '신뢰할 수 있는', '능동적인'
    }
    
    negative_keywords = {
        '소극적', '무관심', '비협조적', '고집스러운', '경직된', '회피형', '무의욕',
        '산만함', '지각', '불성실', '감정적', '주관적', '독단적', '폐쇄적',
        '수동적', '미루기', '책임회피', '소통부족', '부정적', '완고함', '예민함',
        '소극적인', '실수가 잦은', '무의욕자', '개인주의', '소통단절', '부정적인',
        '수동적인', '책임감 결여'
    }
    
    return positive_keywords, negative_keywords


def get_keyword_score(keyword: str, positive_keywords: set, negative_keywords: set) -> float:
    """키워드의 감정 점수 반환"""
    if keyword in positive_keywords:
        return 1.0
    elif keyword in negative_keywords:
        return -1.0
    else:
        return 0.0


# --- 모듈 3 전용 LLM 호출 함수들 ---

def call_llm_for_peer_evaluation_context(keywords: str, work_situation: str, weight: float) -> Dict:
    """동료평가 맥락 문장 생성"""
    print(f"LLM Call (Peer Evaluation Context): keywords='{keywords[:30]}...', weight={weight}")

    system_prompt = """
    당신은 동료평가 분석 전문가입니다. 
    개인 이름을 절대 사용하지 말고 '동료' 또는 '해당 직원'이라는 표현만 사용하세요.
    주어진 키워드들과 업무 상황을 바탕으로 구체적이고 자연스러운 평가 문장을 생성하세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """
    
    human_prompt = f"""
    다음 키워드들을 바탕으로 업무 상황에서의 평가 문장을 한 문장으로 작성하세요.

    키워드: {keywords}
    업무 상황: {work_situation[:100] + "..." if len(work_situation) > 100 else work_situation}
    평가 비중: {weight}

    중요: 개인 이름, 사번 절대 사용 금지. '동료', '해당 직원' 등 일반적 표현만 사용.

    JSON 응답:
    {{
        "context_sentence": "[구체적인 업무 상황과 연결된 평가 문장 (150자 이내)]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({"keywords": keywords, "work_situation": work_situation, "weight": weight})
        json_output_raw = response.content

        json_output = _extract_json_from_llm_response(str(json_output_raw))

        llm_parsed_data = json.loads(json_output)

        context_sentence = llm_parsed_data.get("context_sentence", "업무 진행 과정에서 동료가 다양한 특성을 보임")
        
        # 개인정보 제거
        context_sentence = context_sentence.replace("{evaluated_name}", "동료")
        context_sentence = context_sentence.replace("{evaluator_name}", "동료")

        if not isinstance(context_sentence, str) or not context_sentence:
            raise ValueError(f"LLM 반환 문장 {context_sentence}가 유효하지 않습니다.")

        return {"context_sentence": context_sentence}
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"context_sentence": "업무 진행 과정에서 동료가 다양한 특성을 보임"}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"context_sentence": "업무 진행 과정에서 동료가 다양한 특성을 보임"}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"context_sentence": "업무 진행 과정에서 동료가 다양한 특성을 보임"}


def call_llm_for_keyword_weighted_analysis(keyword_collections: List[str], evaluation_weights: List[float]) -> Dict:
    """키워드 가중치 분석 (기존 로직을 LLM 스타일로 래핑)"""
    print(f"LLM Call (Keyword Weighted Analysis): {len(keyword_collections)}개 키워드 컬렉션 분석")
    
    try:
        positive_keywords, negative_keywords = get_predefined_keywords()
        
        weighted_scores = defaultdict(float)
        keyword_frequency = defaultdict(int)
        total_weight = sum(evaluation_weights) if evaluation_weights else 0
        
        for i in range(len(keyword_collections)):
            keywords = [k.strip() for k in keyword_collections[i].split(',') if k.strip()]
            weight = evaluation_weights[i] if i < len(evaluation_weights) else 1.0
            
            for keyword in keywords:
                keyword_frequency[keyword] += 1
                score = get_keyword_score(keyword, positive_keywords, negative_keywords)
                weighted_scores[keyword] += score * weight
        
        if total_weight > 0:
            for keyword in weighted_scores:
                weighted_scores[keyword] = weighted_scores[keyword] / total_weight
        
        positive_keywords_result = {k: v for k, v in weighted_scores.items() if v > 0}
        negative_keywords_result = {k: v for k, v in weighted_scores.items() if v < 0}
        
        top_positive = dict(sorted(positive_keywords_result.items(), key=lambda x: x[1], reverse=True)[:5])
        top_negative = dict(sorted(negative_keywords_result.items(), key=lambda x: x[1])[:3])
        
        analysis_result = {
            "weighted_scores": dict(weighted_scores),
            "keyword_frequency": dict(keyword_frequency),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "total_evaluations": len(keyword_collections),
            "average_weight": total_weight / len(evaluation_weights) if evaluation_weights else 0,
            "total_weight": total_weight
        }
        
        return analysis_result
        
    except Exception as e:
        print(f"키워드 가중치 분석 실패: {str(e)}")
        return {
            "weighted_scores": {},
            "keyword_frequency": {},
            "top_positive": {},
            "top_negative": {},
            "total_evaluations": 0,
            "average_weight": 0,
            "total_weight": 0
        }


def call_llm_for_final_feedback_generation(weighted_analysis: Dict, top_sentences: List[str], target_emp_no: str) -> Dict:
    """최종 피드백 생성 (강점, 우려, 협업관찰)"""
    print(f"LLM Call (Final Feedback Generation): {target_emp_no} 피드백 생성")

    top_positive = weighted_analysis.get("top_positive", {})
    top_negative = weighted_analysis.get("top_negative", {})
    total_evals = weighted_analysis.get("total_evaluations", 0)
    avg_weight = weighted_analysis.get("average_weight", 0)
    
    positive_text = ", ".join([f"{k}({v:.2f})" for k, v in list(top_positive.items())[:3]])
    negative_text = ", ".join([f"{k}({v:.2f})" for k, v in list(top_negative.items())[:2]])
    sentences_text = "\n".join([f"- {s}" for s in top_sentences[:3]])
    
    system_prompt = """
    당신은 동료평가 전문 분석가입니다. 
    개인 이름이나 사번을 절대 사용하지 마세요.
    주어진 분석 결과를 바탕으로 구조화된 피드백을 생성하세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """

    human_prompt = f"""
    다음 동료평가 분석 결과를 바탕으로 구조화된 피드백을 생성하세요.

    === 분석 데이터 ===
    총 평가 수: {total_evals}개
    평균 비중: {avg_weight:.1f}

    주요 긍정 키워드: {positive_text}
    주요 부정 키워드: {negative_text}

    핵심 평가 문장들:
    {sentences_text}

    중요: 개인 이름, 사번 절대 사용 금지. '동료', '해당 직원' 등 일반적 표현만 사용.

    JSON 응답:
    {{
        "strengths": "[구체적이고 균형잡힌 강점 서술]",
        "concerns": "[건설적이고 개선지향적인 우려사항]",
        "collaboration_observations": "[객관적이고 행동중심의 협업 관찰]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({
            "weighted_analysis": weighted_analysis, 
            "top_sentences": top_sentences, 
            "target_emp_no": target_emp_no
        })
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(str(json_output_raw))
        
        llm_parsed_data = json.loads(json_output)
        
        strengths = llm_parsed_data.get("strengths", "동료들로부터 긍정적인 평가를 받고 있습니다")
        concerns = llm_parsed_data.get("concerns", "지속적인 성장을 위한 개선 영역이 있습니다")
        collaboration = llm_parsed_data.get("collaboration_observations", "팀 내에서 협업에 참여하고 있습니다")

        if not isinstance(strengths, str) or not strengths:
            raise ValueError(f"LLM 반환 강점 {strengths}가 유효하지 않습니다.")
        if not isinstance(concerns, str) or not concerns:
            raise ValueError(f"LLM 반환 우려 {concerns}가 유효하지 않습니다.")
        if not isinstance(collaboration, str) or not collaboration:
            raise ValueError(f"LLM 반환 협업관찰 {collaboration}가 유효하지 않습니다.")

        return {
            "strengths": strengths,
            "concerns": concerns,
            "collaboration_observations": collaboration
        }

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return _generate_fallback_feedback(top_positive, top_negative, total_evals)
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return _generate_fallback_feedback(top_positive, top_negative, total_evals)
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return _generate_fallback_feedback(top_positive, top_negative, total_evals)


def _generate_fallback_feedback(top_positive: Dict, top_negative: Dict, total_evals: int) -> Dict:
    """LLM 호출 실패 시 대안 피드백 생성"""
    
    if top_positive:
        positive_keywords = list(top_positive.keys())[:2]
        strengths = f"동료 평가에서 {', '.join(positive_keywords)} 등의 긍정적 특성이 높게 평가되고 있습니다"
    else:
        strengths = "동료들과의 협업에서 다양한 긍정적 측면을 보여주고 있습니다"
    
    if top_negative:
        negative_keywords = list(top_negative.keys())[:1] 
        concerns = f"{negative_keywords[0]} 등의 영역에서 추가적인 개선과 발전이 필요할 것으로 보입니다"
    else:
        concerns = "전반적으로 안정적이나 지속적인 성장을 위한 노력이 필요합니다"
    
    collaboration = f"총 {total_evals}명의 동료로부터 평가받았으며 팀 내에서의 역할을 수행하고 있습니다"
    
    return {
        "strengths": strengths,
        "concerns": concerns,
        "collaboration_observations": collaboration
    }