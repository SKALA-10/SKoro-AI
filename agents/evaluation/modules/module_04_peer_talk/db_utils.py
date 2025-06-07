"""
agent.py - 동료평가 분석 에이전트들
"""

import json
import re
from typing import List, Dict
from collections import defaultdict

from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from llm_utils import llm
from db_utils import (
    fetch_peer_evaluations_for_target,
    fetch_keywords_for_peer_evaluations,
    fetch_tasks_for_peer_evaluations_fixed,
    fetch_task_summaries_fixed,
    get_team_evaluation_id,
    row_to_dict
)


def complete_data_mapping_agent(state: Dict, engine) -> Dict:
    """
    DB에서 동료 평가 데이터를 조회하여 PeerTalkState로 매핑하는 에이전트
    """
    try:
        # 입력 검증
        period_id = int(state["분기"])
        target_emp_no = state["평가받는사번"]
        
        if not target_emp_no:
            raise ValueError("평가받는사번이 필요합니다.")

        print(f"[CompleteDataMappingAgent] {target_emp_no}: 완전한 데이터 매핑 시작 (분기: {period_id})")

        # 1. 동료 평가 리스트 조회
        peer_evals = fetch_peer_evaluations_for_target(engine, period_id, target_emp_no)
        
        if not peer_evals:
            print(f"[CompleteDataMappingAgent] {target_emp_no}: 평가 데이터 없음")
            # 빈 데이터일 경우 기본값 설정
            for field in ["평가하는사번_리스트", "비중", "키워드모음", "구체적업무내용", "성과지표ID_리스트"]:
                state[field] = []
            for field in ["동료평가요약줄글들", "강점", "우려", "협업관찰"]:
                state[field] = []
            state["_weighted_analysis"] = {}
            return state

        peer_eval_ids = [pe["peer_evaluation_id"] for pe in peer_evals]
        print(f"[CompleteDataMappingAgent] {target_emp_no}: {len(peer_evals)}개 평가 발견")

        # 2. 기본 평가 정보 매핑
        state["평가하는사번_리스트"] = [pe["evaluator_emp_no"] for pe in peer_evals]
        state["비중"] = [pe["weight"] for pe in peer_evals]

        # 3. 키워드 모음 조회 및 매핑
        keyword_map = fetch_keywords_for_peer_evaluations(engine, peer_eval_ids)
        state["키워드모음"] = [
            ", ".join(keyword_map.get(pid, [])) if keyword_map.get(pid) else ""
            for pid in peer_eval_ids
        ]

        # 4. 업무 내용 조회 및 매핑 (수정된 함수 사용)
        task_map = fetch_tasks_for_peer_evaluations_fixed(engine, peer_eval_ids)
        all_task_ids = [tid for tids in task_map.values() for tid in tids]
        summary_map = fetch_task_summaries_fixed(engine, period_id, all_task_ids) if all_task_ids else {}
        
        # 각 평가별 첫 번째 task의 summary 사용
        state["구체적업무내용"] = []
        for pid in peer_eval_ids:
            if pid in task_map and task_map[pid]:
                first_task_id = task_map[pid][0]
                summary = summary_map.get(first_task_id, "")
                state["구체적업무내용"].append(summary)
            else:
                state["구체적업무내용"].append("")

        # 5. 성과지표ID_리스트 초기화
        state["성과지표ID_리스트"] = ["1"] * len(peer_evals)  # 기본값

        # 6. 기타 필드들 초기화
        state["동료평가요약줄글들"] = []
        state["강점"] = []
        state["우려"] = []
        state["협업관찰"] = []
        state["_weighted_analysis"] = {}
        
        print(f"[CompleteDataMappingAgent] {target_emp_no}: 완전한 매핑 완료")
        print(f"  - 평가자: {len(state['평가하는사번_리스트'])}명")
        print(f"  - 키워드: {len([k for k in state['키워드모음'] if k])}개 평가")
        print(f"  - 업무내용: {len([c for c in state['구체적업무내용'] if c])}개")
        print(f"  - 비중: {state['비중']}")
        
    except Exception as e:
        print(f"[CompleteDataMappingAgent] {target_emp_no}: 매핑 실패 - {str(e)}")
        import traceback
        traceback.print_exc()
        # 오류 발생시 기본값으로 초기화
        for field in ["평가하는사번_리스트", "비중", "키워드모음", "구체적업무내용", "성과지표ID_리스트"]:
            state[field] = []
        for field in ["동료평가요약줄글들", "강점", "우려", "협업관찰"]:
            state[field] = []
        state["_weighted_analysis"] = {}
    
    return state


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


def simple_context_generation_agent(state: Dict) -> Dict:
    """완전히 새로운 간단한 맥락 문장 생성 에이전트"""
    
    employee_id = state["평가받는사번"]
    print(f"[SimpleContextAgent] {employee_id}: 간단한 문장 생성 시작")
    
    # 평가 데이터가 없으면 스킵
    if not state["평가하는사번_리스트"]:
        print(f"[SimpleContextAgent] {employee_id}: 평가 데이터 없음")
        state["동료평가요약줄글들"] = []
        return state
    
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
❌ 피해야 할 표현: "리더십을 발휘함"
✅ IT 분야: "시스템 요구사항 분석 단계에서 동료가 팀원들의 다양한 의견을 조율하며 배려심을 보였으나, 때때로 회피형 태도로 기술적 결정을 미루는 경향을 보임"
✅ 영업 분야: "고객 프레젠테이션 준비 과정에서 해당 직원이 꼼꼼한 자료 분석을 통해 책임감을 보였지만, 무의욕한 태도로 팀워크에 부정적 영향을 미침"
✅ 인사 분야: "신입사원 교육 프로그램 기획 시 동료가 창의적 아이디어를 제시하며 긍정마인드를 발휘했으나, 실행 단계에서 소극적인 모습을 보임"

중요: 절대로 개인 이름이나 사번을 사용하지 마세요. 반드시 '동료', '해당 직원', '팀원' 등의 표현만 사용하세요.

출력: 구체적인 업무 상황과 연결된 평가 문장만 반환 (150자 이내)
""")
    ])
    
    요약문장들 = []
    
    # 각 평가별로 문장 생성
    for i in range(len(state["평가하는사번_리스트"])):
        키워드 = state["키워드모음"][i]
        업무내용 = state["구체적업무내용"][i]
        비중 = state["비중"][i]
        피평가자 = state["평가받는사번"]
        
        try:
            # 업무내용에서 핵심 키워드 추출
            work_keywords = _extract_work_keywords(업무내용)
            
            # 업무 내용을 요약하되 핵심 기술/프로세스 유지
            work_content_processed = _process_work_content_for_context(업무내용, work_keywords)
            
            # 업무 상황 간단히 추출
            work_situation = 업무내용[:100] + "..." if len(업무내용) > 100 else 업무내용
            
            # 메시지 생성
            messages = detailed_prompt.format_messages(
                keywords=키워드,
                work_situation=work_situation,
                weight=비중
            )
            
            response = llm(messages)
            요약문장 = response.content.strip()
            
            # 혹시 모를 템플릿 변수나 개인 이름 제거
            요약문장 = 요약문장.replace("{evaluated_name}", "동료")
            요약문장 = 요약문장.replace("{evaluator_name}", "동료")
            요약문장 = 요약문장.replace("evaluated_name", "동료")
            요약문장 = 요약문장.replace("evaluator_name", "동료")
            
            # 문장 품질 검증 및 보정
            요약문장 = _validate_and_enhance_sentence(요약문장, 키워드, work_keywords, 비중)
            
            요약문장들.append(요약문장)
            print(f"[SimpleContextAgent] {i+1}번째 문장 생성 완료")
            
        except Exception as e:
            # 실패 시 고도화된 기본 템플릿 사용
            fallback = _create_enhanced_fallback_sentence(피평가자, 키워드, 업무내용, 비중)
            요약문장들.append(fallback)
            print(f"[SimpleContextAgent] {i+1}번째 생성 실패, 고도화된 기본 템플릿 사용: {str(e)}")
    
    state["동료평가요약줄글들"] = 요약문장들
    print(f"[SimpleContextAgent] {employee_id}: 총 {len(요약문장들)}개 문장 생성 완료")
    
    return state


def weighted_analysis_agent(state: Dict) -> Dict:
    """비중을 반영한 키워드 가중치 분석 (새로운 키워드 감정 분석 포함)"""
    
    employee_id = state["평가받는사번"]
    print(f"[WeightedAnalysisAgent] {employee_id}: 가중치 분석 시작")
    
    # 기존 정의된 키워드들
    positive_keywords = {
        "배려", "긍정마인드", "열정", "책임감 있는", "성실한", "꼼꼼한",
        "추진력 있는", "문제해결력", "주도적인", "목표지향적", "의사결정력",
        "신뢰할 수 있는", "능동적인", "조율력", "유쾌한", "밝은", "리더십 있는",
        "열린 소통", "유연한", "빠른 실행력", "친절한", "협업", "현실적인",
        "아이디어", "고객중심", "창의적인", "분석적인", "체계적인", "논리적인",
        "침착한", "신중한", "적극적인", "전문적인", "세심한", "효율적인"
    }
    
    negative_keywords = {
        "소극적인", "실수가 잦은", "기한 미준수", "감정적인", "불쾌한 언행",
        "방어적인", "회피형", "개인주의", "무관심", "소통단절", "무의욕자",
        "부정적인", "부정적", "갑질", "근무태만", "다혈질", "리더십 없는",
        "이기주의", "수동적인", "비판적인", "의욕없는", "거짓말", "고집",
        "산만한", "느린 일처리"
    }
    
    # 평가 데이터가 없으면 스킵
    if not state["키워드모음"]:
        print(f"[WeightedAnalysisAgent] {employee_id}: 키워드 데이터 없음, 스킵")
        state["_weighted_analysis"] = _create_empty_analysis()
        return state
    
    # 1. 모든 고유 키워드 수집
    all_keywords = set()
    for keyword_string in state["키워드모음"]:
        keywords = [k.strip() for k in keyword_string.split(',') if k.strip()]
        all_keywords.update(keywords)
    
    print(f"[WeightedAnalysisAgent] {employee_id}: 총 {len(all_keywords)}개 고유 키워드 발견")
    
    # 2. 새로운 키워드 감정 분석
    analyzed_keywords = {}
    new_keywords = [k for k in all_keywords 
                   if k not in positive_keywords and k not in negative_keywords]
    
    if new_keywords:
        print(f"[WeightedAnalysisAgent] {employee_id}: {len(new_keywords)}개 새로운 키워드 감정 분석 중...")
        
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
        
        for keyword in new_keywords:
            try:
                messages = sentiment_prompt.format_messages(keyword=keyword)
                response = llm(messages)
                sentiment = response.content.strip().lower()
                
                if "긍정" in sentiment:
                    score = 1.0
                elif "부정" in sentiment:
                    score = -1.0
                else:
                    score = 0.0
                
                analyzed_keywords[keyword] = score
                print(f"  └ '{keyword}' → {sentiment} (점수: {score})")
                
            except Exception as e:
                analyzed_keywords[keyword] = 0.0
                print(f"  └ '{keyword}' → 분석 실패, 중립 처리 ({str(e)})")
    
    # 3. 가중치 분석 수행
    weighted_scores = defaultdict(float)
    keyword_frequency = defaultdict(int)
    total_weight = sum(state["비중"])
    
    def get_keyword_score(keyword: str) -> float:
        """키워드 감정 점수 반환"""
        if keyword in positive_keywords:
            return 1.0
        elif keyword in negative_keywords:
            return -1.0
        elif keyword in analyzed_keywords:
            return analyzed_keywords[keyword]
        else:
            return 0.0
    
    # 각 평가별로 키워드 가중치 계산
    for i in range(len(state["키워드모음"])):
        keywords = [k.strip() for k in state["키워드모음"][i].split(',') if k.strip()]
        weight = state["비중"][i]
        
        for keyword in keywords:
            keyword_frequency[keyword] += 1
            score = get_keyword_score(keyword)
            weighted_scores[keyword] += score * weight
    
    # 정규화 (총 비중으로 나누기)
    if total_weight > 0:
        for keyword in weighted_scores:
            weighted_scores[keyword] = weighted_scores[keyword] / total_weight
    
    # 상위 키워드 추출
    positive_keywords_result = {k: v for k, v in weighted_scores.items() if v > 0}
    negative_keywords_result = {k: v for k, v in weighted_scores.items() if v < 0}
    
    # 정렬 (점수 높은 순)
    top_positive = dict(sorted(positive_keywords_result.items(), key=lambda x: x[1], reverse=True)[:5])
    top_negative = dict(sorted(negative_keywords_result.items(), key=lambda x: x[1])[:3])
    
    # 분석 결과 저장
    analysis_result = {
        "weighted_scores": dict(weighted_scores),
        "keyword_frequency": dict(keyword_frequency),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "total_evaluations": len(state["평가하는사번_리스트"]),
        "average_weight": total_weight / len(state["비중"]) if state["비중"] else 0,
        "total_weight": total_weight,
        "new_keywords_analyzed": len(new_keywords)
    }
    
    state["_weighted_analysis"] = analysis_result
    
    print(f"[WeightedAnalysisAgent] {employee_id}: 분석 완료")
    print(f"  - 총 키워드: {len(weighted_scores)}")
    print(f"  - 긍정 키워드: {len(top_positive)}")
    print(f"  - 부정 키워드: {len(top_negative)}")
    print(f"  - 새로 분석된 키워드: {len(new_keywords)}")
    print(f"  - 평균 비중: {analysis_result['average_weight']:.2f}")
    
    return state


def _create_empty_analysis():
    """빈 분석 결과 생성"""
    return {
        "weighted_scores": {},
        "keyword_frequency": {},
        "top_positive": {},
        "top_negative": {},
        "total_evaluations": 0,
        "average_weight": 0,
        "total_weight": 0,
        "new_keywords_analyzed": 0
    }


def improved_feedback_generation_agent(state: Dict) -> Dict:
    """최종 구조화된 피드백 생성 - LLM이 반드시 생성하도록 강화"""
    
    employee_id = state["평가받는사번"]
    print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 강화된 피드백 생성 시작")
    
    # 최소한의 데이터만 확인
    if not state["평가하는사번_리스트"]:
        print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 평가 데이터 없음, 강제 생성 시도")
        # 그래도 LLM으로 생성 시도
    
    # 가중치 분석 결과가 없어도 진행
    analysis = state.get("_weighted_analysis", {})
    
    # 최대 3번 재시도
    for attempt in range(3):
        try:
            print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 생성 시도 {attempt + 1}/3")
            
            # 상위 요약 문장들 선별
            top_sentences = _get_top_sentences(state)
            
            # 더 강화된 프롬프트 템플릿
            forced_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                당신은 전문 HR 컨설턴트입니다. 
                주어진 동료평가 상황들을 구체적으로 반영하여 피드백을 작성하세요.
                절대로 "평가 대상자", "평가받는 사람", "해당 직원", "대상자" 같은 표현을 사용하지 마세요.
                """),
                ("human", """
                다음 동료평가 상황들을 바탕으로 구체적인 피드백을 생성해주세요.

                평가 건수: {total_evaluations}
                
                **실제 동료평가 상황들**:
                {context_sentences}
                
                관찰된 긍정 특성: {positive_keywords}
                개선 필요 영역: {negative_keywords}

                **절대 금지**: "평가 대상자", "평가받는 사람", "해당 직원", "대상자" 표현 사용 금지
                **필수**: 위의 동료평가 상황들을 구체적으로 반영하여 작성

                다음 JSON 형식으로 작성하세요:
                {{
                  "강점": "동료평가 상황에서 관찰된 구체적 긍정 행동과 결과",
                  "우려": "동료평가 상황에서 나타난 개선 필요 부분을 건설적으로 표현",  
                  "협업관찰": "동료평가 상황에서 보여진 협업 스타일과 소통 방식"
                }}

                작성 지침:
                1. 위에 제시된 "실제 동료평가 상황들"의 내용을 구체적으로 반영
                2. 단순한 일반론이 아닌 실제 관찰된 사례 기반으로 작성
                3. "평가 대상자" 같은 표현을 사용하지 말고 직접 서술
                4. 동료평가 상황의 맥락과 세부사항을 활용

                올바른 작성 방향:
                - 동료평가 상황에서 언급된 구체적 행동이나 특성을 반영
                - 문제 해결, 의사소통, 협업 등의 실제 사례를 바탕으로 서술
                - 관찰된 상황의 맥락을 그대로 활용하되 "평가 대상자" 표현만 제거

                **중요**: 동료평가 상황들의 구체적 내용을 충실히 반영하여 작성하세요.
                """)
            ])
            
            # 메시지 생성 (평가 대상 정보 제거)
            messages = forced_prompt.format_messages(
                total_evaluations=len(state["평가하는사번_리스트"]),
                positive_keywords=", ".join(analysis.get("top_positive", {}).keys()) if analysis.get("top_positive") else "협업, 책임감",
                negative_keywords=", ".join(analysis.get("top_negative", {}).keys()) if analysis.get("top_negative") else "일부 개선 영역",
                context_sentences="\n".join([f"- {s}" for s in top_sentences]) if top_sentences else "- 기본적인 업무 협력 상황"
            )
            
            response = llm(messages)
            
            # JSON 파싱 시도 (더 강화된 방식)
            content = response.content.strip()
            print(f"[Debug] LLM 응답: {content[:200]}...")
            
            # JSON 추출 시도
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                feedback = json.loads(json_str)
                
                # 필수 키 확인
                required_keys = ["강점", "우려", "협업관찰"]
                if all(key in feedback for key in required_keys):
                    # 성공적으로 파싱됨
                    state["강점"] = [feedback["강점"]]
                    state["우려"] = [feedback["우려"]]
                    state["협업관찰"] = [feedback["협업관찰"]]
                    
                    print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 피드백 생성 성공!")
                    print(f"  - 강점: {state['강점'][0]}")
                    print(f"  - 우려: {state['우려'][0]}")
                    print(f"  - 협업관찰: {state['협업관찰'][0]}")
                    
                    # 임시 분석 데이터 정리
                    if "_weighted_analysis" in state:
                        del state["_weighted_analysis"]
                    
                    return state
                else:
                    print(f"[ImprovedFeedbackGenerationAgent] 필수 키 누락, 재시도...")
                    continue
            else:
                print(f"[ImprovedFeedbackGenerationAgent] JSON 형식 불일치, 재시도...")
                continue
                
        except json.JSONDecodeError as e:
            print(f"[ImprovedFeedbackGenerationAgent] JSON 파싱 오류: {str(e)}, 재시도...")
            continue
        except Exception as e:
            print(f"[ImprovedFeedbackGenerationAgent] 생성 오류: {str(e)}, 재시도...")
            continue
    
    # 3번 시도 후에도 실패하면 강제로 최소한의 응답 생성
    print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 3번 시도 실패, 최소한의 응답 강제 생성")
    
    # 키워드나 문장에서 추출하여 강제 생성
    keywords = " ".join(state.get("키워드모음", []))
    
    state["강점"] = [f"동료 평가에서 나타난 {keywords.split(',')[0] if keywords else '협업'} 등의 긍정적 특성을 바탕으로 팀에 기여하고 있습니다"]
    state["우려"] = ["지속적인 성장을 위해 일부 영역에서 추가적인 개발과 개선이 필요한 상황으로 보입니다"]
    state["협업관찰"] = ["팀 내에서의 역할 수행과 동료들과의 관계에서 전반적으로 안정적인 모습을 보여주고 있습니다"]
    
    # 임시 분석 데이터 정리
    if "_weighted_analysis" in state:
        del state["_weighted_analysis"]
    
    return state


def _get_top_sentences(state):
    """비중 높은 순으로 요약 문장 정렬"""
    if not state["동료평가요약줄글들"] or not state["비중"]:
        return []
    
    sentences_with_weights = list(zip(state["동료평가요약줄글들"], state["비중"]))
    sorted_sentences = sorted(sentences_with_weights, key=lambda x: x[1], reverse=True)
    
    # 상위 3개 문장만 선별 (핵심만)
    return [s[0] for s in sorted_sentences[:3]]


def database_storage_agent(state: Dict, engine) -> Dict:
    """
    동료평가 분석 결과를 feedback_reports 테이블에 저장하는 에이전트
    """
    
    employee_id = state["평가받는사번"]
    period_id = int(state["분기"])
    
    print(f"[DatabaseStorageAgent] {employee_id}: DB 저장 시작 (분기: {period_id})")
    
    try:
        # 1. peer_review_result 텍스트 생성 (줄바꿈 포함)
        peer_review_result = _format_peer_review_result(state)
        
        print(f"[DatabaseStorageAgent] 저장될 내용:")
        print(peer_review_result)
        print("-" * 50)
        
        # 2. team_evaluation_id 조회 (해당 분기, 해당 직원)
        team_evaluation_id = get_team_evaluation_id(engine, period_id, employee_id)
        
        if not team_evaluation_id:
            print(f"[DatabaseStorageAgent] {employee_id}: team_evaluation_id를 찾을 수 없음")
            return state
        
        # 3. 기존 데이터 확인 및 처리
        from sqlalchemy import text
        with engine.connect() as conn:
            # 기존 데이터 확인
            check_query = text("""
                SELECT feedback_report_id 
                FROM feedback_reports 
                WHERE team_evaluation_id = :team_evaluation_id 
                  AND emp_no = :emp_no
            """)
            existing = conn.execute(check_query, {
                "team_evaluation_id": team_evaluation_id,
                "emp_no": employee_id
            }).fetchone()
            
            if existing:
                # 기존 데이터 업데이트
                update_query = text("""
                    UPDATE feedback_reports 
                    SET peer_review_result = :peer_review_result
                    WHERE feedback_report_id = :feedback_report_id
                """)
                conn.execute(update_query, {
                    "peer_review_result": peer_review_result,
                    "feedback_report_id": row_to_dict(existing)["feedback_report_id"]
                })
                conn.commit()
                
                print(f"[DatabaseStorageAgent] {employee_id}: 기존 데이터 업데이트 완료")
                
            else:
                # 새 데이터 삽입
                insert_query = text("""
                    INSERT INTO feedback_reports 
                    (team_evaluation_id, emp_no, peer_review_result)
                    VALUES (:team_evaluation_id, :emp_no, :peer_review_result)
                """)
                result = conn.execute(insert_query, {
                    "team_evaluation_id": team_evaluation_id,
                    "emp_no": employee_id,
                    "peer_review_result": peer_review_result
                })
                conn.commit()
                
                feedback_report_id = result.lastrowid
                print(f"[DatabaseStorageAgent] {employee_id}: 새 데이터 삽입 완료 (ID: {feedback_report_id})")
        
        print(f"[DatabaseStorageAgent] {employee_id}: DB 저장 성공!")
        
    except Exception as e:
        print(f"[DatabaseStorageAgent] {employee_id}: DB 저장 실패 - {str(e)}")
        import traceback
        traceback.print_exc()
        # 에러가 발생해도 state는 반환 (파이프라인 중단 방지)
    
    return state


def _format_peer_review_result(state: Dict) -> str:
    """
    강점, 우려, 협업관찰을 줄바꿈 포함한 텍스트로 포맷팅
    """
    # 각 항목에서 첫 번째 요소 추출 (리스트 형태이므로)
    강점 = state["강점"][0] if state["강점"] else "동료들로부터 긍정적인 평가를 받고 있습니다."
    우려 = state["우려"][0] if state["우려"] else "지속적인 성장을 위한 개선 영역이 있습니다."
    협업관찰 = state["협업관찰"][0] if state["협업관찰"] else "팀 내에서 협업에 참여하고 있습니다."
    
    # 줄바꿈 포함하여 텍스트 생성
    peer_review_result = f"""강점: {강점}
우려: {우려}
협업관찰: {협업관찰}"""
    
    return peer_review_result