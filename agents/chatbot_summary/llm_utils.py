"""
LLM 관련 유틸리티 모듈
팀 피드백 요약 시스템 - LLM 요약 생성 함수
"""

from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI


class LLMSummarizer:
    """LLM을 사용한 피드백 요약 클래스"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        LLM 요약기 초기화
        
        Args:
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도 (0-1)
        """
        self.llm_client = ChatOpenAI(model=model_name, temperature=temperature)
    
    def summarize_team_feedbacks(self, team_name: str, feedbacks: List[Dict]) -> Optional[str]:
        """
        팀 피드백을 LLM으로 요약
        
        Args:
            team_name: 팀 이름
            feedbacks: 피드백 리스트
            
        Returns:
            Optional[str]: 요약 결과 또는 None (실패 시)
        """
        # 피드백 텍스트 생성 (완전 익명)
        feedbacks_text = ""
        for i, feedback in enumerate(feedbacks, 1):
            feedbacks_text += f"{i}. {feedback['content']}\n"
        
        # 프롬프트 생성
        prompt = self._create_summary_prompt(team_name, feedbacks_text)
        
        try:
            response = self.llm_client.invoke(prompt)
            summary = response.content.strip()
            return summary
            
        except Exception as e:
            print(f"❌ LLM 요약 실패: {e}")
            return None
    
    def _create_summary_prompt(self, team_name: str, feedbacks_text: str) -> str:
        """
        요약용 프롬프트 생성
        
        Args:
            team_name: 팀 이름
            feedbacks_text: 피드백 텍스트
            
        Returns:
            str: 완성된 프롬프트
        """
        return f"""
당신은 HR 전문가입니다. 다음은 {team_name} 팀 사원들이 팀장에게 익명으로 전달한 피드백입니다.
이 내용들을 팀장이 보기 쉽도록 건설적이고 부드러운 톤으로 요약해주세요.

## 익명 피드백 내용:
{feedbacks_text}

## 요약 지침:
1. 주요 피드백을 카테고리별로 분류해주세요
2. 공통적으로 언급되는 사항이나 중요한 내용은 **굵게** 표시하여 강조해주세요
3. 개인을 특정하지 않고 전체적인 관점에서 요약해주세요
4. 팀장이 기분 나쁘지 않도록 부드럽고 건설적인 톤으로 작성해주세요
5. 비판보다는 개선 기회로 접근하여 긍정적으로 표현해주세요
6. 문장은 "-습니다", "-있습니다" 형태의 정중한 존댓말로 작성해주세요
7. 각 섹션 사이에 줄바꿈을 추가해서 읽기 쉽게 해주세요
8. 제목 없이 바로 내용부터 시작해주세요
9. 섹션 제목은 ## (큰제목)으로 표시해주세요
10. "불만", "문제" 같은 부정적인 단어 대신 "개선 희망사항", "발전 방향" 같은 긍정적인 표현을 사용해주세요

## 요약 작성 방법:
- 실제 피드백 내용을 바탕으로 구체적으로 작성해주세요
- 여러 팀원이 공통으로 언급한 내용은 **굵게** 강조해주세요
- 중요하거나 우선순위가 높은 내용도 **굵게** 강조해주세요
- 예시 문구를 그대로 사용하지 말고, 실제 피드백에 맞는 내용으로 작성해주세요

## 형식 예시:
## 1. [실제 카테고리명]
- 구체적인 피드백 내용을 요약합니다.
- 공통으로 언급된 중요한 사항은 굵은 글씨로 표시합니다.
- 추가적인 개선 방향을 제시합니다.

## 2. [실제 카테고리명]
- 실제 피드백을 바탕으로 한 구체적인 내용입니다.
- 반복적으로 나타난 핵심 포인트는 굵은 글씨로 강조합니다.
- 발전을 위한 제안사항을 포함합니다.

## 개선 방향 제안
- 실제 피드백을 종합한 구체적인 개선 방향을 제시합니다.
- 가장 중요하게 다뤄야 할 사항은 굵은 글씨로 강조합니다.
- 팀 발전을 위한 실질적인 제안을 포함합니다.

요약해주세요:
"""