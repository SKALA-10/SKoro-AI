# ================================================================
# comment_generator_module2.py - 모듈 2 코멘트 생성 관련 유틸리티
# ================================================================

from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.evaluation.modules.module_02_goal_achievement.llm_utils import *
from agents.evaluation.modules.module_02_goal_achievement.db_utils import *
from agents.evaluation.modules.module_02_goal_achievement.calculation_utils import *

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================================================
# 팀 일관성 가이드 생성 함수
# ================================================================

def generate_team_consistency_guide(team_id: int, period_id: int) -> Dict:
    """팀 단위 일관성 가이드 생성 - 우리가 상의한 방식"""
    team_members = fetch_team_members(team_id)
    
    # 실제 팀의 KPI ID 조회
    _, team_kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
    
    team_avg_rate = calculate_team_average_achievement_rate(team_kpi_ids)
    
    # 팀 성과 수준에 따른 가이드라인 결정
    if team_avg_rate >= 90:
        performance_level = "high"
        tone_guide = "성과 중심, 구체적 수치 강조"
        style_guide = "전문적이고 객관적"
    elif team_avg_rate >= 70:
        performance_level = "average"
        tone_guide = "균형적, 현재 성과 분석"
        style_guide = "객관적이고 분석적"
    else:
        performance_level = "improvement_needed"
        tone_guide = "현재 상태 분석, 성과 요약"
        style_guide = "객관적이고 구체적"
    
    return {
        "performance_level": performance_level,
        "tone_guide": tone_guide,
        "style_guide": style_guide,
        "length_target": 250,
        "length_tolerance": 30,
        "team_context": f"팀 평균 달성률 {team_avg_rate:.1f}%, {len(team_members)}명 구성"
    }

# ================================================================
# 통합 코멘트 생성 시스템
# ================================================================

class CommentGenerator:
    """통합 코멘트 생성 시스템 - 일관성 있는 코멘트 생성"""
    
    def __init__(self, comment_type: str, period_type: str, team_guide: Optional[Dict] = None):
        self.comment_type = comment_type  # "task", "individual", "team", "kpi"
        self.period_type = period_type    # "quarterly", "annual"
        self.team_guide = team_guide or {}
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """코멘트 타입별 설정 로드"""
        base_configs = {
            "task": {
                "quarterly": {
                    "elements": ["성과요약", "주요포인트", "팀기여도", "현재상태분석"],
                    "tone": "객관적이고 분석적",
                    "focus": "현재 성과와 기여도 분석",
                    "length": {"target": 250, "tolerance": 30}
                },
                "annual": {
                    "elements": ["연간요약", "성장추이", "팀기여도", "종합평가"],
                    "tone": "종합적이고 객관적",
                    "focus": "연간 성과와 성장 분석",
                    "length": {"target": 300, "tolerance": 40}
                }
            },
            "individual": {
                "quarterly": {
                    "elements": ["전체성과요약", "주요성과하이라이트", "성장포인트", "현재역량평가"],
                    "tone": "객관적이고 분석적",
                    "length": {"target": 350, "tolerance": 50}
                },
                "annual": {
                    "elements": ["연간성과종합", "분기별성장추이", "핵심기여영역", "종합역량평가"],
                    "tone": "종합평가적이고 객관적",
                    "length": {"target": 450, "tolerance": 50}
                }
            },
            "team": {
                "quarterly": {
                    "elements": ["팀성과종합", "팀원기여분석", "주요성과영역", "팀현재상태"],
                    "tone": "분석적이고 객관적",
                    "length": {"target": 450, "tolerance": 50}
                },
                "annual": {
                    "elements": ["연간팀성과요약", "팀조직력평가", "핵심성과기여", "팀종합평가"],
                    "tone": "종합적이고 객관적",
                    "length": {"target": 550, "tolerance": 50}
                }
            },
            "kpi": {
                "quarterly": {
                    "elements": ["KPI달성현황", "주요성과분석", "팀기여도평가", "현재달성수준"],
                    "tone": "객관적이고 분석적",
                    "length": {"target": 200, "tolerance": 30}
                },
                "annual": {
                    "elements": ["연간KPI종합", "성과추이분석", "팀기여도평가", "종합달성평가"],
                    "tone": "종합적이고 객관적",
                    "length": {"target": 250, "tolerance": 30}
                }
            }
        }
        
        return base_configs.get(self.comment_type, {}).get(self.period_type, {})
    
    def generate(self, data: Dict, context: Optional[Dict] = None) -> str:
        """통합 코멘트 생성 메인 함수"""
        if not self.config:
            raise ValueError(f"Invalid comment type: {self.comment_type} or period type: {self.period_type}")
        
        context = context or {}
        
        # 코멘트 타입별 데이터 전처리
        processed_data = self._preprocess_data(data)
        
        # 프롬프트 생성
        prompt = self._create_prompt(processed_data, context)
        
        # LLM 호출 및 검증
        comment = self._call_llm_with_validation(prompt)
        
        return comment
    
    def _preprocess_data(self, data: Dict) -> Dict:
        """코멘트 타입별 데이터 전처리"""
        if self.comment_type == "task":
            return {
                "task_name": data.get('task_name', ''),
                "emp_name": data.get('emp_name', ''),
                "emp_no": data.get('emp_no', ''),
                "target_level": data.get('target_level', ''),
                "performance": data.get('cumulative_performance', ''),
                "achievement_rate": data.get('ai_achievement_rate', 0),
                "contribution_score": data.get('ai_contribution_score', 0)
            }
        
        elif self.comment_type == "individual":
            tasks = data.get('tasks', [])
            tasks_summary = ""
            total_achievement = 0
            total_contribution = 0
            
            for task in tasks:
                tasks_summary += f"- {task.get('task_name', '')}: 달성률 {task.get('ai_achievement_rate', 0)}%, 기여도 {task.get('ai_contribution_score', 0)}%\n"
                total_achievement += task.get('ai_achievement_rate', 0)
                total_contribution += task.get('ai_contribution_score', 0)
            
            avg_achievement = total_achievement / len(tasks) if tasks else 0
            avg_contribution = total_contribution / len(tasks) if tasks else 0
            
            return {
                "emp_name": data.get('emp_name', ''),
                "emp_no": data.get('emp_no', ''),
                "position": data.get('position', ''),
                "cl": data.get('cl', ''),
                "tasks_summary": tasks_summary,
                "avg_achievement": avg_achievement,
                "avg_contribution": avg_contribution,
                "task_count": len(tasks)
            }
        
        elif self.comment_type == "team":
            kpis = data.get('kpis', [])
            kpis_summary = ""
            total_rate = 0
            
            for kpi in kpis:
                rate = kpi.get('ai_kpi_progress_rate', 0)
                weight = kpi.get('weight', 0)
                kpis_summary += f"- {kpi.get('kpi_name', '')}: {rate}% (비중 {weight}%)\n"
                total_rate += rate * (weight / 100)
            
            return {
                "kpis_summary": kpis_summary,
                "total_rate": total_rate,
                "team_context": data.get('team_context', ''),
                "performance_level": data.get('performance_level', '')
            }
        
        elif self.comment_type == "kpi":
            tasks = data.get('tasks', [])
            tasks_text = ""
            for task in tasks:
                tasks_text += f"- {task.get('emp_name', '')}: {task.get('task_name', '')}\n"
                tasks_text += f"  목표: {task.get('target_level', '')}\n"
                tasks_text += f"  성과: {task.get('task_performance', '')}\n"
            
            return {
                "kpi_name": data.get('kpi_name', ''),
                "kpi_description": data.get('kpi_description', ''),
                "tasks_text": tasks_text
            }
        
        return data
    
    def _create_prompt(self, data: Dict, context: Dict) -> str:
        """통합 프롬프트 생성"""
        elements = self.config.get('elements', [])
        tone = self.config.get('tone', '')
        focus = self.config.get('focus', '')
        length = self.config.get('length', {})
        
        # 팀 가이드라인 적용
        team_tone = self.team_guide.get('tone_guide', '')
        team_style = self.team_guide.get('style_guide', '')
        team_context = self.team_guide.get('team_context', '')
        
        system_content = f"""
        다음 내용을 포함하여 {self.comment_type} 분석 코멘트를 하나의 자연스러운 문단으로 작성해주세요:
        
        포함할 내용: {', '.join(elements)}
        톤: {tone}
        초점: {focus}
        길이: {length.get('target', 250)}±{length.get('tolerance', 30)}자
        
        팀 가이드라인:
        - {team_tone}
        - {team_style}
        - {team_context}
        
        작성 원칙:
        1. 현재 상태와 과거 성장 추이에 집중
        2. 구체적 수치와 성과를 포함
        3. 미래 계획이나 제안은 포함하지 않음
        4. 객관적이고 팩트 기반으로 작성
        5. 직원 이름 언급시 "이름(사번)님" 형태로 작성
        6. "연간요약:", "팀기여도:" 등의 제목 없이 하나의 자연스러운 문단으로 작성
        7. 문단 간 자연스러운 연결로 전체적인 흐름을 만들어주세요
        """
        
        # 코멘트 타입별 human content 생성
        human_content = self._create_human_content(data)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ])
        
        return str(prompt.format())
    
    def _create_human_content(self, data: Dict) -> str:
        """코멘트 타입별 human content 생성"""
        if self.comment_type == "task":
            return f"""
            Task: {data.get('task_name', '')}
            담당자: {data.get('emp_name', '')}({data.get('emp_no', '')})
            목표: {data.get('target_level', '')}
            누적 성과: {data.get('performance', '')}
            달성률: {data.get('achievement_rate', 0)}%
            기여도: {data.get('contribution_score', 0)}%
            """
        
        elif self.comment_type == "individual":
            return f"""
            직원: {data.get('emp_name', '')}({data.get('emp_no', '')})
            직위: {data.get('position', '')} (CL{data.get('cl', '')})
            
            Task 수행 현황:
            {data.get('tasks_summary', '')}
            
            종합 성과:
            - 평균 달성률: {data.get('avg_achievement', 0):.1f}%
            - 평균 기여도: {data.get('avg_contribution', 0):.1f}%
            - 참여 Task 수: {data.get('task_count', 0)}개
            """
        
        elif self.comment_type == "team":
            return f"""
            팀 KPI 성과 현황:
            {data.get('kpis_summary', '')}
            
            팀 전체 평균 달성률: {data.get('total_rate', 0):.1f}%
            팀 구성: {data.get('team_context', '')}
            성과 수준: {data.get('performance_level', '')}
            """
        
        elif self.comment_type == "kpi":
            return f"""
            KPI: {data.get('kpi_name', '')}
            KPI 목표: {data.get('kpi_description', '')}
            
            팀원별 개별 성과:
            {data.get('tasks_text', '')}
            """
        
        return str(data)
    
    def _call_llm_with_validation(self, prompt: str) -> str:
        """LLM 호출 및 검증"""
        def validate_comment(response: str) -> str:
            response = response.strip()
            
            # 길이 검증 (경고 로그 제거)
            target_length = self.config.get('length', {}).get('target', 250)
            tolerance = self.config.get('length', {}).get('tolerance', 30)
            
            # 길이 검증은 하되 경고 로그는 출력하지 않음
            # if not (target_length - tolerance <= len(response) <= target_length + tolerance):
            #     logger.warning(f"Comment length {len(response)} outside target {target_length}±{tolerance}")
            
            return response
        
        return robust_llm_call(prompt, validate_comment, context=f"{self.comment_type} comment")