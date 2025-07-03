"""
중간 레포트 톤 보정 Agent - 상태 관리 및 메인 에이전트 클래스
Middle report tone correction agent - state management and main agent class
"""

import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List
from sqlalchemy import Engine, text
from langchain_openai import ChatOpenAI

from db_utils import MiddleDatabaseManager

@dataclass
class MiddleReportData:
    """중간 레포트 데이터를 담는 클래스"""
    id: int
    report_json: Dict[str, Any]
    team_name: str
    evaluation_period: str
    table_name: str  # "team_evaluations"

@dataclass
class MiddleToneCorrection:
    """중간 레포트 톤 보정 결과를 담는 클래스"""
    original_text: str
    corrected_text: str
    llm_response: str
    processing_time: float

class MiddleReportToneAgent:
    """중간 레포트 톤 보정 Agent"""
    
    def __init__(self, engine: Engine, llm_client: ChatOpenAI):
        self.db_manager = MiddleDatabaseManager(engine)
        # LLMClient를 여기서 지연 import하여 순환 참조 방지
        from llm_utils import MiddleLLMClient
        self.llm_client = MiddleLLMClient(llm_client)
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'reconstructed': 0
        }

    def load_reports_from_db(self) -> List[MiddleReportData]:
        """DB에서 모든 중간 레포트 로드"""
        # safe_json_loads_middle을 여기서 지연 import
        from llm_utils import safe_json_loads_middle
        
        reports = []
        print("=== 중간 레포트 로드 시작 ===")

        try:
            # team_evaluations 테이블의 middle_report 컬럼에서 레포트 로드
            print("\n📂 중간 레포트 로드 중...")
            middle_reports = self.db_manager.fetch_middle_reports()
            print(f"DB에서 {len(middle_reports)}개의 중간 레포트를 가져왔습니다.")
               
            for report_row in middle_reports:
                try:
                    print(f"\n🔄 중간 레포트 ID {report_row['id']} 처리 중...")
                
                    # 안전한 JSON 로드 사용
                    report_json = safe_json_loads_middle(report_row['middle_report'])
                    
                    if report_json is None:
                        print(f"❌ JSON 처리 최종 실패, 스킵")
                        continue
                        
                    print(f"✅ JSON 처리 성공")
                
                    report_data = MiddleReportData(
                        id=report_row['id'],
                        report_json=report_json,
                        team_name=self.extract_team_name(report_json),
                        evaluation_period=self.extract_evaluation_period(report_json),
                        table_name="team_evaluations"
                    )
                
                    reports.append(report_data)
                    print(f"✅ 중간 레포트 ID {report_row['id']} 로드 성공")
                    print(f"   - 팀명: {report_data.team_name}")
                    print(f"   - 평가기간: {report_data.evaluation_period}")
                
                except Exception as e:
                    print(f"❌ 레포트 처리 오류 (중간 레포트 ID: {report_row['id']}): {e}")
                    continue

        except Exception as e:
            print(f"❌ DB 로드 중 전체 오류: {e}")
            return []

        print(f"\n📊 중간 레포트 로드 완료!")
        print(f"   - 총 로드된 레포트: {len(reports)}개")

        return reports
    
    def extract_team_name(self, report_data: Dict[str, Any]) -> str:
        """팀명 추출"""
        try:
            team_info = report_data.get('중간평가_기본정보', {})
            return team_info.get('팀명') or "알 수 없는 팀"
        except Exception as e:
            print(f"팀명 추출 오류: {e}")
            return "알 수 없는 팀"
    
    def extract_evaluation_period(self, report_data: Dict[str, Any]) -> str:
        """평가 기간 추출"""
        try:
            team_info = report_data.get('중간평가_기본정보', {})
            return team_info.get('평가_기간') or "기간 미상"
        except Exception as e:
            print(f"평가 기간 추출 오류: {e}")
            return "기간 미상"
    
    def correct_report_tone(self, report_data: MiddleReportData) -> MiddleToneCorrection:
        """LLM을 사용한 중간 레포트 톤 보정 실행"""
        start_time = datetime.now()
        
        try:
            original_text = json.dumps(report_data.report_json, ensure_ascii=False, indent=2)
            
            llm_response = self.llm_client.correct_tone(report_data.report_json)
            
            try:
                corrected_text = llm_response
                if "```json" in llm_response:
                    corrected_text = llm_response.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_response:
                    corrected_text = llm_response.split("```")[1].strip()
                
                # JSON 유효성 검증
                json.loads(corrected_text)
                print(f"✅ LLM 응답 JSON 파싱 성공")
                
            except (json.JSONDecodeError, IndexError) as e:
                print(f"⚠️ LLM 응답 JSON 파싱 오류: {e}")
                corrected_text = original_text
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MiddleToneCorrection(
                original_text=original_text,
                corrected_text=corrected_text,
                llm_response=llm_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ 톤 보정 중 오류: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            original_text = json.dumps(report_data.report_json, ensure_ascii=False, indent=2)
            
            return MiddleToneCorrection(
                original_text=original_text,
                corrected_text=original_text,
                llm_response=f"오류 발생: {str(e)}",
                processing_time=processing_time
            )
    
    def save_corrected_report(self, report_data: MiddleReportData, corrected_text: str) -> bool:
        """보정된 레포트를 DB에 저장"""
        try:
            return self.db_manager.update_middle_report(report_data.id, corrected_text)
        except Exception as e:
            print(f"❌ DB 저장 오류 (ID: {report_data.id}): {e}")
            return False
    
    def process_all_reports(self) -> List[MiddleToneCorrection]:
        """모든 중간 레포트 처리 및 DB 저장"""
        try:
            reports = self.load_reports_from_db()
        except Exception as e:
            print(f"❌ 레포트 로드 실패: {e}")
            return []
        
        if reports is None or len(reports) == 0:
            print("⚠️ 처리할 중간 레포트가 없습니다.")
            return []
        
        corrections = []
        
        print(f"\n🚀 총 {len(reports)}개의 중간 레포트를 LLM으로 처리합니다.")
        
        for i, report_data in enumerate(reports, 1):
            print(f"\n[{i}/{len(reports)}] 처리 중: {report_data.team_name} - {report_data.evaluation_period}")
            
            try:
                self.processing_stats['total_processed'] += 1
                
                correction = self.correct_report_tone(report_data)
                corrections.append(correction)
                
                print(f"  ⏱️  처리 시간: {correction.processing_time:.2f}초")
                
                success = self.save_corrected_report(report_data, correction.corrected_text)
                if success:
                    print(f"  ✅ DB 업데이트 성공")
                    self.processing_stats['successful'] += 1
                else:
                    print(f"  ❌ DB 업데이트 실패")
                    self.processing_stats['failed'] += 1
                
            except Exception as e:
                print(f"  ❌ 처리 오류: {e}")
                self.processing_stats['failed'] += 1
                continue
        
        return corrections
    
    def get_processing_statistics(self) -> Dict[str, int]:
        """처리 통계 반환"""
        return self.processing_stats.copy()
    
    def reset_statistics(self):
        """통계 초기화"""
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'reconstructed': 0
        }
        print("📊 중간 레포트 Agent 통계가 초기화되었습니다.")
    
    def print_summary(self):
        """처리 결과 요약 출력"""
        stats = self.processing_stats
        success_rate = (stats['successful'] / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
        
        print(f"""
📊 MiddleReportToneAgent 처리 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔢 처리 통계:
   - 총 처리된 레포트: {stats['total_processed']}개
   - 성공: {stats['successful']}개
   - 실패: {stats['failed']}개
   - 성공률: {success_rate:.1f}%

🔧 JSON 처리:
   - 재구성된 레포트: {stats['reconstructed']}개

🎯 Agent 상태: {'✅ 정상 완료' if stats['total_processed'] > 0 else '⏳ 대기 중'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)

# 테스트 및 검증 함수
def test_middle_json_reconstruction(engine):
    """중간 레포트 JSON 재구성 기능 테스트"""
    from llm_utils import safe_json_loads_middle
    print("=== 중간 레포트 JSON 재구성 기능 테스트 ===")
    
    with engine.connect() as connection:
        query = text("SELECT middle_report FROM team_evaluations WHERE middle_report IS NOT NULL LIMIT 1")
        result = connection.execute(query).fetchone()
        
        if result:
            raw_text = result[0]
            
            # 재구성 테스트
            reconstructed = safe_json_loads_middle(raw_text)
            
            if reconstructed:
                print("✅ 중간 레포트 JSON 재구성 성공!")
                print(f"재구성된 팀명: {reconstructed.get('중간평가_기본정보', {}).get('팀명', '불명')}")
                print(f"평가기간: {reconstructed.get('중간평가_기본정보', {}).get('평가_기간', '불명')}")
                team_members = reconstructed.get('팀원별_중간평가', {}).get('팀원_평가표', [])
                print(f"팀원 수: {len(team_members)}명")
                return True
            else:
                print("❌ 중간 레포트 JSON 재구성 실패")
                return False
        else:
            print("❌ 테스트 데이터 없음")
            return False

def quick_test_middle(engine, llm_client):
    """중간 레포트 Agent 빠른 테스트"""
    print("=== 중간 레포트 Agent 빠른 테스트 ===")
    
    # JSON 재구성 테스트
    if not test_middle_json_reconstruction(engine):
        print("❌ 기본 기능 테스트 실패")
        return
    
    # Agent 초기화 테스트
    try:
        middle_agent = MiddleReportToneAgent(engine, llm_client)
        print("✅ 중간 레포트 Agent 초기화 성공")
        
        # 레포트 로드 테스트
        reports = middle_agent.load_reports_from_db()
        print(f"✅ 중간 레포트 로드 성공: {len(reports)}개")
        
        if reports:
            print("✅ 모든 기본 기능 정상 작동")
            return True
        else:
            print("⚠️ 처리할 중간 레포트가 없음")
            return True
            
    except Exception as e:
        print(f"❌ 중간 레포트 Agent 테스트 실패: {e}")
        return False