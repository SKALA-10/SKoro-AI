"""
상태 관리 데이터클래스, 메인 에이전트 클래스
State management data classes and main agent class
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI
from datetime import datetime

from db_utils import DatabaseManager

class ReportType(Enum):
    EMPLOYEE_QUARTERLY = "employee_quarterly"  # 팀원용 분기별
    EMPLOYEE_ANNUAL = "employee_annual"        # 팀원용 연말

@dataclass
class ReportData:
    """레포트 데이터를 담는 클래스"""
    id: int
    report_json: Dict[str, Any]
    report_type: ReportType
    employee_name: str
    period: str
    table_name: str  # "feedback_reports" 또는 "final_evaluation_reports"

@dataclass
class ToneCorrection:
    """톤 보정 결과를 담는 클래스"""
    original_text: str
    corrected_text: str
    report_type: ReportType
    llm_response: str
    processing_time: float

class EmployeeReportToneAgent:
    """팀원용 레포트 톤 보정 Agent"""
    
    def __init__(self, engine: Engine, llm_client: ChatOpenAI):
        self.db_manager = DatabaseManager(engine)
        # LLMClient를 여기서 지연 import하여 순환 참조 방지
        from llm_utils import LLMClient
        self.llm_client = LLMClient(llm_client)
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'quarterly_reports': 0,
            'annual_reports': 0,
            'reconstructed': 0
        }

    def load_reports_from_db(self) -> List[ReportData]:
        """DB에서 모든 팀원용 레포트 로드 (재구성 방식 적용)"""
        # safe_json_loads를 여기서 지연 import
        from llm_utils import safe_json_loads
        
        reports = []
        print("=== DB에서 레포트 로드 시작 (재구성 방식) ===")

        try:
            # 분기별 레포트 로드
            print("\n📂 분기별 레포트 로드 중...")
            quarterly_reports = self.db_manager.fetch_employee_quarterly_reports()
            print(f"DB에서 {len(quarterly_reports)}개의 분기별 레포트를 가져왔습니다.")
               
            for report_row in quarterly_reports:
                try:
                    print(f"\n🔄 분기별 레포트 ID {report_row['id']} 처리 중...")
                
                    # 새로운 안전한 JSON 로드 사용
                    report_json = safe_json_loads(report_row['report'])
                    
                    if report_json is None:
                        print(f"❌ JSON 처리 최종 실패, 스킵")
                        continue
                        
                    print(f"✅ JSON 처리 성공")
                
                    report_data = ReportData(
                        id=report_row['id'],
                        report_json=report_json,
                        report_type=self.identify_report_type(report_json),
                        employee_name=self.extract_employee_name(report_json),
                        period=self.extract_period(report_json),
                        table_name="feedback_reports"
                    )
                
                    reports.append(report_data)
                    print(f"✅ 분기별 레포트 ID {report_row['id']} 로드 성공")
                    print(f"   - 직원명: {report_data.employee_name}")
                    print(f"   - 기간: {report_data.period}")
                
                except Exception as e:
                    print(f"❌ 레포트 처리 오류 (분기별 레포트 ID: {report_row['id']}): {e}")
                    continue

            # 연말 레포트 로드
            print("\n📂 연말 레포트 로드 중...")
            annual_reports = self.db_manager.fetch_employee_annual_reports()
            print(f"DB에서 {len(annual_reports)}개의 연말 레포트를 가져왔습니다.")

            for report_row in annual_reports:
                try:
                    print(f"\n🔄 연말 레포트 ID {report_row['id']} 처리 중...")
                
                    # 새로운 안전한 JSON 로드 사용
                    report_json = safe_json_loads(report_row['report'])
                    
                    if report_json is None:
                        print(f"❌ JSON 처리 최종 실패, 스킵")
                        continue
                        
                    print(f"✅ JSON 처리 성공")
                
                    report_data = ReportData(
                        id=report_row['id'],
                        report_json=report_json,
                        report_type=self.identify_report_type(report_json),
                        employee_name=self.extract_employee_name(report_json),
                        period=self.extract_period(report_json),
                        table_name="final_evaluation_reports"
                    )
                
                    reports.append(report_data)
                    print(f"✅ 연말 레포트 ID {report_row['id']} 로드 성공")
                    print(f"   - 직원명: {report_data.employee_name}")
                    print(f"   - 기간: {report_data.period}")
                
                except Exception as e:
                    print(f"❌ 레포트 처리 오류 (연말 레포트 ID: {report_row['id']}): {e}")
                    continue

        except Exception as e:
            print(f"❌ DB 로드 중 전체 오류: {e}")
            return []

        print(f"\n📊 레포트 로드 완료!")
        print(f"   - 총 로드된 레포트: {len(reports)}개")
        quarterly_count = sum(1 for r in reports if r.report_type == ReportType.EMPLOYEE_QUARTERLY)
        annual_count = sum(1 for r in reports if r.report_type == ReportType.EMPLOYEE_ANNUAL)
        print(f"   - 분기별 레포트: {quarterly_count}개")
        print(f"   - 연말 레포트: {annual_count}개")

        return reports
        reports = []
        print("=== DB에서 레포트 로드 시작 (재구성 방식) ===")

        try:
            # 분기별 레포트 로드
            print("\n📂 분기별 레포트 로드 중...")
            quarterly_reports = self.db_manager.fetch_employee_quarterly_reports()
            print(f"DB에서 {len(quarterly_reports)}개의 분기별 레포트를 가져왔습니다.")
               
            for report_row in quarterly_reports:
                try:
                    print(f"\n🔄 분기별 레포트 ID {report_row['id']} 처리 중...")
                
                    # 새로운 안전한 JSON 로드 사용
                    report_json = safe_json_loads(report_row['report'])
                    
                    if report_json is None:
                        print(f"❌ JSON 처리 최종 실패, 스킵")
                        continue
                        
                    print(f"✅ JSON 처리 성공")
                
                    report_data = ReportData(
                        id=report_row['id'],
                        report_json=report_json,
                        report_type=self.identify_report_type(report_json),
                        employee_name=self.extract_employee_name(report_json),
                        period=self.extract_period(report_json),
                        table_name="feedback_reports"
                    )
                
                    reports.append(report_data)
                    print(f"✅ 분기별 레포트 ID {report_row['id']} 로드 성공")
                    print(f"   - 직원명: {report_data.employee_name}")
                    print(f"   - 기간: {report_data.period}")
                
                except Exception as e:
                    print(f"❌ 레포트 처리 오류 (분기별 레포트 ID: {report_row['id']}): {e}")
                    continue

            # 연말 레포트 로드
            print("\n📂 연말 레포트 로드 중...")
            annual_reports = self.db_manager.fetch_employee_annual_reports()
            print(f"DB에서 {len(annual_reports)}개의 연말 레포트를 가져왔습니다.")

            for report_row in annual_reports:
                try:
                    print(f"\n🔄 연말 레포트 ID {report_row['id']} 처리 중...")
                
                    # 새로운 안전한 JSON 로드 사용
                    report_json = safe_json_loads(report_row['report'])
                    
                    if report_json is None:
                        print(f"❌ JSON 처리 최종 실패, 스킵")
                        continue
                        
                    print(f"✅ JSON 처리 성공")
                
                    report_data = ReportData(
                        id=report_row['id'],
                        report_json=report_json,
                        report_type=self.identify_report_type(report_json),
                        employee_name=self.extract_employee_name(report_json),
                        period=self.extract_period(report_json),
                        table_name="final_evaluation_reports"
                    )
                
                    reports.append(report_data)
                    print(f"✅ 연말 레포트 ID {report_row['id']} 로드 성공")
                    print(f"   - 직원명: {report_data.employee_name}")
                    print(f"   - 기간: {report_data.period}")
                
                except Exception as e:
                    print(f"❌ 레포트 처리 오류 (연말 레포트 ID: {report_row['id']}): {e}")
                    continue

        except Exception as e:
            print(f"❌ DB 로드 중 전체 오류: {e}")
            return []

        print(f"\n📊 레포트 로드 완료!")
        print(f"   - 총 로드된 레포트: {len(reports)}개")
        quarterly_count = sum(1 for r in reports if r.report_type == ReportType.EMPLOYEE_QUARTERLY)
        annual_count = sum(1 for r in reports if r.report_type == ReportType.EMPLOYEE_ANNUAL)
        print(f"   - 분기별 레포트: {quarterly_count}개")
        print(f"   - 연말 레포트: {annual_count}개")

        return reports
    
    def identify_report_type(self, report_data: Dict[str, Any]) -> ReportType:
        """레포트 타입 식별"""
        try:
            period = report_data.get('기본_정보', {}).get('업무_수행_기간', '')
            
            if '연말' in period:
                return ReportType.EMPLOYEE_ANNUAL
            elif '분기' in period:
                return ReportType.EMPLOYEE_QUARTERLY
            
            return ReportType.EMPLOYEE_QUARTERLY
        except Exception as e:
            print(f"레포트 타입 식별 오류: {e}")
            return ReportType.EMPLOYEE_QUARTERLY
    
    def extract_employee_name(self, report_data: Dict[str, Any]) -> str:
        """직원 이름 추출"""
        try:
            basic_info = report_data.get('기본_정보', {})
            return basic_info.get('성명') or basic_info.get('이름') or "알 수 없음"
        except Exception as e:
            print(f"직원 이름 추출 오류: {e}")
            return "알 수 없음"
    
    def extract_period(self, report_data: Dict[str, Any]) -> str:
        """업무 수행 기간 추출"""
        try:
            return report_data.get('기본_정보', {}).get('업무_수행_기간', '기간 미상')
        except Exception as e:
            print(f"기간 추출 오류: {e}")
            return "기간 미상"
    
    def correct_report_tone(self, report_data: ReportData) -> ToneCorrection:
        """LLM을 사용한 레포트 톤 보정 실행"""
        start_time = datetime.now()
        
        try:
            original_text = json.dumps(report_data.report_json, ensure_ascii=False, indent=2)
            
            llm_response = self.llm_client.correct_tone(report_data.report_json, report_data.report_type)
            
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
            
            return ToneCorrection(
                original_text=original_text,
                corrected_text=corrected_text,
                report_type=report_data.report_type,
                llm_response=llm_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ 톤 보정 중 오류: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            original_text = json.dumps(report_data.report_json, ensure_ascii=False, indent=2)
            
            return ToneCorrection(
                original_text=original_text,
                corrected_text=original_text,
                report_type=report_data.report_type,
                llm_response=f"오류 발생: {str(e)}",
                processing_time=processing_time
            )
    
    def save_corrected_report(self, report_data: ReportData, corrected_text: str) -> bool:
        """보정된 레포트를 DB에 저장"""
        try:
            if report_data.table_name == "feedback_reports":
                return self.db_manager.update_feedback_report(report_data.id, corrected_text)
            elif report_data.table_name == "final_evaluation_reports":
                return self.db_manager.update_final_evaluation_report(report_data.id, corrected_text)
            else:
                print(f"❌ 알 수 없는 테이블명: {report_data.table_name}")
                return False
        except Exception as e:
            print(f"❌ DB 저장 오류 (ID: {report_data.id}): {e}")
            return False
    
    def process_all_reports(self) -> List[ToneCorrection]:
        """모든 팀원용 레포트 처리 및 DB 저장"""
        try:
            reports = self.load_reports_from_db()
        except Exception as e:
            print(f"❌ 레포트 로드 실패: {e}")
            return []
        
        if reports is None or len(reports) == 0:
            print("⚠️ 처리할 레포트가 없습니다.")
            return []
        
        corrections = []
        
        print(f"\n🚀 총 {len(reports)}개의 팀원용 레포트를 LLM으로 처리합니다.")
        
        for i, report_data in enumerate(reports, 1):
            print(f"\n[{i}/{len(reports)}] 처리 중: {report_data.employee_name} - {report_data.period}")
            
            try:
                self.processing_stats['total_processed'] += 1
                
                if report_data.report_type == ReportType.EMPLOYEE_QUARTERLY:
                    self.processing_stats['quarterly_reports'] += 1
                else:
                    self.processing_stats['annual_reports'] += 1
                
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
            'quarterly_reports': 0,
            'annual_reports': 0,
            'reconstructed': 0
        }
        print("📊 통계가 초기화되었습니다.")
    
    def print_summary(self):
        """처리 결과 요약 출력"""
        stats = self.processing_stats
        success_rate = (stats['successful'] / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
        
        print(f"""
📊 EmployeeReportToneAgent 처리 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔢 처리 통계:
   - 총 처리된 레포트: {stats['total_processed']}개
   - 성공: {stats['successful']}개
   - 실패: {stats['failed']}개
   - 성공률: {success_rate:.1f}%

📋 레포트 타입별:
   - 분기별 레포트: {stats['quarterly_reports']}개
   - 연말 레포트: {stats['annual_reports']}개

🔧 JSON 처리:
   - 재구성된 레포트: {stats['reconstructed']}개

🎯 Agent 상태: {'✅ 정상 완료' if stats['total_processed'] > 0 else '⏳ 대기 중'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)

# 테스트 및 검증 함수
def test_json_reconstruction(engine):
    """JSON 재구성 기능 테스트"""
    from llm_utils import safe_json_loads
    print("=== JSON 재구성 기능 테스트 ===")
    
    with engine.connect() as connection:
        from sqlalchemy import text
        query = text("SELECT report FROM feedback_reports WHERE report IS NOT NULL LIMIT 1")
        result = connection.execute(query).fetchone()
        
        if result:
            raw_text = result[0]
            
            # 재구성 테스트
            reconstructed = safe_json_loads(raw_text)
            
            if reconstructed:
                print("✅ JSON 재구성 성공!")
                print(f"재구성된 성명: {reconstructed.get('기본_정보', {}).get('성명', '불명')}")
                print(f"Task 개수: {len(reconstructed.get('팀_업무_목표_및_개인_달성률', {}).get('업무표', []))}")
                return True
            else:
                print("❌ JSON 재구성 실패")
                return False
        else:
            print("❌ 테스트 데이터 없음")
            return False

def quick_test(engine, llm_client):
    """빠른 테스트"""
    print("=== 빠른 Agent 테스트 ===")
    
    # JSON 재구성 테스트
    if not test_json_reconstruction(engine):
        print("❌ 기본 기능 테스트 실패")
        return
    
    # Agent 초기화 테스트
    try:
        agent = EmployeeReportToneAgent(engine, llm_client)
        print("✅ Agent 초기화 성공")
        
        # 레포트 로드 테스트
        reports = agent.load_reports_from_db()
        print(f"✅ 레포트 로드 성공: {len(reports)}개")
        
        if reports:
            print("✅ 모든 기본 기능 정상 작동")
            return True
        else:
            print("⚠️ 처리할 레포트가 없음")
            return True
            
    except Exception as e:
        print(f"❌ Agent 테스트 실패: {e}")
        return False