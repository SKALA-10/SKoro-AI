# ================================================================
# db_utils.py - 데이터베이스 관련 유틸리티
# ================================================================

import os
import json
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# .env 파일 로드 (쥬피터와 동일한 방식)
load_dotenv("C:/Users/Administrator/Desktop/skala10/SKoro-AI/.env", override=True)

# DB 연결 설정
def setup_database_connection():
    """데이터베이스 연결 설정"""
    
    print("=== 환경변수 확인 ===")
    print(f"DB_TYPE: {os.getenv('DB_TYPE')}")
    print(f"DB_USERNAME: {os.getenv('DB_USERNAME')}")
    print(f"DB_PASSWORD: {os.getenv('DB_PASSWORD')}")
    print(f"DB_HOST: {os.getenv('DB_HOST')}")
    print(f"DB_PORT: {os.getenv('DB_PORT')}")
    print(f"DB_NAME: {os.getenv('DB_NAME')}")
    
    db_type = os.getenv("DB_TYPE")
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # MariaDB/MySQL의 경우 mysql+pymysql 드라이버 사용
    if db_type.lower() in ["mariadb", "mysql"]:
        DATABASE_URL = (
            f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
    else:
        DATABASE_URL = (
            f"{db_type}://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

    print(
        f"Generated DATABASE_URL: mysql+pymysql://{db_username}:***@{db_host}:{db_port}/{db_name}"
    )

    # DB 엔진 생성
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # 연결 테스트
    try:
        with engine.connect() as connection:
            print("✅ Database connection successful!")
    except OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        raise e
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise e

    return engine

# 전역 엔진 인스턴스
engine = setup_database_connection()

def row_to_dict(row: Row) -> Dict:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()


# ================================================================
# DB 조회 함수들
# ================================================================

def fetch_employee_basic_info(emp_no: str) -> Optional[Dict]:
    """직원 기본 정보 조회"""
    with engine.connect() as connection:
        query = text(
            """
            SELECT emp_no, emp_name, cl, position, team_id
            FROM employees 
            WHERE emp_no = :emp_no
        """
        )
        result = connection.execute(query, {"emp_no": emp_no}).fetchone()
        return row_to_dict(result) if result else None


def fetch_task_data_for_passionate(
    emp_no: str, period_id: int, report_type: str
) -> List[Dict]:
    """Passionate 평가용 Task 데이터 조회"""
    with engine.connect() as connection:
        if report_type == "annual":
            # 연말: 전체 분기 데이터
            query = text(
                """
                SELECT ts.task_summary, ts.task_performance, task_id, period_id
                FROM task_summaries ts
                WHERE ts.task_id IN (
                    SELECT task_id FROM tasks WHERE emp_no = :emp_no
                ) AND ts.period_id <= :period_id
                ORDER BY ts.period_id
            """
            )
        else:
            # 분기: 해당 분기만
            query = text(
                """
                SELECT ts.task_summary, ts.task_performance, task_id, period_id
                FROM task_summaries ts
                WHERE ts.task_id IN (
                    SELECT task_id FROM tasks WHERE emp_no = :emp_no
                ) AND ts.period_id = :period_id
            """
            )

        results = connection.execute(
            query, {"emp_no": emp_no, "period_id": period_id}
        ).fetchall()
        return [row_to_dict(row) for row in results]


def fetch_task_data_for_proactive(
    emp_no: str, period_id: int, report_type: str
) -> List[Dict]:
    """Proactive 평가용 Task 데이터 조회"""
    return fetch_task_data_for_passionate(emp_no, period_id, report_type)


def fetch_task_data_for_professional(
    emp_no: str, period_id: int, report_type: str
) -> List[Dict]:
    """Professional 평가용 Task 데이터 조회"""
    return fetch_task_data_for_passionate(emp_no, period_id, report_type)


def fetch_peer_talk_data(emp_no: str, period_id: int) -> Dict:
    """Peer Talk 데이터 조회 (실제 테이블 구조 반영)"""
    with engine.connect() as connection:
        # feedback_reports에서 ai_peer_talk_summary 조회
        query = text(
            """
            SELECT fr.ai_peer_talk_summary
            FROM feedback_reports fr
            JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
            WHERE fr.emp_no = :emp_no AND te.period_id <= :period_id
            ORDER BY te.period_id DESC
            LIMIT 1
        """
        )
        result = connection.execute(
            query, {"emp_no": emp_no, "period_id": period_id}
        ).fetchone()

        if result and result.ai_peer_talk_summary:
            return {"peer_talk_summary": result.ai_peer_talk_summary}

        return {"peer_talk_summary": "동료평가 데이터 없음"}


def fetch_collaboration_matrix_data(emp_no: str, team_id: int, period_id: int) -> Dict:
    """협업 매트릭스에서 개인 데이터 추출"""
    with engine.connect() as connection:
        query = text(
            """
            SELECT ai_collaboration_matrix
            FROM team_evaluations
            WHERE team_id = :team_id AND period_id <= :period_id
            AND ai_collaboration_matrix IS NOT NULL
            ORDER BY period_id DESC
            LIMIT 1
        """
        )
        result = connection.execute(
            query, {"team_id": team_id, "period_id": period_id}
        ).fetchone()

        if result and result.ai_collaboration_matrix:
            try:
                matrix_data = json.loads(result.ai_collaboration_matrix)
                collaboration_matrix = matrix_data.get("collaboration_matrix", [])

                # 해당 직원의 데이터 찾기
                for member in collaboration_matrix:
                    if member.get("emp_no") == emp_no:
                        return member

            except json.JSONDecodeError:
                pass

        return {}


def fetch_evaluation_criteria_from_db(prompt_type: str = "4p_evaluation") -> str:
    """DB prompts 테이블에서 평가 기준 가져오기 - 쥬피터와 동일한 방식"""
    with engine.connect() as connection:
        query = text("""
            SELECT prompt 
            FROM prompts 
            LIMIT 1
        """)
        
        result = connection.execute(query, {"prompt_type": prompt_type}).fetchone()
        
        if result:
            return result.prompt
        else:
            raise ValueError(f"DB에서 {prompt_type} 평가 기준을 찾을 수 없습니다.")


# ================================================================
# DB 저장 함수들
# ================================================================

def save_quarterly_4p_results(feedback_report_id: int, integrated_result: Dict) -> bool:
    """분기 4P 결과를 feedback_reports 테이블에 저장"""

    # 각 P별 대표 evidence 1개씩 선택 (가장 임팩트 있는 것)
    passionate_highlight = (
        integrated_result["passionate"]["evidence"][0]
        if integrated_result["passionate"]["evidence"]
        else "열정적 업무 수행"
    )
    proactive_highlight = (
        integrated_result["proactive"]["evidence"][0]
        if integrated_result["proactive"]["evidence"]
        else "주도적 업무 진행"
    )
    professional_highlight = (
        integrated_result["professional"]["evidence"][0]
        if integrated_result["professional"]["evidence"]
        else "전문적 업무 수행"
    )
    people_highlight = (
        integrated_result["people"]["evidence"][0]
        if integrated_result["people"]["evidence"]
        else "협력적 업무 참여"
    )

    # 분기용 깔끔한 5줄 개조식 텍스트
    quarterly_text = f"""* Passionate 성과 하이라이트: {passionate_highlight}
* Proactive 주도적 성과: {proactive_highlight}
* Professional 전문성 발휘: {professional_highlight}  
* People 협업 기여: {people_highlight}
* 종합 평가: {integrated_result['comprehensive_assessment']}""".strip()

    quarterly_format = {
        "evaluation_text": quarterly_text,
        "scores": {
            "passionate": integrated_result["passionate"]["score"],
            "proactive": integrated_result["proactive"]["score"],
            "professional": integrated_result["professional"]["score"],
            "people": integrated_result["people"]["score"],
            "average": integrated_result["average_score"],
        },
    }

    with engine.connect() as connection:
        # ai_4p_evaluation 컬럼이 있는지 확인하고 없으면 추가
        try:
            query = text(
                """
                UPDATE feedback_reports 
                SET ai_4p_evaluation = :ai_4p_evaluation
                WHERE feedback_report_id = :feedback_report_id
            """
            )

            result = connection.execute(
                query,
                {
                    "feedback_report_id": feedback_report_id,
                    "ai_4p_evaluation": json.dumps(
                        quarterly_format, ensure_ascii=False
                    ),
                },
            )
            connection.commit()
            return result.rowcount > 0

        except Exception as e:
            print(f"분기 저장 오류: {e}")
            # 컬럼이 없다면 추가 시도
            try:
                connection.execute(
                    text(
                        "ALTER TABLE feedback_reports ADD COLUMN ai_4p_evaluation TEXT"
                    )
                )
                connection.commit()
                print("ai_4p_evaluation 컬럼 추가됨")

                # 다시 저장 시도
                result = connection.execute(
                    query,
                    {
                        "feedback_report_id": feedback_report_id,
                        "ai_4p_evaluation": json.dumps(
                            quarterly_format, ensure_ascii=False
                        ),
                    },
                )
                connection.commit()
                return result.rowcount > 0
            except Exception as e2:
                print(f"컬럼 추가 실패: {e2}")
                return False


def save_annual_4p_results(
    final_evaluation_report_id: int, integrated_result: Dict
) -> bool:
    """연말 4P 결과를 final_evaluation_reports 테이블에 저장"""

    # 연말용 상세 포맷
    annual_format = {
        "passionate": {
            "score": integrated_result["passionate"]["score"],
            "level": integrated_result["passionate"]["bars_level"],
            "reasoning": integrated_result["passionate"]["reasoning"],
            "evidence": integrated_result["passionate"]["evidence"],
            "improvement_points": integrated_result["passionate"]["improvement_points"],
        },
        "proactive": {
            "score": integrated_result["proactive"]["score"],
            "level": integrated_result["proactive"]["bars_level"],
            "reasoning": integrated_result["proactive"]["reasoning"],
            "evidence": integrated_result["proactive"]["evidence"],
            "improvement_points": integrated_result["proactive"]["improvement_points"],
        },
        "professional": {
            "score": integrated_result["professional"]["score"],
            "level": integrated_result["professional"]["bars_level"],
            "reasoning": integrated_result["professional"]["reasoning"],
            "evidence": integrated_result["professional"]["evidence"],
            "improvement_points": integrated_result["professional"][
                "improvement_points"
            ],
        },
        "people": {
            "score": integrated_result["people"]["score"],
            "level": integrated_result["people"]["bars_level"],
            "reasoning": integrated_result["people"]["reasoning"],
            "evidence": integrated_result["people"]["evidence"],
            "improvement_points": integrated_result["people"]["improvement_points"],
        },
        "overall": {
            "average_score": integrated_result["average_score"],
            "overall_level": integrated_result["overall_level"],
            "top_strength": integrated_result["top_strength"],
            "improvement_area": integrated_result["improvement_area"],
            "balance_analysis": integrated_result["balance_analysis"],
            "comprehensive_assessment": integrated_result["comprehensive_assessment"],
        },
    }

    with engine.connect() as connection:
        try:
            query = text(
                """
                UPDATE final_evaluation_reports 
                SET ai_4p_evaluation = :ai_4p_evaluation
                WHERE final_evaluation_report_id = :final_evaluation_report_id
            """
            )

            result = connection.execute(
                query,
                {
                    "final_evaluation_report_id": final_evaluation_report_id,
                    "ai_4p_evaluation": json.dumps(annual_format, ensure_ascii=False),
                },
            )
            connection.commit()
            return result.rowcount > 0

        except Exception as e:
            print(f"연말 저장 오류: {e}")
            # 컬럼이 없다면 추가 시도
            try:
                connection.execute(
                    text(
                        "ALTER TABLE final_evaluation_reports ADD COLUMN ai_4p_evaluation TEXT"
                    )
                )
                connection.commit()
                print("final_evaluation_reports.ai_4p_evaluation 컬럼 추가됨")

                # 다시 저장 시도
                result = connection.execute(
                    query,
                    {
                        "final_evaluation_report_id": final_evaluation_report_id,
                        "ai_4p_evaluation": json.dumps(
                            annual_format, ensure_ascii=False
                        ),
                    },
                )
                connection.commit()
                return result.rowcount > 0
            except Exception as e2:
                print(f"컬럼 추가 실패: {e2}")
                return False