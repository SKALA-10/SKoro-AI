from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from evaluation_feedback_service import EvaluationFeedbackService
from peer_evaluation_service import PeerEvaluationService

scheduler = BackgroundScheduler()
evaluation_feedback_service = EvaluationFeedbackService()
peer_evaluation_service = PeerEvaluationService()

# 이런게 있으면 진행하게 수정
# 동료 평가 동료 매칭
def start_match_peer_evaluators():
    today = datetime.now().date()
    # start_date - 7일이 오늘인 평가기간 조회
    period = YourService.get_periods_starting_in_7_days(today)

    # Scheduler를 실행할 Service 호출
    if period is not None:
        evaluation_feedback_service.match_peer_evaluators(period_id)

# 팀원 피드백 모아서 요약 및 저장
def start_summarize_and_save_team_feedback():
    today = datetime.now().date()
    # end_date + 7일이 오늘인 평가기간 조회
    period = YourService.get_periods_ended_7_days_ago(today)

    # Scheduler를 실행할 Service 호출
    if period is not None:
        peer_evaluation_service.summarize_and_save_team_feedback(period_id)
    
# 평가 실행
def start_evaluation():
    today = datetime.now().date()

    # end_date가 오늘인 평가기간 조회
    period = YourService.get_periods_ended_7_days_ago(today)

    # period 내부의 is_final 여부를 통해 평가 결정
    if period is not None:
        if is_final is True:
            # 최종 평가 실행
        else:
            # 분기 평가 실행(팀 피드백 레포트, 개인 피드백 레포트 호출)

# 매일 자정에 실행
scheduler.add_job(start_match_peer_evaluators, "cron", hour=0, minute=0)
scheduler.add_job(start_summarize_and_save_team_feedback, "cron", hour=0, minute=0)
scheduler.add_job(start_evaluation, "cron", hour=0, minute=0)

scheduler.start()
