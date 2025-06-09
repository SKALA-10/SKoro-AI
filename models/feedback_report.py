from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from config.db import Base

class FeedbackReport(Base):
    __tablename__ = "feedback_reports"

    feedback_report_id = Column(Integer, primary_key=True, autoincrement=True)
    report = Column(Text, nullable=True)
    ranking = Column(Integer, nullable=True)
    contribution_rate = Column(Integer, nullable=True)
    skill = Column(String, nullable=True)
    attitude = Column(String, nullable=True)
    ai_overall_contribution_summary_comment = Column(Text, nullable=True)
    ai_peer_talk_summary = Column(Text, nullable=True)

    team_evaluation_id = Column(Integer, ForeignKey("team_evaluations.team_evaluation_id"), nullable=False)
    emp_no = Column(String, ForeignKey("employees.emp_no"), nullable=False)

    # 관계 매핑 (필요 시 사용)
    # team_evaluation = relationship("TeamEvaluation", backref="feedback_reports")
    # employee = relationship("Employee", backref="feedback_reports")
