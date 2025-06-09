from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float
from config.db import Base

class FinalEvaluationReport(Base):
    __tablename__ = "final_evaluation_reports"

    final_evaluation_report_id = Column(Integer, primary_key=True, autoincrement=True)
    report = Column(Text, nullable=True)
    ranking = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)
    contribution_rate = Column(Integer, nullable=True)
    skill = Column(String, nullable=True)
    ai_annual_achievement_rate = Column(Integer, nullable=True)
    ai_annual_performance_summary_comment = Column(Text, nullable=True)
    ai_annual_peer_talk_summary = Column(Text, nullable=True)

    team_evaluation_id = Column(Integer, ForeignKey("team_evaluations.team_evaluation_id"), nullable=False)
    emp_no = Column(String, ForeignKey("employees.emp_no"), nullable=False)

    # 관계 매핑 (필요 시 사용)
    # team_evaluation = relationship("TeamEvaluation", backref="final_evaluation_reports")
    # employee = relationship("Employee", backref="final_evaluation_reports")
