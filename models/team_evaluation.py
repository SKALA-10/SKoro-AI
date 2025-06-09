from sqlalchemy import Column, Integer, Text, Enum, ForeignKey
from config.db import Base
import enum

# 상태값 Enum 정의
class Status(enum.Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"

class TeamEvaluation(Base):
    __tablename__ = "team_evaluations"

    team_evaluation_id = Column(Integer, primary_key=True, autoincrement=True)
    report = Column(Text, nullable=True)
    status = Column(Enum(Status), nullable=True)
    average_achievement_rate = Column(Integer, nullable=True)
    relative_performance = Column(Integer, nullable=True)
    year_over_year_growth = Column(Integer, nullable=True)
    team_performance_summary = Column(Text, nullable=True)
    ai_team_overall_analysis_comment = Column(Text, nullable=True)

    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    period_id = Column(Integer, ForeignKey("periods.period_id"), nullable=False)

    # 관계 매핑 (필요 시 사용)
    # team = relationship("Team", backref="team_evaluations")
    # period = relationship("Period", backref="team_evaluations")
