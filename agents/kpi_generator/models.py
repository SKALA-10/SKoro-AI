from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, ForeignKey, BigInteger, String, Text, DateTime, Enum
import enum

Base = declarative_base()

class Grade(Base):
    __tablename__ = "grades"
    grade_id = Column(BigInteger, primary_key=True)
    grade_s = Column(String(255), nullable=True)
    grade_a = Column(String(255), nullable=True)
    grade_b = Column(String(255), nullable=True)
    grade_c = Column(String(255), nullable=True)
    grade_d = Column(String(255), nullable=True)
    grade_rule = Column(Text, nullable=True)
    task_id = Column(BigInteger, ForeignKey('tasks.task_id'), nullable=True)
    team_kpi_id = Column(BigInteger, ForeignKey('team_kpis.team_kpi_id'), nullable=True)

    def __repr__(self):
        return f"<Grade(grade_id={self.grade_id}, task_id={self.task_id}, team_kpi_id={self.team_kpi_id})>"


class TeamKpi(Base):
    __tablename__ = "team_kpis"

    team_kpi_id = Column(BigInteger, primary_key=True)
    year = Column(Integer, nullable=True)
    ai_kpi_progress_rate = Column(Integer, nullable=True)
    weight = Column(Integer, nullable=True)
    created_at = Column(DateTime(6), nullable=True)
    updated_at = Column(DateTime(6), nullable=True)
    team_id = Column(BigInteger, nullable=True)
    ai_kpi_analysis_comment = Column(Text, nullable=True)
    kpi_description = Column(String(255), nullable=True)
    kpi_name = Column(String(255), nullable=True)

    def __repr__(self):
        return (
            f"<TeamKpi(team_kpi_id={self.team_kpi_id}, kpi_name='{self.kpi_name}', "
            f"weight={self.weight}, year={self.year})>"
        )

class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(BigInteger, primary_key=True)
    start_date = Column(DateTime(6), nullable=True)
    end_date = Column(DateTime(6), nullable=True)
    task_name = Column(String(255), nullable=True)
    task_detail = Column(Text, nullable=True)
    target_level = Column(Text, nullable=True)
    weight = Column(Integer, nullable=True)
    emp_no = Column(String(255), nullable=True)
    team_kpi_id = Column(BigInteger, ForeignKey('team_kpis.team_kpi_id'), nullable=True)

    def __repr__(self):
        return f"<Task(task_id={self.task_id}, task_name='{self.task_name}', emp_no='{self.emp_no}')>"

class RoleEnum(enum.Enum):
    ADMIN = "ADMIN"
    MANAGER = "MANAGER"
    MEMBER = "MEMBER"

class Employee(Base):
    __tablename__ = "employees"

    emp_no = Column(String(255), primary_key=True, index=True)
    emp_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, unique=True)
    password = Column(String(255), nullable=True)
    profile_image = Column(String(255), nullable=True)
    position = Column(String(255), nullable=True)
    role = Column(Enum(RoleEnum), nullable=True)
    cl = Column(Integer, nullable=True)
    salary = Column(Integer, nullable=True)
    team_id = Column(BigInteger, ForeignKey("teams.team_id"), nullable=True)
    created_at = Column(DateTime(6), nullable=True)
    updated_at = Column(DateTime(6), nullable=True)

    def __repr__(self):
        return f"<Employee(emp_no={self.emp_no}, emp_name={self.emp_name}, role={self.role})>"

