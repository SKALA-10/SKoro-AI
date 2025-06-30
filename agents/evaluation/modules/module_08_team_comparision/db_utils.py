# ================================================================
# db_utils_module8.py - 모듈 8 데이터베이스 관련 유틸리티
# ================================================================

import sys
import os
import json
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from dotenv import load_dotenv
from config.settings import *

load_dotenv()

# DB 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

def get_year_from_period(period_id: int) -> int:
    """period_id로 연도 조회"""
    with engine.connect() as connection:
        query = text("SELECT year FROM periods WHERE period_id = :period_id")
        result = connection.execute(query, {"period_id": period_id}).scalar_one_or_none()
        return result if result else 2025  # fallback

# ================================================================
# 데이터 조회 함수들
# ================================================================

def fetch_team_kpis_data(team_id: int, period_id: int) -> Optional[Dict]:
    """팀 KPI 데이터와 종합 달성률 조회"""
    with engine.connect() as connection:
        # period_id로 연도 계산
        year = get_year_from_period(period_id)
        
        # team_evaluations.average_achievement_rate 조회
        overall_query = text("""
            SELECT te.average_achievement_rate as overall_rate
            FROM team_evaluations te
            WHERE te.team_id = :team_id AND te.period_id = :period_id
        """)
        
        overall_result = connection.execute(overall_query, {
            "team_id": team_id, 
            "period_id": period_id
        }).fetchone()
        
        if not overall_result:
            return None
        
        # KPI 목록 조회 - team_kpi_id 추가
        kpi_query = text("""
            SELECT 
                tk.team_kpi_id,
                tk.kpi_name,
                tk.kpi_description,
                tk.ai_kpi_progress_rate as rate,
                tk.weight
            FROM team_kpis tk
            WHERE tk.team_id = :team_id AND tk.year = :year
            ORDER BY tk.team_kpi_id
        """)
        
        kpi_results = connection.execute(kpi_query, {"team_id": team_id, "year": year}).fetchall()
        
        kpis = []
        for row in kpi_results:
            kpis.append({
                "team_kpi_id": row.team_kpi_id,
                "kpi_name": row.kpi_name,
                "kpi_description": row.kpi_description or "",
                "rate": row.rate or 0,
                "weight": row.weight or 0
            })
        
        return {
            "team_id": team_id,
            "overall_rate": overall_result.overall_rate or 0,
            "kpis": kpis
        }

def fetch_multiple_teams_kpis(team_ids: List[int], period_id: int) -> List[Dict]:
    """여러 팀의 KPI 데이터 배치 조회"""
    if not team_ids:
        return []
    
    # period_id로 연도 계산
    year = get_year_from_period(period_id)
    
    with engine.connect() as connection:
        team_ids_str = ','.join(map(str, team_ids))
        query = text(f"""
            SELECT 
                tk.team_id,
                tk.team_kpi_id,
                tk.kpi_name,
                tk.kpi_description,
                tk.ai_kpi_progress_rate as rate,
                tk.weight
            FROM team_kpis tk
            WHERE tk.team_id IN ({team_ids_str}) AND tk.year = :year
            ORDER BY tk.team_id, tk.team_kpi_id
        """)
        
        results = connection.execute(query, {"year": year}).fetchall()
        
        # 팀별로 그룹화
        teams_kpis = {}
        for row in results:
            team_id = row.team_id
            if team_id not in teams_kpis:
                teams_kpis[team_id] = []
            
            teams_kpis[team_id].append({
                "team_id": team_id,
                "team_kpi_id": row.team_kpi_id,
                "kpi_name": row.kpi_name,
                "kpi_description": row.kpi_description or "",
                "rate": row.rate or 0,
                "weight": row.weight or 0
            })
        
        # 리스트로 평탄화
        all_kpis = []
        for team_kpis in teams_kpis.values():
            all_kpis.extend(team_kpis)
        
        return all_kpis

def fetch_team_evaluation_id(team_id: int, period_id: int) -> Optional[int]:
    """team_evaluation_id 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT team_evaluation_id 
            FROM team_evaluations 
            WHERE team_id = :team_id AND period_id = :period_id
        """)
        result = connection.execute(query, {
            "team_id": team_id, 
            "period_id": period_id
        }).scalar_one_or_none()
        return result

# ================================================================
# DB 저장 함수
# ================================================================

def save_team_comparison_results(team_evaluation_id: int, comparison_json: Dict) -> bool:
    """팀 비교 결과 DB 저장"""
    with engine.connect() as connection:
        # comparison_result 추출
        comparison_result = comparison_json.get("overall", {}).get("comparison_result", "")
        
        query = text("""
            UPDATE team_evaluations
            SET 
                ai_team_comparison = :comparison_json,
                relative_performance = :comparison_result
            WHERE team_evaluation_id = :team_evaluation_id
        """)
        
        result = connection.execute(query, {
            "team_evaluation_id": team_evaluation_id,
            "comparison_json": json.dumps(comparison_json, ensure_ascii=False),
            "comparison_result": comparison_result
        })
        connection.commit()
        return result.rowcount > 0