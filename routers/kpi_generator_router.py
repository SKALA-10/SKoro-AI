from typing import Dict, Any

from fastapi import APIRouter
from sqlalchemy import select

from agents.kpi_generator.agent import GeneratedKpiResponse, run_kpi_generation_for_team
from agents.kpi_generator.db_utils import AsyncSessionLocal
from agents.kpi_generator.models import Employee

router = APIRouter(tags=["KPI Generation"])

@router.post("/team/{team_id}/generate", response_model=GeneratedKpiResponse)
async def generate_team_kpis(team_id: int):
    async with AsyncSessionLocal() as db:
        result = await run_kpi_generation_for_team(team_id, db)
        await db.commit()
        return GeneratedKpiResponse(**result)


@router.post("/generate", response_model=Dict[str, Any])
async def generate_all_teams_kpis():
    all_results = {}
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Employee.team_id).where(Employee.team_id.isnot(None)).distinct())
        team_ids = [r[0] for r in res.all()]
        for team_id in team_ids:
            try:
                result = await run_kpi_generation_for_team(team_id, db)
                all_results[str(team_id)] = f"{len(result['tasks'])} tasks generated."
            except Exception as e:
                await db.rollback()
                all_results[str(team_id)] = f"Failed: {e}"
                continue
        await db.commit()
    return {"status": "Completed", "details": all_results}