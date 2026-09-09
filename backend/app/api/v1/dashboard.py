"""Overview dashboard endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from app.auth.dependencies import CurrentUser, DbSession
from app.auth.permissions import Permission
from app.auth.dependencies import require_permission
from app.schemas.analytics import DashboardSummary
from app.services import analytics_service

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

RangeKey = Literal["24h", "7d", "30d", "90d"]


@router.get(
    "/summary",
    response_model=DashboardSummary,
    summary="Risk intelligence overview",
)
def summary(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_DASHBOARD))],
    range: RangeKey = Query(default="30d", description="Time window for the trend charts."),
) -> DashboardSummary:
    """KPIs, trends and recent alerts.

    Every figure is aggregated from stored rows at request time; nothing here is
    cached, estimated or hardcoded.
    """
    return analytics_service.dashboard_summary(db, range)
