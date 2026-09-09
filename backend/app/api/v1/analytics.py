"""Analytics and reporting endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import AuditAction
from app.schemas.analytics import AnalyticsSummary, OutcomeAnalysis, RiskSummaryReport
from app.services import analytics_service, audit_service, export_service

router = APIRouter(tags=["Analytics & Reports"])

RangeKey = Literal["24h", "7d", "30d", "90d"]


@router.get(
    "/analytics",
    response_model=AnalyticsSummary,
    summary="Analytics: volume, risk trend, detection signals and outcomes",
)
def analytics(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_ANALYTICS))],
    range: RangeKey = Query(default="30d"),
) -> AnalyticsSummary:
    """All series computed from stored data at request time."""
    return analytics_service.analytics_summary(db, range)


@router.get(
    "/analytics/outcomes",
    response_model=OutcomeAnalysis,
    summary="False-positive analysis from recorded case outcomes",
)
def outcomes(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_ANALYTICS))],
) -> OutcomeAnalysis:
    """Derived from real analyst decisions.

    Alerts with no case are reported as pending rather than being counted as
    either a confirmation or a false positive.
    """
    return analytics_service.outcome_analysis(db)


@router.get(
    "/reports/risk-summary",
    response_model=RiskSummaryReport,
    summary="Risk summary report for a period",
)
def risk_summary(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_REPORTS))],
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
) -> RiskSummaryReport:
    """Period totals, distribution, leading signals and the highest-risk rows."""
    return analytics_service.risk_summary_report(db, date_from, date_to)


@router.get(
    "/reports/risk-summary/export",
    summary="Download the risk summary report as CSV or Excel",
    response_class=Response,
)
def export_risk_summary(
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_REPORTS))],
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    format: Literal["csv", "xlsx"] = Query(default="csv"),
) -> Response:
    """Export the report's highest-risk transactions table."""
    report = analytics_service.risk_summary_report(db, date_from, date_to)
    rows = [
        {
            "transaction_ref": row["transactionRef"],
            "customer_ref": row["customerRef"],
            "customer_name": row["customerName"],
            "amount": row["amount"],
            "currency": row["currency"],
            "occurred_at": row["occurredAt"],
            "location_city": row["location"],
            "device_ref": "",
            "payment_method": "",
            "risk_score": row["riskScore"],
            "risk_level": row["riskLevel"],
            "primary_reason": row["primaryReason"] or "",
            "status": "",
        }
        for row in report.highest_risk_transactions
    ]

    payload = (
        export_service.to_xlsx(rows, title="Risk Summary")
        if format == "xlsx"
        else export_service.to_csv(rows)
    )
    filename = export_service.export_filename(format, prefix="sentinel-risk-summary")

    audit_service.record_and_commit(
        db,
        action=AuditAction.EXPORT_PERFORMED,
        actor=user,
        resource_type="report",
        resource_id=filename,
        description=f"Exported the risk summary report as {format.upper()}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        new_state={"format": format, "row_count": len(rows)},
    )
    return Response(
        content=payload,
        media_type=export_service.MEDIA_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
