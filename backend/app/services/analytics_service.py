"""Dashboard, analytics and reporting aggregates.

Every figure returned here is computed in SQL over stored rows. Nothing is
estimated, extrapolated or hardcoded - if a number cannot be derived from the
data it is omitted or reported as null, not filled in with something plausible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Float, Integer, case as sql_case, cast, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.enums import (
    CLOSED_CASE_STATUSES,
    DETECTION_REASON_LABELS,
    OPEN_CASE_STATUSES,
    AlertStatus,
    CaseStatus,
    DetectionReason,
    RiskLevel,
)
from app.db.models.alert import Alert
from app.db.models.case import Case, CaseTransaction
from app.db.models.customer import Customer
from app.db.models.transaction import RiskScore, Transaction
from app.schemas.analytics import (
    AnalyticsSummary,
    DashboardKpis,
    DashboardSummary,
    OutcomeAnalysis,
    RiskSummaryReport,
)
from app.schemas.common import CountByLabel, TimeSeriesPoint

#: Ranges offered by the dashboard's time selector.
RANGE_WINDOWS: dict[str, timedelta] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}

#: Bucket width per range, so each chart has a sensible number of points.
RANGE_GRANULARITY: dict[str, str] = {
    "24h": "hour",
    "7d": "day",
    "30d": "day",
    "90d": "week",
}


def _window(range_key: str) -> tuple[datetime, str]:
    delta = RANGE_WINDOWS.get(range_key, RANGE_WINDOWS["30d"])
    granularity = RANGE_GRANULARITY.get(range_key, "day")
    return datetime.now(UTC) - delta, granularity


def _bucket(granularity: str):
    return func.date_trunc(granularity, Transaction.occurred_at)


def get_kpis(db: Session) -> DashboardKpis:
    """Headline counters for the overview page."""
    total_transactions = int(db.execute(select(func.count(Transaction.id))).scalar() or 0)
    total_customers = int(db.execute(select(func.count(Customer.id))).scalar() or 0)

    scored, average_score = db.execute(
        select(func.count(RiskScore.id), func.avg(cast(RiskScore.business_score, Float)))
    ).one()
    scored = int(scored or 0)

    suspicious = int(
        db.execute(
            select(func.count(RiskScore.id)).where(
                RiskScore.business_score >= settings.risk_level_medium_min
            )
        ).scalar()
        or 0
    )
    high_risk = int(
        db.execute(
            select(func.count(RiskScore.id)).where(
                RiskScore.business_score >= settings.risk_level_high_min
            )
        ).scalar()
        or 0
    )
    open_cases = int(
        db.execute(
            select(func.count(Case.id)).where(Case.status.in_(OPEN_CASE_STATUSES))
        ).scalar()
        or 0
    )
    total_alerts = int(db.execute(select(func.count(Alert.id))).scalar() or 0)
    unread_alerts = int(
        db.execute(
            select(func.count(Alert.id)).where(Alert.status == AlertStatus.NEW)
        ).scalar()
        or 0
    )

    # Share of scored transactions that crossed the alert threshold. Defined
    # against scored rows, not all rows, so it does not drift as a backlog of
    # unscored transactions builds up.
    detection_rate = (
        round(
            int(
                db.execute(
                    select(func.count(RiskScore.id)).where(
                        RiskScore.business_score >= settings.alert_min_score
                    )
                ).scalar()
                or 0
            )
            / scored
            * 100,
            2,
        )
        if scored
        else 0.0
    )

    return DashboardKpis(
        total_transactions=total_transactions,
        suspicious_transactions=suspicious,
        high_risk_transactions=high_risk,
        open_investigations=open_cases,
        average_risk_score=round(float(average_score or 0.0), 1),
        detection_rate=detection_rate,
        scored_transactions=scored,
        total_customers=total_customers,
        total_alerts=total_alerts,
        unread_alerts=unread_alerts,
    )


def transaction_volume(db: Session, range_key: str = "30d") -> list[TimeSeriesPoint]:
    """Transactions per bucket over the selected range."""
    since, granularity = _window(range_key)
    bucket = _bucket(granularity)
    rows = db.execute(
        select(bucket.label("bucket"), func.count(Transaction.id))
        .where(Transaction.occurred_at >= since)
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [TimeSeriesPoint(bucket=r[0].isoformat(), value=float(r[1])) for r in rows]


def suspicious_trend(db: Session, range_key: str = "30d") -> list[TimeSeriesPoint]:
    """Suspicious transactions per bucket, with high-risk as a second series."""
    since, granularity = _window(range_key)
    bucket = _bucket(granularity)
    rows = db.execute(
        select(
            bucket.label("bucket"),
            func.count(RiskScore.id),
            func.sum(
                sql_case(
                    (RiskScore.business_score >= settings.risk_level_high_min, 1), else_=0
                )
            ),
        )
        .select_from(Transaction)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(
            Transaction.occurred_at >= since,
            RiskScore.business_score >= settings.risk_level_medium_min,
        )
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [
        TimeSeriesPoint(
            bucket=r[0].isoformat(), value=float(r[1]), secondary=float(r[2] or 0)
        )
        for r in rows
    ]


def risk_trend(db: Session, range_key: str = "30d") -> list[TimeSeriesPoint]:
    """Average risk score per bucket."""
    since, granularity = _window(range_key)
    bucket = _bucket(granularity)
    rows = db.execute(
        select(bucket.label("bucket"), func.avg(cast(RiskScore.business_score, Float)))
        .select_from(Transaction)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(Transaction.occurred_at >= since)
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [
        TimeSeriesPoint(bucket=r[0].isoformat(), value=round(float(r[1] or 0), 2))
        for r in rows
    ]


def high_risk_trend(db: Session, range_key: str = "30d") -> list[TimeSeriesPoint]:
    since, granularity = _window(range_key)
    bucket = _bucket(granularity)
    rows = db.execute(
        select(bucket.label("bucket"), func.count(RiskScore.id))
        .select_from(Transaction)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(
            Transaction.occurred_at >= since,
            RiskScore.business_score >= settings.risk_level_high_min,
        )
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [TimeSeriesPoint(bucket=r[0].isoformat(), value=float(r[1])) for r in rows]


def risk_distribution(db: Session) -> list[CountByLabel]:
    """Count per risk band, always returning all four."""
    rows = dict(
        db.execute(
            select(RiskScore.risk_level, func.count(RiskScore.id)).group_by(
                RiskScore.risk_level
            )
        ).all()
    )
    return [
        CountByLabel(
            label=level.value.title(), value=int(rows.get(level, 0)), code=level.value
        )
        for level in RiskLevel
    ]


def risk_by_location(db: Session, limit: int = 10) -> list[CountByLabel]:
    """Suspicious transaction count by city.

    Synthetic data. This says nothing about real geographic fraud patterns.
    """
    rows = db.execute(
        select(Transaction.location_city, func.count(RiskScore.id))
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(RiskScore.business_score >= settings.risk_level_medium_min)
        .group_by(Transaction.location_city)
        .order_by(func.count(RiskScore.id).desc())
        .limit(limit)
    ).all()
    return [CountByLabel(label=r[0], value=int(r[1]), code=r[0]) for r in rows]


def detection_reasons(db: Session, limit: int = 10) -> list[CountByLabel]:
    """How often each rule was the leading factor."""
    rows = db.execute(
        select(RiskScore.primary_reason, func.count(RiskScore.id))
        .where(RiskScore.primary_reason.isnot(None))
        .group_by(RiskScore.primary_reason)
        .order_by(func.count(RiskScore.id).desc())
        .limit(limit)
    ).all()
    result: list[CountByLabel] = []
    for code, count in rows:
        try:
            label = DETECTION_REASON_LABELS[DetectionReason(code)]
        except ValueError:
            label = str(code)
        result.append(CountByLabel(label=label, value=int(count), code=str(code)))
    return result


def recent_alerts(db: Session, limit: int = 8) -> list[dict[str, Any]]:
    """Latest high-risk transactions for the dashboard panel."""
    rows = db.execute(
        select(Alert, Transaction, Customer)
        .join(Transaction, Alert.transaction_id == Transaction.id)
        .join(Customer, Transaction.customer_id == Customer.id)
        .order_by(Alert.triggered_at.desc())
        .limit(limit)
    ).all()
    from app.services.masking import mask_name

    return [
        {
            "alertRef": alert.alert_ref,
            "transactionRef": transaction.transaction_ref,
            "transactionId": transaction.id,
            "customerRef": customer.customer_ref,
            "customerName": mask_name(customer.full_name),
            "amount": float(transaction.amount),
            "currency": transaction.currency,
            "riskScore": alert.risk_score,
            "riskLevel": alert.risk_level.value,
            "reason": alert.reason_summary,
            "status": alert.status.value,
            "triggeredAt": alert.triggered_at.isoformat(),
        }
        for alert, transaction, customer in rows
    ]


def dashboard_summary(db: Session, range_key: str = "30d") -> DashboardSummary:
    return DashboardSummary(
        kpis=get_kpis(db),
        transaction_volume=transaction_volume(db, range_key),
        risk_distribution=risk_distribution(db),
        suspicious_trend=suspicious_trend(db, range_key),
        risk_by_location=risk_by_location(db),
        detection_reasons=detection_reasons(db),
        recent_alerts=recent_alerts(db),
        generated_at=datetime.now(UTC),
        is_demo_data=_is_demo_dataset(db),
    )


def _is_demo_dataset(db: Session) -> bool:
    """True when any transaction is flagged as demo data."""
    return bool(
        db.execute(select(Transaction.id).where(Transaction.is_demo.is_(True)).limit(1)).first()
    )


def outcome_analysis(db: Session) -> OutcomeAnalysis:
    """False-positive analysis, from real analyst decisions.

    Alerts with no case are counted as ``pending_review``: nobody has judged
    them, so treating them as either outcome would misrepresent the workload.
    """
    total_alerts = int(db.execute(select(func.count(Alert.id))).scalar() or 0)
    status_counts = dict(
        db.execute(select(Case.status, func.count(Case.id)).group_by(Case.status)).all()
    )

    cases_opened = sum(int(v) for v in status_counts.values())
    confirmed = int(status_counts.get(CaseStatus.RESOLVED, 0))
    false_positive = int(status_counts.get(CaseStatus.FALSE_POSITIVE, 0))
    closed_no_action = int(status_counts.get(CaseStatus.CLOSED, 0))
    reviewed = confirmed + false_positive + closed_no_action
    still_open = sum(int(status_counts.get(s, 0)) for s in OPEN_CASE_STATUSES)

    # Distinct alerts that have a case linked through their transaction.
    alerts_with_case = int(
        db.execute(
            select(func.count(func.distinct(Alert.id)))
            .select_from(Alert)
            .join(CaseTransaction, CaseTransaction.transaction_id == Alert.transaction_id)
        ).scalar()
        or 0
    )

    return OutcomeAnalysis(
        total_alerts=total_alerts,
        cases_opened=cases_opened,
        reviewed=reviewed,
        confirmed_suspicious=confirmed,
        false_positive=false_positive,
        closed_no_action=closed_no_action,
        still_open=still_open,
        pending_review=max(total_alerts - alerts_with_case, 0),
        false_positive_rate=(
            round(false_positive / reviewed * 100, 1) if reviewed else None
        ),
    )


def amount_by_risk_level(db: Session) -> list[CountByLabel]:
    """Total value transacted in each risk band."""
    rows = dict(
        db.execute(
            select(RiskScore.risk_level, func.sum(Transaction.amount))
            .join(Transaction, RiskScore.transaction_id == Transaction.id)
            .group_by(RiskScore.risk_level)
        ).all()
    )
    return [
        CountByLabel(
            label=level.value.title(), value=int(rows.get(level, 0) or 0), code=level.value
        )
        for level in RiskLevel
    ]


def payment_method_breakdown(db: Session) -> list[CountByLabel]:
    rows = db.execute(
        select(Transaction.payment_method, func.count(Transaction.id))
        .group_by(Transaction.payment_method)
        .order_by(func.count(Transaction.id).desc())
    ).all()
    return [
        CountByLabel(label=r[0].value.replace("_", " "), value=int(r[1]), code=r[0].value)
        for r in rows
    ]


def analytics_summary(db: Session, range_key: str = "30d") -> AnalyticsSummary:
    return AnalyticsSummary(
        transaction_volume=transaction_volume(db, range_key),
        risk_trend=risk_trend(db, range_key),
        high_risk_trend=high_risk_trend(db, range_key),
        detection_reasons=detection_reasons(db),
        risk_by_location=risk_by_location(db),
        amount_by_risk_level=amount_by_risk_level(db),
        payment_method_breakdown=payment_method_breakdown(db),
        outcome_analysis=outcome_analysis(db),
        generated_at=datetime.now(UTC),
    )


def risk_summary_report(
    db: Session, date_from: datetime | None = None, date_to: datetime | None = None
) -> RiskSummaryReport:
    """A risk summary over a period, for the Reports page."""
    period_end = date_to or datetime.now(UTC)
    period_start = date_from or (period_end - timedelta(days=30))

    scoped = (Transaction.occurred_at >= period_start, Transaction.occurred_at <= period_end)

    total, total_value = db.execute(
        select(func.count(Transaction.id), func.coalesce(func.sum(Transaction.amount), 0)).where(
            *scoped
        )
    ).one()

    suspicious, high_risk, avg_score = db.execute(
        select(
            func.sum(
                sql_case(
                    (RiskScore.business_score >= settings.risk_level_medium_min, 1), else_=0
                )
            ),
            func.sum(
                sql_case(
                    (RiskScore.business_score >= settings.risk_level_high_min, 1), else_=0
                )
            ),
            func.avg(cast(RiskScore.business_score, Float)),
        )
        .select_from(Transaction)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(*scoped)
    ).one()

    alerts_raised = int(
        db.execute(
            select(func.count(Alert.id)).where(
                Alert.triggered_at >= period_start, Alert.triggered_at <= period_end
            )
        ).scalar()
        or 0
    )
    cases_opened = int(
        db.execute(
            select(func.count(Case.id)).where(
                Case.opened_at >= period_start, Case.opened_at <= period_end
            )
        ).scalar()
        or 0
    )
    cases_closed = int(
        db.execute(
            select(func.count(Case.id)).where(
                Case.status.in_(CLOSED_CASE_STATUSES),
                Case.closed_at.isnot(None),
                Case.closed_at >= period_start,
                Case.closed_at <= period_end,
            )
        ).scalar()
        or 0
    )

    top_rows = db.execute(
        select(Transaction, RiskScore, Customer)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .join(Customer, Transaction.customer_id == Customer.id)
        .where(*scoped)
        .order_by(RiskScore.business_score.desc())
        .limit(20)
    ).all()

    from app.services.masking import mask_name

    return RiskSummaryReport(
        title="Risk Summary Report",
        period_start=period_start,
        period_end=period_end,
        generated_at=datetime.now(UTC),
        total_transactions=int(total or 0),
        total_value=float(total_value or 0),
        suspicious_transactions=int(suspicious or 0),
        high_risk_transactions=int(high_risk or 0),
        average_risk_score=round(float(avg_score or 0), 1),
        alerts_raised=alerts_raised,
        cases_opened=cases_opened,
        cases_closed=cases_closed,
        risk_distribution=risk_distribution(db),
        top_detection_reasons=detection_reasons(db),
        top_locations=risk_by_location(db),
        highest_risk_transactions=[
            {
                "transactionRef": t.transaction_ref,
                "customerRef": c.customer_ref,
                "customerName": mask_name(c.full_name),
                "amount": float(t.amount),
                "currency": t.currency,
                "occurredAt": t.occurred_at.isoformat(),
                "location": t.location_city,
                "riskScore": r.business_score,
                "riskLevel": r.risk_level.value,
                "primaryReason": r.primary_reason,
            }
            for t, r, c in top_rows
        ],
        is_demo_data=_is_demo_dataset(db),
    )
