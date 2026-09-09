"""Customer risk profiles: baseline behaviour versus recent behaviour.

The page's whole point is the contrast between the two, so they are computed
over explicitly different windows and never blended: the baseline covers
everything *before* the recent window, and the recent block covers the last
seven days. Comparing a window against a baseline that includes it would mute
exactly the drift the page exists to show.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import Float, Select, case as sql_case, func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.db.enums import OPEN_CASE_STATUSES, RiskLevel
from app.db.models.case import Case
from app.db.models.customer import Beneficiary, Customer, Device
from app.db.models.transaction import RiskScore, Transaction
from app.schemas.common import TimeSeriesPoint
from app.schemas.customer import (
    BehaviourBaseline,
    CustomerIdentity,
    CustomerListItem,
    CustomerRiskProfile,
    RecentBehaviour,
)
from app.services.masking import mask_name

RECENT_WINDOW_DAYS = 7

# Drift bands for the behaviour-change indicator, as a percentage increase.
BEHAVIOUR_CHANGE_BANDS: tuple[tuple[float, RiskLevel], ...] = (
    (200.0, RiskLevel.CRITICAL),
    (100.0, RiskLevel.HIGH),
    (40.0, RiskLevel.MEDIUM),
)


def _risk_level_for(score: float) -> RiskLevel:
    if score >= settings.risk_level_critical_min:
        return RiskLevel.CRITICAL
    if score >= settings.risk_level_high_min:
        return RiskLevel.HIGH
    if score >= settings.risk_level_medium_min:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _pct_change(current: float, baseline: float) -> float:
    """Percentage change, guarding the zero-baseline case."""
    if baseline <= 0:
        return 0.0 if current <= 0 else 100.0
    return round((current - baseline) / baseline * 100, 1)


def _behaviour_level(amount_pct: float, frequency_pct: float) -> RiskLevel:
    worst = max(amount_pct, frequency_pct)
    for threshold, level in BEHAVIOUR_CHANGE_BANDS:
        if worst >= threshold:
            return level
    return RiskLevel.LOW


def build_list_query(
    *,
    search: str | None = None,
    risk_level: RiskLevel | None = None,
    city: str | None = None,
) -> Select:
    """Customers with their aggregated risk position.

    Aggregating in SQL keeps this to one query regardless of customer count.
    """
    stmt = (
        select(
            Customer,
            func.count(Transaction.id).label("transaction_count"),
            func.coalesce(func.avg(Transaction.amount.cast(Float)), 0).label("average_amount"),
            func.coalesce(func.max(RiskScore.business_score), 0).label("max_risk_score"),
            func.coalesce(func.avg(RiskScore.business_score.cast(Float)), 0).label(
                "average_risk_score"
            ),
            func.coalesce(
                func.sum(
                    sql_case(
                        (RiskScore.business_score >= settings.risk_level_high_min, 1), else_=0
                    )
                ),
                0,
            ).label("high_risk_count"),
            func.max(Transaction.occurred_at).label("last_activity_at"),
        )
        .outerjoin(Transaction, Transaction.customer_id == Customer.id)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.id)
        .group_by(Customer.id)
    )
    if search:
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(
            Customer.customer_ref.ilike(pattern) | Customer.full_name.ilike(pattern)
        )
    if city:
        stmt = stmt.where(Customer.home_city == city)
    if risk_level is not None:
        bounds = {
            RiskLevel.LOW: (0, settings.risk_level_medium_min - 1),
            RiskLevel.MEDIUM: (settings.risk_level_medium_min, settings.risk_level_high_min - 1),
            RiskLevel.HIGH: (settings.risk_level_high_min, settings.risk_level_critical_min - 1),
            RiskLevel.CRITICAL: (settings.risk_level_critical_min, 100),
        }[risk_level]
        stmt = stmt.having(
            func.coalesce(func.max(RiskScore.business_score), 0).between(*bounds)
        )
    return stmt


def to_list_item(row) -> CustomerListItem:
    customer = row[0]
    max_score = int(row.max_risk_score or 0)
    return CustomerListItem(
        id=customer.id,
        customer_ref=customer.customer_ref,
        display_name=mask_name(customer.full_name) or customer.customer_ref,
        home_city=customer.home_city,
        segment=customer.segment,
        transaction_count=int(row.transaction_count or 0),
        average_amount=round(float(row.average_amount or 0), 2),
        max_risk_score=max_score,
        average_risk_score=round(float(row.average_risk_score or 0), 1),
        risk_level=_risk_level_for(max_score),
        high_risk_count=int(row.high_risk_count or 0),
        last_activity_at=row.last_activity_at,
    )


def get_customer(db: Session, customer_ref: str) -> Customer:
    customer = db.execute(
        select(Customer).where(Customer.customer_ref == customer_ref)
    ).scalar_one_or_none()
    if customer is None:
        raise NotFoundError(f"Customer {customer_ref} could not be found.")
    return customer


def build_profile(db: Session, customer: Customer) -> CustomerRiskProfile:
    """Assemble the full risk profile page for one customer."""
    now = datetime.now(UTC)
    recent_start = now - timedelta(days=RECENT_WINDOW_DAYS)

    totals = db.execute(
        select(
            func.count(Transaction.id),
            func.coalesce(func.avg(Transaction.amount.cast(Float)), 0),
            func.min(Transaction.occurred_at),
            func.max(Transaction.occurred_at),
        ).where(Transaction.customer_id == customer.id)
    ).one()
    total_transactions = int(totals[0] or 0)

    scores = db.execute(
        select(
            func.coalesce(func.max(RiskScore.business_score), 0),
            func.coalesce(func.avg(RiskScore.business_score.cast(Float)), 0),
            func.coalesce(
                func.sum(
                    sql_case(
                        (RiskScore.business_score >= settings.risk_level_medium_min, 1), else_=0
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    sql_case(
                        (RiskScore.business_score >= settings.risk_level_high_min, 1), else_=0
                    )
                ),
                0,
            ),
        )
        .select_from(Transaction)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(Transaction.customer_id == customer.id)
    ).one()
    max_score = int(scores[0] or 0)

    baseline = _baseline(db, customer, recent_start, totals)
    recent = _recent(db, customer, recent_start, baseline)

    open_cases = int(
        db.execute(
            select(func.count(Case.id)).where(
                Case.customer_id == customer.id, Case.status.in_(OPEN_CASE_STATUSES)
            )
        ).scalar()
        or 0
    )

    return CustomerRiskProfile(
        identity=CustomerIdentity.model_validate(customer),
        risk_level=_risk_level_for(max_score),
        risk_score=max_score,
        average_risk_score=round(float(scores[1] or 0), 1),
        total_transactions=total_transactions,
        flagged_transactions=int(scores[2] or 0),
        high_risk_transactions=int(scores[3] or 0),
        open_cases=open_cases,
        baseline=baseline,
        recent=recent,
        amount_history=_amount_history(db, customer),
        risk_history=_risk_history(db, customer),
        top_reasons=_top_reasons(db, customer),
    )


def _baseline(
    db: Session, customer: Customer, recent_start: datetime, totals
) -> BehaviourBaseline:
    """Established behaviour, computed over everything before the recent window."""
    rows = db.execute(
        select(
            func.count(Transaction.id),
            func.coalesce(func.avg(Transaction.amount.cast(Float)), 0),
            func.coalesce(
                func.percentile_cont(0.5).within_group(Transaction.amount.cast(Float)), 0
            ),
            func.min(Transaction.occurred_at),
            func.max(Transaction.occurred_at),
        ).where(
            Transaction.customer_id == customer.id, Transaction.occurred_at < recent_start
        )
    ).one()

    count = int(rows[0] or 0)
    first_seen, last_seen = rows[3], rows[4]
    history_days = max((last_seen - first_seen).days, 1) if first_seen and last_seen else 1

    hour_rows = db.execute(
        select(
            func.min(func.extract("hour", Transaction.occurred_at)),
            func.max(func.extract("hour", Transaction.occurred_at)),
        ).where(
            Transaction.customer_id == customer.id, Transaction.occurred_at < recent_start
        )
    ).one()

    locations = [
        r[0]
        for r in db.execute(
            select(Transaction.location_city, func.count(Transaction.id))
            .where(
                Transaction.customer_id == customer.id,
                Transaction.occurred_at < recent_start,
            )
            .group_by(Transaction.location_city)
            .order_by(func.count(Transaction.id).desc())
            .limit(4)
        ).all()
    ]

    known_devices = int(
        db.execute(
            select(func.count(Device.id)).where(Device.customer_id == customer.id)
        ).scalar()
        or 0
    )
    known_beneficiaries = int(
        db.execute(
            select(func.count(Beneficiary.id)).where(Beneficiary.customer_id == customer.id)
        ).scalar()
        or 0
    )

    return BehaviourBaseline(
        average_amount=round(float(rows[1] or 0), 2),
        median_amount=round(float(rows[2] or 0), 2),
        average_daily_transactions=round(count / history_days, 2) if count else 0.0,
        active_hours_start=int(hour_rows[0] or 0),
        active_hours_end=int(hour_rows[1] or 23),
        usual_locations=locations or [customer.home_city],
        known_devices=known_devices,
        known_beneficiaries=known_beneficiaries,
        total_transactions=count,
        history_days=history_days,
    )


def _recent(
    db: Session, customer: Customer, recent_start: datetime, baseline: BehaviourBaseline
) -> RecentBehaviour:
    """The last seven days, measured against the baseline."""
    rows = db.execute(
        select(func.count(Transaction.id), func.coalesce(func.avg(Transaction.amount.cast(Float)), 0))
        .where(
            Transaction.customer_id == customer.id, Transaction.occurred_at >= recent_start
        )
    ).one()
    count = int(rows[0] or 0)
    average_amount = round(float(rows[1] or 0), 2)
    daily = round(count / RECENT_WINDOW_DAYS, 2)

    # Cities, devices and beneficiaries appearing in the window that never
    # appeared before it.
    def _new_count(column, table=None) -> int:
        prior = select(column).where(
            Transaction.customer_id == customer.id, Transaction.occurred_at < recent_start
        )
        current = select(func.count(func.distinct(column))).where(
            Transaction.customer_id == customer.id,
            Transaction.occurred_at >= recent_start,
            column.isnot(None),
            column.notin_(prior),
        )
        return int(db.execute(current).scalar() or 0)

    new_locations = _new_count(Transaction.location_city)
    new_devices = _new_count(Transaction.device_id)
    new_beneficiaries = _new_count(Transaction.beneficiary_id)

    amount_change = _pct_change(average_amount, baseline.average_amount)
    frequency_change = _pct_change(daily, baseline.average_daily_transactions)

    return RecentBehaviour(
        window_days=RECENT_WINDOW_DAYS,
        transaction_count=count,
        average_amount=average_amount,
        daily_transactions=daily,
        new_locations=new_locations,
        new_devices=new_devices,
        new_beneficiaries=new_beneficiaries,
        amount_change_pct=amount_change,
        frequency_change_pct=frequency_change,
        behaviour_change_level=_behaviour_level(amount_change, frequency_change),
    )


def _amount_history(db: Session, customer: Customer, days: int = 90) -> list[TimeSeriesPoint]:
    """Daily transaction value and count."""
    since = datetime.now(UTC) - timedelta(days=days)
    bucket = func.date_trunc("day", Transaction.occurred_at)
    rows = db.execute(
        select(
            bucket.label("bucket"),
            func.coalesce(func.avg(Transaction.amount.cast(Float)), 0),
            func.count(Transaction.id),
        )
        .where(Transaction.customer_id == customer.id, Transaction.occurred_at >= since)
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [
        TimeSeriesPoint(
            bucket=r[0].isoformat(), value=round(float(r[1]), 2), secondary=float(r[2])
        )
        for r in rows
    ]


def _risk_history(db: Session, customer: Customer, days: int = 90) -> list[TimeSeriesPoint]:
    since = datetime.now(UTC) - timedelta(days=days)
    bucket = func.date_trunc("day", Transaction.occurred_at)
    rows = db.execute(
        select(
            bucket.label("bucket"),
            func.coalesce(func.avg(RiskScore.business_score.cast(Float)), 0),
            func.coalesce(func.max(RiskScore.business_score), 0),
        )
        .select_from(Transaction)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(Transaction.customer_id == customer.id, Transaction.occurred_at >= since)
        .group_by(bucket)
        .order_by(bucket)
    ).all()
    return [
        TimeSeriesPoint(
            bucket=r[0].isoformat(), value=round(float(r[1]), 1), secondary=float(r[2])
        )
        for r in rows
    ]


def _top_reasons(db: Session, customer: Customer, limit: int = 6) -> list[dict]:
    from app.db.enums import DETECTION_REASON_LABELS, DetectionReason

    rows = db.execute(
        select(RiskScore.primary_reason, func.count(RiskScore.id))
        .join(Transaction, RiskScore.transaction_id == Transaction.id)
        .where(Transaction.customer_id == customer.id, RiskScore.primary_reason.isnot(None))
        .group_by(RiskScore.primary_reason)
        .order_by(func.count(RiskScore.id).desc())
        .limit(limit)
    ).all()

    result = []
    for code, count in rows:
        try:
            label = DETECTION_REASON_LABELS[DetectionReason(code)]
        except ValueError:
            label = str(code)
        result.append({"code": code, "label": label, "value": int(count)})
    return result
