"""Customer risk profile endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.core.middleware import get_client_ip, get_user_agent
from app.db.enums import AuditAction, RiskLevel
from app.db.models.customer import Customer
from app.schemas.common import Page
from app.schemas.customer import CustomerListItem, CustomerRiskProfile
from app.services import audit_service, customer_service

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.get(
    "",
    response_model=Page[CustomerListItem],
    summary="List customers with their aggregated risk position",
)
def list_customers(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CUSTOMERS))],
    search: str | None = Query(default=None, max_length=120),
    risk_level: RiskLevel | None = Query(default=None),
    city: str | None = Query(default=None, max_length=64),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Page[CustomerListItem]:
    """Customer list, aggregated in SQL so it scales with the dataset."""
    stmt = customer_service.build_list_query(search=search, risk_level=risk_level, city=city)

    total = int(
        db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
    )
    rows = db.execute(
        stmt.order_by(func.coalesce(func.max(Customer.id), 0).desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    items = [customer_service.to_list_item(r) for r in rows]
    # Highest risk first - that is the order an analyst wants to work in.
    items.sort(key=lambda c: (c.max_risk_score, c.high_risk_count), reverse=True)
    return Page.build(items, total, page, page_size)


@router.get(
    "/{customer_ref}",
    response_model=CustomerRiskProfile,
    summary="Customer risk profile: baseline versus recent behaviour",
    responses={404: {"description": "No such customer."}},
)
def get_customer(
    customer_ref: str,
    request: Request,
    db: DbSession,
    user: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CUSTOMERS))],
) -> CustomerRiskProfile:
    """The full profile page.

    Personal details are masked in the response schema. The baseline block and
    the recent block are computed over deliberately separate windows so the
    contrast between them is real.
    """
    customer = customer_service.get_customer(db, customer_ref)
    profile = customer_service.build_profile(db, customer)

    audit_service.record_and_commit(
        db,
        action=AuditAction.CUSTOMER_VIEWED,
        actor=user,
        resource_type="customer",
        resource_id=customer.customer_ref,
        description=f"Viewed risk profile for {customer.customer_ref}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )
    return profile


@router.get(
    "/{customer_ref}/risk",
    response_model=CustomerRiskProfile,
    summary="Customer risk profile (alias of the profile endpoint)",
)
def get_customer_risk(
    customer_ref: str,
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CUSTOMERS))],
) -> CustomerRiskProfile:
    """Risk-only view. Same payload, without recording a profile view."""
    customer = customer_service.get_customer(db, customer_ref)
    return customer_service.build_profile(db, customer)
