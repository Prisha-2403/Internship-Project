"""Aggregates every v1 route module into one router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    admin,
    alerts,
    analytics,
    cases,
    customers,
    dashboard,
    imports,
    model,
    search,
    transactions,
)
from app.auth.router import router as auth_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(dashboard.router)
api_router.include_router(transactions.router)
api_router.include_router(customers.router)
api_router.include_router(alerts.router)
api_router.include_router(cases.router)
api_router.include_router(analytics.router)
api_router.include_router(model.router)
api_router.include_router(imports.router)
api_router.include_router(admin.router)
api_router.include_router(search.router)
