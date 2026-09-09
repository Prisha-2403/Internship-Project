"""Global search for the top navigation bar."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission
from app.repositories import transaction_repository as repo
from app.services.masking import mask_name

router = APIRouter(tags=["Search"])


class SearchHit(BaseModel):
    type: str
    ref: str
    label: str
    detail: str | None = None


class SearchResults(BaseModel):
    query: str
    hits: list[SearchHit]


@router.get("/search", response_model=SearchResults, summary="Search transactions and customers")
def search(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_TRANSACTIONS))],
    q: str = Query(min_length=2, max_length=64, description="Transaction or customer reference."),
) -> SearchResults:
    """Look up a transaction or customer by reference or name.

    Customer names are masked, matching the rest of the interface.
    """
    raw = repo.search_global(db, q)
    hits: list[SearchHit] = []

    for item in raw["transactions"]:
        hits.append(
            SearchHit(
                type="transaction",
                ref=item["ref"],
                label=item["ref"],
                detail=f"{item['amount']:,.2f}",
            )
        )
    for item in raw["customers"]:
        hits.append(
            SearchHit(
                type="customer",
                ref=item["ref"],
                label=f"{item['ref']} - {mask_name(item['name'])}",
                detail=item["city"],
            )
        )
    return SearchResults(query=q, hits=hits)
