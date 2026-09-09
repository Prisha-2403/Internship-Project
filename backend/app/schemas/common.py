"""Shared response envelopes and query primitives."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas read directly from ORM instances."""

    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    """A slice of a result set plus the counts a table needs to paginate."""

    items: list[T]
    total: int = Field(description="Total rows matching the filters, ignoring pagination.")
    page: int = Field(description="1-based page number.")
    page_size: int
    total_pages: int

    @classmethod
    def build(cls, items: list[T], total: int, page: int, page_size: int) -> "Page[T]":
        total_pages = max(1, -(-total // page_size)) if page_size else 1
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )


class MessageResponse(BaseModel):
    """Simple acknowledgement for actions with nothing else to return."""

    message: str
    success: bool = True


class CountByLabel(BaseModel):
    """One category and its count - the shape every categorical chart consumes."""

    label: str
    value: int
    code: str | None = None


class TimeSeriesPoint(BaseModel):
    """One bucket in a time series."""

    bucket: str = Field(description="ISO date or datetime identifying the bucket.")
    value: float
    secondary: float | None = None
