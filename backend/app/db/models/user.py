"""Access-control tables: ``roles`` and ``users``."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.db.enums import RoleCode

if TYPE_CHECKING:
    from app.db.models.case import Case


class Role(Base, TimestampMixin):
    """One of the four access tiers.

    ``level`` mirrors ``enums.ROLE_LEVELS`` and is what permission checks
    compare, so privilege ordering lives in the database as well as in code.
    """

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[RoleCode] = mapped_column(
        Enum(RoleCode, name="role_code"), unique=True, nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    level: Mapped[int] = mapped_column(Integer, nullable=False)

    users: Mapped[list["User"]] = relationship(back_populates="role")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Role {self.code.value}>"


class User(Base, TimestampMixin):
    """A platform operator.

    ``password_hash`` holds a bcrypt digest; the plaintext is never stored,
    logged, or returned by any endpoint.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    role: Mapped["Role"] = relationship(back_populates="users", lazy="joined")
    assigned_cases: Mapped[list["Case"]] = relationship(
        back_populates="assignee", foreign_keys="Case.assigned_to"
    )

    @property
    def role_code(self) -> RoleCode:
        return self.role.code

    @property
    def role_level(self) -> int:
        return self.role.level

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email}>"
