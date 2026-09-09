"""Customer-side reference data: ``customers``, ``devices``, ``beneficiaries``.

PII columns here hold synthetic values only. Masking is applied in the Pydantic
response schemas so unmasked values never leave the API, whatever the caller.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.transaction import Transaction


class Customer(Base, TimestampMixin):
    """An account holder whose transactions are monitored."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-facing identifier, e.g. CUST-1024
    customer_ref: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str] = mapped_column(String(24), nullable=False)
    account_number: Mapped[str] = mapped_column(String(32), nullable=False)

    home_city: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    home_region: Mapped[str] = mapped_column(String(64), nullable=False)
    home_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    home_longitude: Mapped[float] = mapped_column(Float, nullable=False)

    segment: Mapped[str] = mapped_column(String(32), nullable=False, default="RETAIL")
    kyc_level: Mapped[str] = mapped_column(String(16), nullable=False, default="FULL")
    onboarded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    devices: Mapped[list["Device"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    beneficiaries: Mapped[list["Beneficiary"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Customer {self.customer_ref}>"


class Device(Base, TimestampMixin):
    """A device a customer has transacted from.

    ``first_seen_at`` is what the "new device" rule tests against - a device
    first seen at the moment of the transaction under review is new.
    """

    __tablename__ = "devices"
    __table_args__ = (
        UniqueConstraint("customer_id", "device_ref", name="uq_devices_customer_device"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_ref: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    device_type: Mapped[str] = mapped_column(String(32), nullable=False, default="MOBILE")
    operating_system: Mapped[str] = mapped_column(String(32), nullable=False, default="Android")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_trusted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    customer: Mapped["Customer"] = relationship(back_populates="devices")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="device")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Device {self.device_ref}>"


class Beneficiary(Base, TimestampMixin):
    """A payee a customer has sent money to."""

    __tablename__ = "beneficiaries"
    __table_args__ = (
        UniqueConstraint(
            "customer_id", "beneficiary_ref", name="uq_beneficiaries_customer_beneficiary"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    beneficiary_ref: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    account_number: Mapped[str] = mapped_column(String(32), nullable=False)
    bank_name: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    customer: Mapped["Customer"] = relationship(back_populates="beneficiaries")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="beneficiary")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Beneficiary {self.beneficiary_ref}>"
