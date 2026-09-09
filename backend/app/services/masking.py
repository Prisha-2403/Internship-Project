"""PII masking helpers.

Applied inside the Pydantic response schemas rather than at call sites, so an
unmasked value cannot leave the API even if a new endpoint forgets to mask.

Masking here is one-way presentation formatting. It is not encryption and is
not a substitute for restricting who may read a record in the first place.
"""

from __future__ import annotations


def mask_phone(phone: str | None) -> str | None:
    """``9876543210`` -> ``9876******``.

    Keeps the leading four digits so an analyst can correlate records without
    seeing the full subscriber number.
    """
    if not phone:
        return phone
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) <= 4:
        return "*" * len(digits)
    return digits[:4] + "*" * (len(digits) - 4)


def mask_account_number(account: str | None) -> str | None:
    """``123456784821`` -> ``XXXX XXXX 4821`` - last four only."""
    if not account:
        return account
    cleaned = account.replace(" ", "")
    if len(cleaned) <= 4:
        return cleaned
    return f"XXXX XXXX {cleaned[-4:]}"


def mask_email(email: str | None) -> str | None:
    """``priya.sharma@example.com`` -> ``p***********@e******.com``."""
    if not email or "@" not in email:
        return email
    local, _, domain = email.partition("@")
    domain_name, dot, tld = domain.partition(".")

    masked_local = local[0] + "*" * max(len(local) - 1, 1) if local else "*"
    masked_domain = domain_name[0] + "*" * max(len(domain_name) - 1, 1) if domain_name else "*"
    return f"{masked_local}@{masked_domain}{dot}{tld}"


def mask_name(name: str | None) -> str | None:
    """``Priya Sharma`` -> ``Priya S.``

    Analysts need a human handle to discuss a case; the full legal name is not
    required for that and so is not shown.
    """
    if not name:
        return name
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return name
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1][0]}."


def mask_device_ref(device_ref: str | None) -> str | None:
    """Show only the tail of a device fingerprint.

    ``DEV-8fa31c9e4b`` -> ``DEV-****1c9e4b``
    """
    if not device_ref:
        return device_ref
    prefix, sep, tail = device_ref.partition("-")
    if not sep or len(tail) <= 6:
        return device_ref
    return f"{prefix}-****{tail[-6:]}"
