"""Audit log, user management and system settings endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status as http_status
from sqlalchemy import func, select

from app.auth.dependencies import CurrentUser, DbSession, require_permission
from app.auth.permissions import Permission, permissions_for
from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.middleware import get_client_ip, get_user_agent
from app.core.security import hash_password, verify_password
from app.db.enums import AuditAction, RoleCode
from app.db.models.audit import AuditLog
from app.db.models.user import Role, User
from app.schemas.admin import (
    AuditLogOut,
    PasswordChange,
    RoleInfo,
    SystemSettings,
    UserCreate,
    UserUpdate,
)
from app.schemas.auth import UserOut, UserSummary
from app.schemas.common import MessageResponse, Page
from app.services import audit_service

router = APIRouter(tags=["Administration"])


# --- Audit log -------------------------------------------------------------
@router.get(
    "/audit-logs",
    response_model=Page[AuditLogOut],
    summary="Search the audit trail",
)
def audit_logs(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_AUDIT_LOGS))],
    action: AuditAction | None = Query(default=None),
    actor_id: int | None = Query(default=None),
    resource_type: str | None = Query(default=None, max_length=48),
    resource_id: str | None = Query(default=None, max_length=64),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    search: str | None = Query(default=None, max_length=120),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> Page[AuditLogOut]:
    """Read the append-only trail.

    There is no create, update or delete endpoint here by design: entries are
    written by the actions they describe, and the database refuses to modify
    them afterwards.
    """
    stmt = audit_service.build_query(
        action=action,
        actor_id=actor_id,
        resource_type=resource_type,
        resource_id=resource_id,
        date_from=date_from,
        date_to=date_to,
        search=search,
    )
    total = int(
        db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
    )
    rows = (
        db.execute(
            stmt.order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .unique()
        .scalars()
        .all()
    )
    return Page.build([AuditLogOut.model_validate(r) for r in rows], total, page, page_size)


@router.get(
    "/audit-logs/actions",
    response_model=list[str],
    summary="Audit action types for the filter dropdown",
)
def audit_actions(
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_AUDIT_LOGS))],
) -> list[str]:
    return [a.value for a in AuditAction]


# --- User management -------------------------------------------------------
@router.get("/users", response_model=list[UserOut], summary="List platform users")
def list_users(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.MANAGE_USERS))],
) -> list[UserOut]:
    users = db.execute(select(User).order_by(User.full_name)).unique().scalars().all()
    return [UserOut.model_validate(u) for u in users]


@router.get(
    "/users/assignable",
    response_model=list[UserSummary],
    summary="Users a case or alert can be assigned to",
)
def assignable_users(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_CASES))],
) -> list[UserSummary]:
    """Active users only - a case must not be assigned into a void."""
    users = (
        db.execute(select(User).where(User.is_active.is_(True)).order_by(User.full_name))
        .unique()
        .scalars()
        .all()
    )
    return [UserSummary.model_validate(u) for u in users]


@router.post(
    "/users",
    response_model=UserOut,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a user",
)
def create_user(
    payload: UserCreate,
    request: Request,
    db: DbSession,
    actor: Annotated[CurrentUser, Depends(require_permission(Permission.MANAGE_USERS))],
) -> UserOut:
    email = payload.email.strip().lower()
    if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
        raise ConflictError("An account with that email address already exists.")

    role = db.execute(select(Role).where(Role.code == payload.role_code)).scalar_one_or_none()
    if role is None:
        raise ValidationError("The selected role does not exist.")

    user = User(
        email=email,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role_id=role.id,
        is_active=True,
        is_demo=False,
    )
    db.add(user)
    db.flush()

    audit_service.record(
        db,
        action=AuditAction.USER_CREATED,
        actor=actor,
        resource_type="user",
        resource_id=user.id,
        description=f"Created user {user.email} with role {role.name}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        # Note: no password material is recorded, here or anywhere.
        new_state={"email": user.email, "role": role.code.value, "is_active": True},
    )
    db.commit()
    db.refresh(user)
    return UserOut.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserOut, summary="Update a user")
def update_user(
    user_id: int,
    payload: UserUpdate,
    request: Request,
    db: DbSession,
    actor: Annotated[CurrentUser, Depends(require_permission(Permission.MANAGE_USERS))],
) -> UserOut:
    """Change a user's name, role or active state."""
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError("That user could not be found.")

    previous = {
        "full_name": user.full_name,
        "role": user.role_code.value,
        "is_active": user.is_active,
    }
    changes = payload.model_dump(exclude_unset=True)
    action = AuditAction.USER_UPDATED

    if changes.get("full_name"):
        user.full_name = changes["full_name"].strip()

    if changes.get("role_code") is not None:
        new_code = RoleCode(changes["role_code"])
        if new_code != user.role_code:
            # Guard against removing the last administrator.
            if user.role_code is RoleCode.ADMIN and _admin_count(db) <= 1:
                raise ConflictError(
                    "This is the only administrator account. Promote another user first."
                )
            role = db.execute(select(Role).where(Role.code == new_code)).scalar_one_or_none()
            if role is None:
                raise ValidationError("The selected role does not exist.")
            user.role_id = role.id
            action = AuditAction.USER_ROLE_CHANGED

    if changes.get("is_active") is not None and changes["is_active"] != user.is_active:
        if not changes["is_active"]:
            if user.id == actor.id:
                raise ConflictError("You cannot deactivate your own account.")
            if user.role_code is RoleCode.ADMIN and _admin_count(db) <= 1:
                raise ConflictError(
                    "This is the only administrator account and cannot be deactivated."
                )
            action = AuditAction.USER_DEACTIVATED
        user.is_active = bool(changes["is_active"])

    db.flush()
    audit_service.record(
        db,
        action=action,
        actor=actor,
        resource_type="user",
        resource_id=user.id,
        description=f"Updated user {user.email}.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        previous_state=previous,
        new_state={
            "full_name": user.full_name,
            "role": user.role_code.value,
            "is_active": user.is_active,
        },
    )
    db.commit()
    db.refresh(user)
    return UserOut.model_validate(user)


def _admin_count(db: DbSession) -> int:
    return int(
        db.execute(
            select(func.count(User.id))
            .join(Role, User.role_id == Role.id)
            .where(Role.code == RoleCode.ADMIN, User.is_active.is_(True))
        ).scalar()
        or 0
    )


@router.post(
    "/users/me/password",
    response_model=MessageResponse,
    summary="Change your own password",
)
def change_password(
    payload: PasswordChange,
    request: Request,
    db: DbSession,
    user: CurrentUser,
) -> MessageResponse:
    """Requires the current password, so a hijacked session cannot lock out the owner."""
    if not verify_password(payload.current_password, user.password_hash):
        raise ValidationError("The current password is incorrect.")
    if verify_password(payload.new_password, user.password_hash):
        raise ValidationError("The new password must differ from the current one.")

    user.password_hash = hash_password(payload.new_password)
    audit_service.record(
        db,
        action=AuditAction.USER_UPDATED,
        actor=user,
        resource_type="user",
        resource_id=user.id,
        description="Changed own password.",
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
        # Deliberately no password material in either state snapshot.
        new_state={"password_changed": True},
    )
    db.commit()
    return MessageResponse(message="Your password has been updated.")


# --- Settings --------------------------------------------------------------
@router.get(
    "/settings",
    response_model=SystemSettings,
    summary="Effective system configuration (no secrets)",
)
def system_settings(
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_DASHBOARD))],
) -> SystemSettings:
    """Operational thresholds only.

    No signing key, database password or other credential is exposed here.
    """
    return SystemSettings(
        app_name=settings.app_name,
        environment=settings.environment,
        api_version="v1",
        risk_level_medium_min=settings.risk_level_medium_min,
        risk_level_high_min=settings.risk_level_high_min,
        risk_level_critical_min=settings.risk_level_critical_min,
        alert_min_score=settings.alert_min_score,
        ml_contamination=settings.ml_contamination,
        ml_n_estimators=settings.ml_n_estimators,
        ml_max_uplift_points=settings.ml_max_uplift_points,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        login_rate_limit_attempts=settings.login_rate_limit_attempts,
        login_rate_limit_window_seconds=settings.login_rate_limit_window_seconds,
        demo_data_enabled=settings.demo_data_enabled,
    )


@router.get(
    "/settings/roles",
    response_model=list[RoleInfo],
    summary="Roles and their effective permissions",
)
def role_matrix(
    db: DbSession,
    _: Annotated[CurrentUser, Depends(require_permission(Permission.VIEW_DASHBOARD))],
) -> list[RoleInfo]:
    """The permission matrix the UI renders - the same one the API enforces."""
    counts = dict(
        db.execute(
            select(Role.code, func.count(User.id))
            .outerjoin(User, User.role_id == Role.id)
            .group_by(Role.code)
        ).all()
    )
    roles = db.execute(select(Role).order_by(Role.level)).scalars().all()
    return [
        RoleInfo(
            code=role.code,
            name=role.name,
            description=role.description,
            level=role.level,
            permissions=sorted(p.value for p in permissions_for(role.code)),
            user_count=int(counts.get(role.code, 0)),
        )
        for role in roles
    ]
