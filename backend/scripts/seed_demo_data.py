"""Seed the demo dataset.

    python -m scripts.seed_demo_data              # populate if empty
    python -m scripts.seed_demo_data --reset      # wipe demo data first
    python -m scripts.seed_demo_data --transactions 80000

Creates roles, four demo users (one per role), customers, devices,
beneficiaries and transactions, then runs the full detection pipeline - feature
engineering, model training, scoring - and finally derives alerts, a set of
investigation cases with realistic timelines, and audit history.

All data is synthetic. Demo account passwords are generated at seed time and
written to the gitignored ``.env``; none is hardcoded in this file or anywhere
else in the repository.
"""

from __future__ import annotations

import argparse
import secrets
import string
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import REPO_ROOT, settings  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.db.enums import (  # noqa: E402
    ROLE_LABELS,
    ROLE_LEVELS,
    AlertStatus,
    AuditAction,
    CaseEventType,
    CasePriority,
    CaseStatus,
    RiskLevel,
    RoleCode,
    TransactionStatus,
)
from app.db.models.alert import Alert  # noqa: E402
from app.db.models.audit import AuditLog  # noqa: E402
from app.db.models.case import Case, CaseEvent, CaseNote, CaseTransaction  # noqa: E402
from app.db.models.customer import Beneficiary, Customer, Device  # noqa: E402
from app.db.models.ops import ImportJob, ModelVersion  # noqa: E402
from app.db.models.transaction import RiskScore, Transaction, TransactionFeature  # noqa: E402
from app.db.models.user import Role, User  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ml import pipeline  # noqa: E402
from app.services.synthetic_data import SyntheticDataGenerator  # noqa: E402

ROLE_DESCRIPTIONS = {
    RoleCode.ANALYST: "Reviews alerts, investigates transactions and opens cases.",
    RoleCode.SENIOR_ANALYST: "Assigns, escalates and resolves investigations.",
    RoleCode.MANAGER: "Oversees workload, closes cases and reviews the audit trail.",
    RoleCode.ADMIN: "Manages users, settings, data imports and model training.",
}

DEMO_USERS: tuple[tuple[RoleCode, str, str], ...] = (
    (RoleCode.ANALYST, "Priya Raman", "demo_analyst_email"),
    (RoleCode.SENIOR_ANALYST, "Arjun Mehta", "demo_senior_analyst_email"),
    (RoleCode.MANAGER, "Deepa Krishnan", "demo_manager_email"),
    (RoleCode.ADMIN, "Vikram Shah", "demo_admin_email"),
)

CASE_TITLES = (
    "Unusual high-value transfer requires review",
    "Multiple behavioural anomalies on account",
    "Rapid transaction sequence flagged for review",
    "Transaction from unrecognised device",
    "Out-of-pattern activity from new location",
    "Escalating transfer amounts to new beneficiaries",
)


def generate_demo_password() -> str:
    """A strong, readable password for the shared demo accounts."""
    alphabet = string.ascii_letters + string.digits
    return "Demo-" + "".join(secrets.choice(alphabet) for _ in range(16))


def write_demo_password(password: str) -> Path:
    """Record the generated password in the gitignored ``.env``."""
    env_path = REPO_ROOT / ".env"
    lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    )
    output, seen = [], False
    for line in lines:
        if line.startswith("DEMO_USER_PASSWORD="):
            output.append(f"DEMO_USER_PASSWORD={password}")
            seen = True
        else:
            output.append(line)
    if not seen:
        output.append(f"DEMO_USER_PASSWORD={password}")
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return env_path


def ensure_roles(db: Session) -> dict[RoleCode, Role]:
    """Create the four roles if absent; return them by code."""
    existing = {r.code: r for r in db.execute(select(Role)).scalars()}
    for code in RoleCode:
        if code in existing:
            continue
        role = Role(
            code=code,
            name=ROLE_LABELS[code],
            description=ROLE_DESCRIPTIONS[code],
            level=ROLE_LEVELS[code],
        )
        db.add(role)
        existing[code] = role
    db.flush()
    return existing


def ensure_users(db: Session, roles: dict[RoleCode, Role], password: str) -> dict[RoleCode, User]:
    """Create or refresh the four demo accounts."""
    digest = hash_password(password)
    users: dict[RoleCode, User] = {}
    for code, full_name, settings_attr in DEMO_USERS:
        email = getattr(settings, settings_attr).strip().lower()
        user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            user = User(
                email=email,
                full_name=full_name,
                password_hash=digest,
                role_id=roles[code].id,
                is_active=True,
                is_demo=True,
            )
            db.add(user)
        else:
            user.password_hash = digest
            user.role_id = roles[code].id
            user.full_name = full_name
            user.is_active = True
        users[code] = user
    db.flush()
    return users


def wipe_demo_data(db: Session) -> None:
    """Remove generated data, leaving roles and users intact.

    ``audit_logs`` is deliberately not touched: the table refuses DELETE by
    design, and wiping an audit trail is exactly what that rule exists to
    prevent.
    """
    print("  clearing existing demo data...")
    for model in (
        CaseEvent, CaseNote, CaseTransaction, Case, Alert,
        RiskScore, TransactionFeature, Transaction,
        Beneficiary, Device, Customer, ImportJob, ModelVersion,
    ):
        db.execute(delete(model))
    db.commit()


def seed_customers_and_transactions(
    db: Session, *, customer_count: int, target_transactions: int, history_days: int, seed: int
) -> tuple[int, int, dict[str, int]]:
    """Generate and persist customers, devices, beneficiaries and transactions."""
    print(f"  generating {target_transactions:,} synthetic transactions...")
    dataset = SyntheticDataGenerator(
        customer_count=customer_count,
        target_transactions=target_transactions,
        history_days=history_days,
        seed=seed,
    ).generate()

    print(f"  inserting {len(dataset.customers)} customers...")
    customer_rows = [
        {
            "customer_ref": c.customer_ref,
            "full_name": c.full_name,
            "email": c.email,
            "phone": c.phone,
            "account_number": c.account_number,
            "home_city": c.home_city,
            "home_region": c.home_region,
            "home_latitude": c.home_latitude,
            "home_longitude": c.home_longitude,
            "segment": c.segment,
            "kyc_level": c.kyc_level,
            "onboarded_at": c.onboarded_at,
            "is_active": True,
            "is_demo": True,
        }
        for c in dataset.customers
    ]
    db.bulk_insert_mappings(Customer, customer_rows)
    db.flush()

    customer_ids = {
        ref: cid
        for ref, cid in db.execute(select(Customer.customer_ref, Customer.id)).all()
    }

    device_rows, beneficiary_rows = [], []
    for c in dataset.customers:
        cid = customer_ids[c.customer_ref]
        for d in c.devices:
            device_rows.append(
                {
                    "device_ref": d.device_ref,
                    "customer_id": cid,
                    "device_type": d.device_type,
                    "operating_system": d.operating_system,
                    "first_seen_at": d.first_seen_at,
                    "last_seen_at": d.last_seen_at,
                    "is_trusted": d.is_trusted,
                    "usage_count": d.usage_count,
                }
            )
        for b in c.beneficiaries:
            beneficiary_rows.append(
                {
                    "beneficiary_ref": b.beneficiary_ref,
                    "customer_id": cid,
                    "display_name": b.display_name,
                    "account_number": b.account_number,
                    "bank_name": b.bank_name,
                    "first_seen_at": b.first_seen_at,
                    "payment_count": b.payment_count,
                }
            )
    print(f"  inserting {len(device_rows)} devices and {len(beneficiary_rows)} beneficiaries...")
    db.bulk_insert_mappings(Device, device_rows)
    db.bulk_insert_mappings(Beneficiary, beneficiary_rows)
    db.flush()

    device_ids = {
        (cid, ref): did
        for did, cid, ref in db.execute(
            select(Device.id, Device.customer_id, Device.device_ref)
        ).all()
    }
    beneficiary_ids = {
        (cid, ref): bid
        for bid, cid, ref in db.execute(
            select(Beneficiary.id, Beneficiary.customer_id, Beneficiary.beneficiary_ref)
        ).all()
    }

    print(f"  inserting {len(dataset.transactions):,} transactions...")
    transaction_rows = []
    for t in dataset.transactions:
        cid = customer_ids[dataset.customers[t.customer_index].customer_ref]
        transaction_rows.append(
            {
                "transaction_ref": t.transaction_ref,
                "customer_id": cid,
                "amount": round(t.amount, 2),
                "currency": "INR",
                "occurred_at": t.occurred_at,
                "location_city": t.location_city,
                "location_region": t.location_region,
                "latitude": t.latitude,
                "longitude": t.longitude,
                "device_id": device_ids.get((cid, t.device_ref)),
                "beneficiary_id": beneficiary_ids.get((cid, t.beneficiary_ref)),
                "payment_method": t.payment_method,
                "channel": t.channel,
                "status": TransactionStatus.COMPLETED,
                "is_demo": True,
                "injected_scenario": t.injected_scenario,
            }
        )
    for start in range(0, len(transaction_rows), 5000):
        db.bulk_insert_mappings(Transaction, transaction_rows[start : start + 5000])
    db.commit()
    return len(dataset.customers), len(dataset.transactions), dataset.scenario_counts


def seed_cases(db: Session, users: dict[RoleCode, User]) -> int:
    """Open investigations against the highest-risk transactions.

    Each case is walked through a plausible lifecycle so the timeline, the
    workload counters and the false-positive analytics all have real data
    behind them.
    """
    top = db.execute(
        select(Transaction, RiskScore)
        .join(RiskScore, RiskScore.transaction_id == Transaction.id)
        .where(RiskScore.risk_level.in_([RiskLevel.HIGH, RiskLevel.CRITICAL]))
        .order_by(RiskScore.business_score.desc())
        .limit(28)
    ).all()
    if not top:
        return 0

    analyst = users[RoleCode.ANALYST]
    senior = users[RoleCode.SENIOR_ANALYST]
    manager = users[RoleCode.MANAGER]

    # A spread of outcomes so the false-positive analysis is meaningful.
    outcomes: list[tuple[CaseStatus, CasePriority]] = [
        (CaseStatus.NEW, CasePriority.HIGH),
        (CaseStatus.UNDER_REVIEW, CasePriority.CRITICAL),
        (CaseStatus.UNDER_REVIEW, CasePriority.HIGH),
        (CaseStatus.ESCALATED, CasePriority.CRITICAL),
        (CaseStatus.RESOLVED, CasePriority.HIGH),
        (CaseStatus.FALSE_POSITIVE, CasePriority.MEDIUM),
        (CaseStatus.FALSE_POSITIVE, CasePriority.LOW),
        (CaseStatus.CLOSED, CasePriority.MEDIUM),
    ]

    created = 0
    for index, (transaction, score) in enumerate(top):
        status, priority = outcomes[index % len(outcomes)]
        opened_at = transaction.occurred_at + timedelta(minutes=2)

        case = Case(
            case_ref=f"CASE-{2000 + index}",
            title=CASE_TITLES[index % len(CASE_TITLES)],
            summary=(
                f"Automated detection flagged {transaction.transaction_ref} with a risk "
                f"score of {score.business_score}. Opened for analyst review."
            ),
            customer_id=transaction.customer_id,
            status=status,
            priority=priority,
            assigned_to=(analyst if index % 3 else senior).id,
            created_by=analyst.id,
            peak_risk_score=score.business_score,
            opened_at=opened_at,
            closed_at=(
                opened_at + timedelta(hours=4)
                if status in (CaseStatus.RESOLVED, CaseStatus.FALSE_POSITIVE, CaseStatus.CLOSED)
                else None
            ),
            resolution_note=_resolution_note(status),
        )
        db.add(case)
        db.flush()

        db.add(
            CaseTransaction(
                case_id=case.id, transaction_id=transaction.id, linked_by=analyst.id
            )
        )

        timeline: list[tuple[timedelta, CaseEventType, str, int, str | None, str | None]] = [
            (
                timedelta(0),
                CaseEventType.TRANSACTION_FLAGGED,
                f"Transaction {transaction.transaction_ref} automatically flagged "
                f"(risk score {score.business_score}).",
                analyst.id,
                None,
                None,
            ),
            (
                timedelta(minutes=2),
                CaseEventType.CREATED,
                f"Investigation {case.case_ref} opened.",
                analyst.id,
                None,
                status.value,
            ),
            (
                timedelta(minutes=6),
                CaseEventType.TRANSACTION_LINKED,
                f"Transaction {transaction.transaction_ref} linked as evidence.",
                analyst.id,
                None,
                transaction.transaction_ref,
            ),
            (
                timedelta(minutes=14),
                CaseEventType.NOTE_ADDED,
                "Analyst added an investigation note.",
                analyst.id,
                None,
                None,
            ),
        ]
        db.add(
            CaseNote(
                case_id=case.id,
                author_id=analyst.id,
                body=(
                    "Reviewed the detection factors against the customer's 90-day baseline. "
                    "Contacting the relationship team to confirm whether the activity was "
                    "customer-initiated."
                ),
                created_at=opened_at + timedelta(minutes=14),
            )
        )

        if status in (CaseStatus.ESCALATED, CaseStatus.RESOLVED, CaseStatus.CLOSED):
            timeline.append(
                (
                    timedelta(minutes=32),
                    CaseEventType.ASSIGNED,
                    f"Case assigned to {senior.full_name}.",
                    senior.id,
                    analyst.full_name,
                    senior.full_name,
                )
            )
        if status is not CaseStatus.NEW:
            timeline.append(
                (
                    timedelta(hours=1),
                    CaseEventType.STATUS_CHANGED,
                    f"Status changed to {status.value.replace('_', ' ').title()}.",
                    senior.id,
                    CaseStatus.NEW.value,
                    status.value,
                )
            )
        if status in (CaseStatus.RESOLVED, CaseStatus.FALSE_POSITIVE, CaseStatus.CLOSED):
            timeline.append(
                (
                    timedelta(hours=4),
                    CaseEventType.CLOSED,
                    f"Case closed as {status.value.replace('_', ' ').title()}.",
                    manager.id,
                    status.value,
                    status.value,
                )
            )
            db.add(
                CaseNote(
                    case_id=case.id,
                    author_id=senior.id,
                    body=_resolution_note(status) or "Investigation concluded.",
                    created_at=opened_at + timedelta(hours=4),
                )
            )

        for offset, event_type, description, actor_id, from_value, to_value in timeline:
            db.add(
                CaseEvent(
                    case_id=case.id,
                    actor_id=actor_id,
                    event_type=event_type,
                    description=description,
                    from_value=from_value,
                    to_value=to_value,
                    occurred_at=opened_at + offset,
                )
            )
        created += 1

    db.commit()
    return created


def _resolution_note(status: CaseStatus) -> str | None:
    if status is CaseStatus.RESOLVED:
        return (
            "Customer confirmed the transaction was not authorised. Account controls applied "
            "and the beneficiary added to the watch list."
        )
    if status is CaseStatus.FALSE_POSITIVE:
        return (
            "Customer confirmed the activity was legitimate - a planned high-value payment "
            "from a newly issued device. No further action required."
        )
    if status is CaseStatus.CLOSED:
        return "Reviewed with no further action required. Closed by the risk manager."
    return None


def seed_audit_history(db: Session, users: dict[RoleCode, User], summary: dict[str, int]) -> int:
    """Write the audit entries describing this seed run.

    These are genuine records of what the seeder did, not decorative filler.
    """
    admin = users[RoleCode.ADMIN]
    now = datetime.now(UTC)
    entries = [
        AuditLog(
            actor_id=admin.id,
            actor_email=admin.email,
            actor_role=RoleCode.ADMIN.value,
            action=AuditAction.IMPORT_PERFORMED,
            resource_type="dataset",
            resource_id="demo-seed",
            description=(
                f"Seeded {summary['transactions']:,} synthetic transactions across "
                f"{summary['customers']} demo customers."
            ),
            ip_address="127.0.0.1",
            new_state={k: v for k, v in summary.items() if isinstance(v, int)},
            created_at=now - timedelta(minutes=3),
        ),
        AuditLog(
            actor_id=admin.id,
            actor_email=admin.email,
            actor_role=RoleCode.ADMIN.value,
            action=AuditAction.MODEL_TRAINED,
            resource_type="model_version",
            resource_id=summary.get("model_version") or "none",
            description=(
                f"Trained anomaly detection model on {summary['transactions']:,} records."
            ),
            ip_address="127.0.0.1",
            created_at=now - timedelta(minutes=2),
        ),
        AuditLog(
            actor_id=admin.id,
            actor_email=admin.email,
            actor_role=RoleCode.ADMIN.value,
            action=AuditAction.DETECTION_RUN,
            resource_type="detection",
            resource_id="demo-seed",
            description=(
                f"Detection run scored {summary['transactions']:,} transactions and raised "
                f"{summary['alerts']} alerts."
            ),
            ip_address="127.0.0.1",
            created_at=now - timedelta(minutes=1),
        ),
    ]
    for user in users.values():
        entries.append(
            AuditLog(
                actor_id=user.id,
                actor_email=user.email,
                actor_role=user.role_code.value,
                action=AuditAction.LOGIN,
                resource_type="user",
                resource_id=str(user.id),
                description=f"Signed in as {ROLE_LABELS[user.role_code]}.",
                ip_address="127.0.0.1",
                created_at=now - timedelta(minutes=20),
            )
        )
    db.add_all(entries)
    db.commit()
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Sentinel Finance demo data.")
    parser.add_argument("--reset", action="store_true", help="Wipe demo data before seeding.")
    parser.add_argument("--customers", type=int, default=120)
    parser.add_argument("--transactions", type=int, default=40_000)
    parser.add_argument("--history-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Score with business rules only, without fitting a model.",
    )
    args = parser.parse_args()

    started = datetime.now(UTC)
    db: Session = SessionLocal()
    try:
        print("Sentinel Finance - demo data seed")
        print("=" * 60)

        roles = ensure_roles(db)
        password = settings.demo_user_password or generate_demo_password()
        users = ensure_users(db, roles, password)
        db.commit()
        print(f"  roles: {len(roles)}   demo users: {len(users)}")

        if args.reset:
            wipe_demo_data(db)

        existing = db.execute(select(func.count(Transaction.id))).scalar() or 0
        if existing and not args.reset:
            print(f"\n  {existing:,} transactions already present - nothing to do.")
            print("  Re-run with --reset to regenerate.")
            return 0

        customers, transactions, scenarios = seed_customers_and_transactions(
            db,
            customer_count=args.customers,
            target_transactions=args.transactions,
            history_days=args.history_days,
            seed=args.seed,
        )

        print("  running detection pipeline (features, training, scoring)...")
        run = (
            pipeline.run_detection(db)
            if args.skip_training
            else pipeline.train_and_detect(db, notes="Trained during demo data seed.")
        )
        print(
            f"    features={run.features_computed:,} scored={run.transactions_scored:,} "
            f"alerts={run.alerts_raised} model={run.model_version or 'rules-only'} "
            f"({run.duration_seconds:.1f}s)"
        )
        for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            print(f"      {level:9s} {run.level_counts.get(level, 0):,}")

        cases = seed_cases(db, users)
        print(f"  investigation cases: {cases}")

        audit_rows = seed_audit_history(
            db,
            users,
            {
                "customers": customers,
                "transactions": transactions,
                "alerts": run.alerts_raised,
                "cases": cases,
                "model_version": run.model_version,
            },
        )
        print(f"  audit entries: {audit_rows}")

        env_path = write_demo_password(password)

        print()
        print("=" * 60)
        print(f"Seed complete in {(datetime.now(UTC) - started).total_seconds():.1f}s")
        print()
        print("Injected demo scenarios (synthetic bookkeeping, NOT fraud labels):")
        for name, count in sorted(scenarios.items()):
            print(f"    {name:20s} {count:4d}")
        print()
        print("Demo sign-in accounts - all share the generated password below.")
        print(f"Password written to {env_path} as DEMO_USER_PASSWORD.")
        print()
        for code, _, attr in DEMO_USERS:
            print(f"    {ROLE_LABELS[code]:16s} {getattr(settings, attr)}")
        print(f"\n    password: {password}\n")
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
