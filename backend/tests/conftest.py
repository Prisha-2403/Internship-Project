"""Shared pytest fixtures.

The suite is split in two:

* Tests over pure logic - scoring, feature engineering, masking, validation -
  run anywhere with no infrastructure.
* Tests over the API and persistence need PostgreSQL. Those depend on the
  ``db_session`` / ``client`` fixtures, which skip cleanly when no database is
  configured, so ``pytest`` is always runnable.

The database fixtures build an isolated ``sentinel_finance_test`` schema and
never touch development data.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# A signing key must exist before app.core.config is imported.
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-not-used-outside-the-test-suite")
os.environ.setdefault("ENVIRONMENT", "development")

from app.core.config import settings  # noqa: E402


def _database_available() -> bool:
    """True when a PostgreSQL instance accepts the configured credentials."""
    if not settings.postgres_password and not settings.database_url:
        return False
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.sqlalchemy_database_uri, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


DATABASE_AVAILABLE = _database_available()

requires_db = pytest.mark.skipif(
    not DATABASE_AVAILABLE,
    reason=(
        "PostgreSQL is not configured or unreachable. Run scripts/bootstrap_db.py "
        "and alembic upgrade head to enable database-backed tests."
    ),
)


@pytest.fixture(scope="session")
def engine():
    """Engine against a dedicated test database."""
    if not DATABASE_AVAILABLE:
        pytest.skip("PostgreSQL is not available")

    from sqlalchemy import create_engine, text

    test_db = f"{settings.postgres_db}_test"
    test_uri = settings.sqlalchemy_database_uri.rsplit("/", 1)[0] + f"/{test_db}"

    # The application role is deliberately NOCREATEDB - a least-privilege role
    # should not be able to create databases - so the test database is
    # provisioned by scripts/bootstrap_db.py rather than created here.
    try:
        test_engine = create_engine(test_uri, pool_pre_ping=True)
        with test_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(
            f"Test database {test_db!r} does not exist. "
            "Run: python -m scripts.bootstrap_db"
        )

    from app.db.models import Base

    # Enum types and the append-only rules come from the migration; for tests a
    # metadata create is enough and much faster.
    Base.metadata.create_all(test_engine)
    yield test_engine
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()


@pytest.fixture
def db_session(engine) -> Generator:
    """A session wrapped in a transaction that is rolled back after each test."""
    from sqlalchemy.orm import sessionmaker

    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def seeded_roles(db_session):
    """The four roles, as the seeder would create them."""
    from app.db.enums import ROLE_LABELS, ROLE_LEVELS, RoleCode
    from app.db.models.user import Role

    roles = {}
    for code in RoleCode:
        role = Role(
            code=code,
            name=ROLE_LABELS[code],
            description=f"{ROLE_LABELS[code]} role.",
            level=ROLE_LEVELS[code],
        )
        db_session.add(role)
        roles[code] = role
    db_session.flush()
    return roles


@pytest.fixture
def users(db_session, seeded_roles):
    """One active user per role, all sharing a known test password."""
    from app.core.security import hash_password
    from app.db.enums import RoleCode
    from app.db.models.user import User

    password_hash = hash_password(TEST_PASSWORD)
    created = {}
    for code in RoleCode:
        # Not a .local/.test/.invalid address: email-validator rejects
        # special-use domains, so EmailStr would 422 before reaching the route.
        user = User(
            email=f"{code.value.lower()}@sentineltest.com",
            full_name=f"Test {code.value.title()}",
            password_hash=password_hash,
            role_id=seeded_roles[code].id,
            is_active=True,
            is_demo=True,
        )
        db_session.add(user)
        created[code] = user
    db_session.flush()
    return created


TEST_PASSWORD = "Test-Password-12345"


@pytest.fixture
def client(db_session):
    """TestClient with the request session bound to the test transaction."""
    from fastapi.testclient import TestClient

    from app.db.session import get_db
    from app.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client, users):
    """Factory returning Bearer headers for a given role."""
    from app.core.rate_limit import login_limiter

    def _headers(role) -> dict[str, str]:
        login_limiter.clear()  # keep the limiter from leaking between tests
        response = client.post(
            "/api/auth/login",
            json={"email": users[role].email, "password": TEST_PASSWORD},
        )
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return _headers


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Stop one test's failed logins from blocking the next."""
    from app.core.rate_limit import login_limiter

    login_limiter.clear()
    yield
    login_limiter.clear()


@pytest.fixture
def sample_customer(db_session):
    """A customer with enough history for baselines to be meaningful."""
    from datetime import UTC, datetime, timedelta

    from app.db.models.customer import Beneficiary, Customer, Device

    now = datetime.now(UTC)
    customer = Customer(
        customer_ref="CUST-9001",
        full_name="Test Customer",
        email="test.customer@example.invalid",
        phone="9876543210",
        account_number="123456784821",
        home_city="Delhi",
        home_region="Delhi NCR",
        home_latitude=28.6139,
        home_longitude=77.2090,
        segment="RETAIL",
        kyc_level="FULL",
        onboarded_at=now - timedelta(days=400),
        is_active=True,
        is_demo=True,
    )
    db_session.add(customer)
    db_session.flush()

    device = Device(
        device_ref="DEV-0000000001",
        customer_id=customer.id,
        device_type="MOBILE",
        operating_system="Android",
        first_seen_at=now - timedelta(days=380),
        last_seen_at=now,
        is_trusted=True,
        usage_count=50,
    )
    beneficiary = Beneficiary(
        beneficiary_ref="BEN-000000001",
        customer_id=customer.id,
        display_name="Test Payee",
        account_number="999988887777",
        bank_name="Meridian Bank",
        first_seen_at=now - timedelta(days=300),
        payment_count=20,
    )
    db_session.add_all([device, beneficiary])
    db_session.flush()
    return customer, device, beneficiary
