"""Engine, session factory and the FastAPI request-scoped session dependency."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.sqlalchemy_database_uri,
    pool_pre_ping=True,   # transparently drop connections killed server-side
    pool_size=10,
    max_overflow=20,
    echo=False,           # never log SQL (it can contain customer data)
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a session per request and always close it.

    Routes commit explicitly; anything left open is rolled back so a failed
    request can never leak a partial write into the next one.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
