"""Application configuration, loaded from the environment.

Every secret and deployment-specific value is read from environment variables
(or a gitignored ``.env``).  Nothing sensitive is ever hardcoded here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed view over the process environment."""

    model_config = SettingsConfigDict(
        # Read the repo-root .env first, then backend/.env if present.
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---------------------------------------------------------
    app_name: str = "Sentinel Finance"
    environment: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api"

    # --- Database ------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sentinel_finance"
    postgres_user: str = "sentinel"
    postgres_password: str = ""
    database_url: str | None = None

    # Used only by scripts/bootstrap_db.py; the running app never touches these.
    postgres_superuser: str = "postgres"
    postgres_superuser_password: str = ""

    # --- Security ------------------------------------------------------------
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 1
    refresh_token_remember_me_days: int = 14
    cookie_secure: bool = False
    cookie_samesite: str = "lax"

    login_rate_limit_attempts: int = 5
    login_rate_limit_window_seconds: int = 900

    # --- CORS ----------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Risk scoring --------------------------------------------------------
    risk_level_medium_min: int = 30
    risk_level_high_min: int = 60
    risk_level_critical_min: int = 80
    alert_min_score: int = 60

    # --- Machine learning ----------------------------------------------------
    ml_contamination: float = 0.03
    ml_n_estimators: int = 200
    ml_random_state: int = 42
    ml_max_uplift_points: int = 15
    ml_artifact_dir: str = "artifacts"

    # --- Demo data -----------------------------------------------------------
    demo_data_enabled: bool = True
    demo_user_password: str = ""
    demo_analyst_email: str = "analyst@sentinel.demo"
    demo_senior_analyst_email: str = "senior.analyst@sentinel.demo"
    demo_manager_email: str = "manager@sentinel.demo"
    demo_admin_email: str = "admin@sentinel.demo"

    @field_validator("cookie_samesite")
    @classmethod
    def _validate_samesite(cls, value: str) -> str:
        allowed = {"lax", "strict", "none"}
        lowered = value.lower()
        if lowered not in allowed:
            raise ValueError(f"COOKIE_SAMESITE must be one of {sorted(allowed)}")
        return lowered

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_database_uri(self) -> str:
        """Full SQLAlchemy URL, assembled from parts unless DATABASE_URL is set."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifact_path(self) -> Path:
        path = Path(self.ml_artifact_dir)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return path

    def require_secret_key(self) -> str:
        """Return the JWT signing key, refusing to run with a missing/default one."""
        if not self.secret_key or self.secret_key.startswith("replace-me"):
            raise RuntimeError(
                "SECRET_KEY is not configured. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(64))"'
            )
        if self.is_production and len(self.secret_key) < 32:
            raise RuntimeError("SECRET_KEY must be at least 32 characters in production.")
        return self.secret_key


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton so the environment is parsed exactly once."""
    return Settings()


settings = get_settings()
