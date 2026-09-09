"""Create the application role and database.

Run once, with PostgreSQL superuser credentials, before the first migration:

    python -m scripts.bootstrap_db

The superuser password is read from ``POSTGRES_SUPERUSER_PASSWORD`` (or prompted
for) and is used only here. The application itself connects as the far more
limited ``sentinel`` role this script creates.

The application role deliberately gets no CREATEDB and no superuser rights - it
needs to read and write its own tables and nothing else.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import string
import sys
from pathlib import Path

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import REPO_ROOT, settings  # noqa: E402


def generate_password(length: int = 32) -> str:
    """A strong password with no shell-hostile characters."""
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def connect_superuser(password: str) -> psycopg.Connection:
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_superuser,
        password=password,
        dbname="postgres",
        autocommit=True,
    )


def role_exists(conn: psycopg.Connection, role: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        return cur.fetchone() is not None


def database_exists(conn: psycopg.Connection, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        return cur.fetchone() is not None


def bootstrap(app_password: str, superuser_password: str, *, drop_existing: bool) -> None:
    role = settings.postgres_user
    database = settings.postgres_db

    with connect_superuser(superuser_password) as conn:
        if drop_existing:
            for name in (database, f"{database}_test"):
                if not database_exists(conn, name):
                    continue
                print(f"  dropping existing database {name!r}")
                with conn.cursor() as cur:
                    # Evict any lingering sessions or the DROP will block.
                    cur.execute(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = %s AND pid <> pg_backend_pid()",
                        (name,),
                    )
                    cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))

        with conn.cursor() as cur:
            if role_exists(conn, role):
                print(f"  role {role!r} exists - updating its password")
                cur.execute(
                    sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                        sql.Identifier(role), sql.Literal(app_password)
                    )
                )
            else:
                print(f"  creating role {role!r} (LOGIN, no CREATEDB, no SUPERUSER)")
                cur.execute(
                    sql.SQL(
                        "CREATE ROLE {} WITH LOGIN PASSWORD {} "
                        "NOSUPERUSER NOCREATEDB NOCREATEROLE INHERIT"
                    ).format(sql.Identifier(role), sql.Literal(app_password))
                )

            for name in (database, f"{database}_test"):
                if database_exists(conn, name):
                    print(f"  database {name!r} already exists")
                    continue
                # The application role is deliberately NOCREATEDB, so the test
                # database is provisioned here rather than by the test suite.
                print(f"  creating database {name!r} owned by {role!r}")
                cur.execute(
                    sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'").format(
                        sql.Identifier(name), sql.Identifier(role)
                    )
                )

    # Lock down the public schema inside each database.
    for name in (database, f"{database}_test"):
        with psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_superuser,
            password=superuser_password,
            dbname=name,
            autocommit=True,
        ) as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(sql.Identifier(role))
            )
            cur.execute(
                sql.SQL("ALTER SCHEMA public OWNER TO {}").format(sql.Identifier(role))
            )
            # PostgreSQL 15+ already removes this, but be explicit.
            cur.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
            print(f"  granted schema ownership on {name!r} to {role!r}")


def write_env_file(app_password: str) -> Path:
    """Write or update the gitignored ``.env`` with the generated credentials."""
    env_path = REPO_ROOT / ".env"
    example = REPO_ROOT / ".env.example"

    lines = (
        env_path.read_text(encoding="utf-8").splitlines()
        if env_path.exists()
        else example.read_text(encoding="utf-8").splitlines()
    )

    updates = {
        "POSTGRES_PASSWORD": app_password,
        "SECRET_KEY": secrets.token_urlsafe(64),
        # Never persisted: the superuser password is not the app's business.
        "POSTGRES_SUPERUSER_PASSWORD": "",
    }
    # Preserve an existing SECRET_KEY so tokens issued earlier stay valid.
    for line in lines:
        if line.startswith("SECRET_KEY=") and len(line.split("=", 1)[1].strip()) > 20:
            if not line.split("=", 1)[1].strip().startswith("replace-me"):
                updates.pop("SECRET_KEY", None)

    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return env_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the Sentinel Finance role and database.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the database first. Destroys all data.",
    )
    parser.add_argument(
        "--app-password",
        default=None,
        help="Password for the application role. Generated if omitted.",
    )
    args = parser.parse_args()

    superuser_password = settings.postgres_superuser_password or getpass.getpass(
        f"Password for PostgreSQL superuser {settings.postgres_superuser!r}: "
    )
    if not superuser_password:
        print("A superuser password is required.", file=sys.stderr)
        return 1

    app_password = args.app_password or generate_password()

    print(f"Bootstrapping {settings.postgres_host}:{settings.postgres_port}")
    try:
        bootstrap(app_password, superuser_password, drop_existing=args.drop_existing)
    except psycopg.OperationalError as exc:
        print(f"\nCould not connect: {exc}", file=sys.stderr)
        print("Check the host, port and superuser password.", file=sys.stderr)
        return 1

    env_path = write_env_file(app_password)
    print()
    print(f"Wrote credentials to {env_path} (gitignored).")
    print("Next:")
    print("    alembic upgrade head")
    print("    python -m scripts.seed_demo_data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
