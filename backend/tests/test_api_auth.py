"""Authentication and authorisation over the live API.

Requires PostgreSQL; skipped cleanly when none is configured.
"""

from __future__ import annotations

import pytest

from app.db.enums import AuditAction, RoleCode
from tests.conftest import TEST_PASSWORD, requires_db

pytestmark = requires_db


class TestLogin:
    def test_valid_credentials_return_a_token_and_the_user(self, client, users):
        response = client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": TEST_PASSWORD},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["access_token"]
        assert body["token_type"] == "bearer"
        assert body["user"]["email"] == users[RoleCode.ANALYST].email
        assert body["user"]["role"]["code"] == "ANALYST"
        assert "VIEW_TRANSACTIONS" in body["user"]["permissions"]

    def test_password_material_never_appears_in_the_response(self, client, users):
        response = client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ADMIN].email, "password": TEST_PASSWORD},
        )
        assert "password" not in response.text.lower()
        assert "$2b$" not in response.text

    def test_a_refresh_cookie_is_set_httponly(self, client, users):
        """Script must not be able to read the durable session token."""
        response = client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": TEST_PASSWORD},
        )
        cookie_header = response.headers.get("set-cookie", "")
        assert "sentinel_refresh" in cookie_header
        assert "httponly" in cookie_header.lower()

    def test_wrong_password_is_rejected(self, client, users):
        response = client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": "wrong-password"},
        )
        assert response.status_code == 401

    def test_unknown_and_wrong_password_give_the_same_message(self, client, users):
        """Otherwise the endpoint becomes an account enumeration oracle."""
        unknown = client.post(
            "/api/auth/login",
            json={"email": "nobody@sentineltest.com", "password": TEST_PASSWORD},
        )
        wrong = client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": "wrong-password"},
        )
        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json()["error"]["message"] == wrong.json()["error"]["message"]

    def test_deactivated_accounts_cannot_sign_in(self, client, users, db_session):
        user = users[RoleCode.ANALYST]
        user.is_active = False
        db_session.flush()
        response = client.post(
            "/api/auth/login", json={"email": user.email, "password": TEST_PASSWORD}
        )
        assert response.status_code == 401
        assert "deactivated" in response.json()["error"]["message"].lower()

    def test_malformed_email_is_a_validation_error(self, client):
        response = client.post(
            "/api/auth/login", json={"email": "not-an-email", "password": "whatever"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_submitted_values_are_never_echoed_back(self, client):
        response = client.post(
            "/api/auth/login", json={"email": "leak-me@sentineltest.com", "password": "s3cret-value"}
        )
        assert "s3cret-value" not in response.text
        assert "leak-me@sentineltest.com" not in response.text

    def test_repeated_failures_are_rate_limited(self, client, users):
        from app.core.config import settings

        email = users[RoleCode.ANALYST].email
        for _ in range(settings.login_rate_limit_attempts):
            client.post("/api/auth/login", json={"email": email, "password": "wrong"})

        blocked = client.post("/api/auth/login", json={"email": email, "password": "wrong"})
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "RATE_LIMITED"

        # A correct password is still refused while the window is open.
        assert (
            client.post(
                "/api/auth/login", json={"email": email, "password": TEST_PASSWORD}
            ).status_code
            == 429
        )


class TestSession:
    def test_me_returns_the_signed_in_user(self, client, users, auth_headers):
        response = client.get("/api/auth/me", headers=auth_headers(RoleCode.MANAGER))
        assert response.status_code == 200
        assert response.json()["role"]["code"] == "MANAGER"

    def test_me_requires_a_token(self, client):
        assert client.get("/api/auth/me").status_code == 401

    def test_a_garbage_token_is_rejected(self, client):
        response = client.get(
            "/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
        )
        assert response.status_code == 401

    def test_a_refresh_token_cannot_be_used_as_a_bearer_token(self, client, users):
        from app.core.security import create_refresh_token

        refresh, _ = create_refresh_token(users[RoleCode.ADMIN].id)
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {refresh}"})
        assert response.status_code == 401

    def test_a_token_for_a_deactivated_user_stops_working_immediately(
        self, client, users, auth_headers, db_session
    ):
        """Deactivation must take effect at once, not when the token expires."""
        headers = auth_headers(RoleCode.ANALYST)
        assert client.get("/api/auth/me", headers=headers).status_code == 200

        users[RoleCode.ANALYST].is_active = False
        db_session.flush()
        assert client.get("/api/auth/me", headers=headers).status_code == 401

    def test_logout_succeeds_and_clears_the_cookie(self, client, auth_headers):
        response = client.post("/api/auth/logout", headers=auth_headers(RoleCode.ANALYST))
        assert response.status_code == 200
        assert "sentinel_refresh" in response.headers.get("set-cookie", "")

    def test_refresh_without_a_cookie_is_rejected(self, client):
        assert client.post("/api/auth/refresh").status_code == 401


class TestLoginAuditing:
    def _actions(self, db_session):
        from sqlalchemy import select

        from app.db.models.audit import AuditLog

        return [row[0] for row in db_session.execute(select(AuditLog.action)).all()]

    def test_successful_login_is_recorded(self, client, users, db_session):
        client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": TEST_PASSWORD},
        )
        assert AuditAction.LOGIN in self._actions(db_session)

    def test_failed_login_is_recorded(self, client, users, db_session):
        client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": "wrong"},
        )
        assert AuditAction.LOGIN_FAILED in self._actions(db_session)

    def test_no_password_material_reaches_the_audit_trail(self, client, users, db_session):
        from sqlalchemy import select

        from app.db.models.audit import AuditLog

        client.post(
            "/api/auth/login",
            json={"email": users[RoleCode.ANALYST].email, "password": "a-secret-password"},
        )
        for entry in db_session.execute(select(AuditLog)).scalars():
            blob = f"{entry.description}{entry.previous_state}{entry.new_state}"
            assert "a-secret-password" not in blob
            assert "$2b$" not in blob


class TestRbacEnforcement:
    """The server is the authority; hiding UI is only a courtesy."""

    @pytest.mark.parametrize(
        ("method", "path", "allowed_roles"),
        [
            ("GET", "/api/audit-logs", {RoleCode.MANAGER, RoleCode.ADMIN}),
            ("GET", "/api/users", {RoleCode.ADMIN}),
            ("POST", "/api/detection/run", {RoleCode.MANAGER, RoleCode.ADMIN}),
            ("POST", "/api/model/train", {RoleCode.ADMIN}),
            ("GET", "/api/imports", {RoleCode.ADMIN}),
            ("GET", "/api/reports/risk-summary", {RoleCode.MANAGER, RoleCode.ADMIN}),
        ],
    )
    def test_privileged_endpoints_reject_lower_roles(
        self, client, auth_headers, method, path, allowed_roles
    ):
        for role in RoleCode:
            response = client.request(method, path, headers=auth_headers(role))
            if role in allowed_roles:
                assert response.status_code != 403, f"{role.value} should reach {path}"
            else:
                assert response.status_code == 403, (
                    f"{role.value} must not reach {method} {path} "
                    f"(got {response.status_code})"
                )

    @pytest.mark.parametrize(
        "path",
        [
            "/api/dashboard/summary",
            "/api/transactions",
            "/api/customers",
            "/api/alerts",
            "/api/cases",
            "/api/analytics",
            "/api/model/status",
            "/api/settings",
        ],
    )
    def test_every_role_can_reach_the_shared_read_endpoints(
        self, client, auth_headers, path
    ):
        for role in RoleCode:
            response = client.get(path, headers=auth_headers(role))
            assert response.status_code == 200, f"{role.value} blocked from {path}"

    def test_permission_list_matches_what_the_api_enforces(self, client, auth_headers):
        """The UI trusts this list; it must not overstate what a role can do."""
        response = client.get("/api/auth/me", headers=auth_headers(RoleCode.ANALYST))
        permissions = response.json()["permissions"]
        assert "MANAGE_USERS" not in permissions
        assert "VIEW_AUDIT_LOGS" not in permissions
        assert client.get(
            "/api/users", headers=auth_headers(RoleCode.ANALYST)
        ).status_code == 403
