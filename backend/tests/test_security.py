"""Password hashing, tokens, rate limiting, masking, logging redaction and RBAC.

These cover the parts of the security posture that can be checked without a
database or a live request.
"""

from __future__ import annotations

import pytest

from app.auth.permissions import (
    Permission,
    has_permission,
    permissions_for,
    role_at_least,
)
from app.core.logging import redact
from app.core.rate_limit import SlidingWindowLimiter, login_rate_limit_key
from app.core.security import (
    MAX_PASSWORD_BYTES,
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.enums import RoleCode
from app.services.audit_service import SENSITIVE_KEYS, scrub
from app.services.masking import (
    mask_account_number,
    mask_device_ref,
    mask_email,
    mask_name,
    mask_phone,
)


class TestPasswordHashing:
    def test_hash_verifies_against_the_original(self):
        digest = hash_password("Str0ng-Passw0rd!")
        assert verify_password("Str0ng-Passw0rd!", digest)

    def test_wrong_password_is_rejected(self):
        assert not verify_password("wrong", hash_password("Str0ng-Passw0rd!"))

    def test_plaintext_never_appears_in_the_digest(self):
        password = "Str0ng-Passw0rd!"
        assert password not in hash_password(password)

    def test_salting_makes_identical_passwords_hash_differently(self):
        assert hash_password("same-password-1") != hash_password("same-password-1")

    def test_overlong_passwords_are_rejected_rather_than_silently_truncated(self):
        """bcrypt truncates at 72 bytes; two long passwords must not collide."""
        with pytest.raises(ValueError):
            hash_password("a" * (MAX_PASSWORD_BYTES + 1))

    def test_malformed_hash_fails_closed(self):
        assert verify_password("anything", "not-a-valid-hash") is False

    def test_empty_hash_fails_closed(self):
        assert verify_password("anything", "") is False


class TestTokens:
    def test_access_token_round_trip(self):
        token, expires_in = create_access_token(7, "MANAGER")
        claims = decode_token(token, "access")
        assert claims["sub"] == "7"
        assert claims["role"] == "MANAGER"
        assert claims["iss"] == "sentinel-finance"
        assert expires_in > 0

    def test_refresh_token_round_trip(self):
        token, max_age = create_refresh_token(7)
        assert decode_token(token, "refresh")["sub"] == "7"
        assert max_age > 0

    def test_remember_me_extends_only_the_refresh_lifetime(self):
        _, standard = create_refresh_token(1, remember_me=False)
        _, extended = create_refresh_token(1, remember_me=True)
        assert extended > standard

    def test_a_refresh_token_cannot_be_replayed_as_an_access_token(self):
        """Token type confusion would turn a long-lived cookie into API access."""
        refresh, _ = create_refresh_token(1)
        with pytest.raises(TokenError):
            decode_token(refresh, "access")

    def test_an_access_token_is_not_accepted_for_refresh(self):
        access, _ = create_access_token(1, "ANALYST")
        with pytest.raises(TokenError):
            decode_token(access, "refresh")

    def test_tampered_tokens_are_rejected(self):
        token, _ = create_access_token(1, "ANALYST")
        with pytest.raises(TokenError):
            decode_token(token + "x", "access")

    def test_garbage_is_rejected(self):
        with pytest.raises(TokenError):
            decode_token("not.a.token", "access")

    def test_a_token_signed_with_another_key_is_rejected(self):
        import jwt

        forged = jwt.encode(
            {"sub": "1", "type": "access", "iss": "sentinel-finance", "exp": 9_999_999_999,
             "iat": 1},
            "an-attackers-key",
            algorithm="HS256",
        )
        with pytest.raises(TokenError):
            decode_token(forged, "access")

    def test_an_unsigned_token_is_rejected(self):
        """The classic `alg: none` downgrade must not be accepted."""
        import base64
        import json

        def b64(payload: dict) -> str:
            raw = json.dumps(payload).encode()
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        forged = f'{b64({"alg": "none", "typ": "JWT"})}.{b64({"sub": "1", "type": "access"})}.'
        with pytest.raises(TokenError):
            decode_token(forged, "access")


class TestRateLimiting:
    def test_requests_are_allowed_below_the_limit(self):
        limiter = SlidingWindowLimiter(max_attempts=3, window_seconds=60)
        limiter.record("key")
        limiter.record("key")
        assert limiter.check("key")[0] is True

    def test_the_limit_blocks_further_attempts(self):
        limiter = SlidingWindowLimiter(max_attempts=3, window_seconds=60)
        for _ in range(3):
            limiter.record("key")
        allowed, retry_after = limiter.check("key")
        assert allowed is False
        assert retry_after > 0

    def test_a_successful_login_clears_the_counter(self):
        limiter = SlidingWindowLimiter(max_attempts=2, window_seconds=60)
        limiter.record("key")
        limiter.record("key")
        limiter.reset("key")
        assert limiter.check("key")[0] is True

    def test_buckets_are_independent(self):
        limiter = SlidingWindowLimiter(max_attempts=1, window_seconds=60)
        limiter.record("a")
        assert limiter.check("b")[0] is True

    def test_key_combines_address_and_account(self):
        """Keying on both stops an attacker locking out a real user by spraying."""
        assert login_rate_limit_key("1.2.3.4", "A@B.com") == "1.2.3.4|a@b.com"
        assert login_rate_limit_key("1.2.3.4", "a@b.com") != login_rate_limit_key(
            "9.9.9.9", "a@b.com"
        )


class TestMasking:
    def test_phone_keeps_only_the_leading_digits(self):
        assert mask_phone("9876543210") == "9876******"

    def test_account_number_keeps_only_the_last_four(self):
        assert mask_account_number("123456784821") == "XXXX XXXX 4821"

    def test_email_is_masked_but_recognisable(self):
        masked = mask_email("priya.sharma@example.com")
        assert masked.startswith("p") and masked.endswith(".com")
        assert "priya.sharma" not in masked
        assert "example" not in masked

    def test_name_keeps_a_usable_handle(self):
        assert mask_name("Priya Sharma") == "Priya S."

    def test_device_reference_keeps_only_a_tail(self):
        assert mask_device_ref("DEV-8fa31c9e4b") == "DEV-****1c9e4b"

    @pytest.mark.parametrize(
        "masker", [mask_phone, mask_account_number, mask_email, mask_name, mask_device_ref]
    )
    def test_maskers_tolerate_missing_values(self, masker):
        assert masker(None) is None
        assert masker("") == ""

    def test_short_values_do_not_leak(self):
        assert "1" not in mask_phone("123")


class TestAuditScrubbing:
    def test_credential_keys_are_redacted(self):
        scrubbed = scrub({"email": "a@b.com", "password": "secret", "password_hash": "$2b$..."})
        assert scrubbed["password"] == "[REDACTED]"
        assert scrubbed["password_hash"] == "[REDACTED]"
        assert scrubbed["email"] == "a@b.com"

    def test_every_declared_sensitive_key_is_covered(self):
        scrubbed = scrub({key: "value" for key in SENSITIVE_KEYS})
        assert all(value == "[REDACTED]" for value in scrubbed.values())

    def test_long_values_are_truncated(self):
        scrubbed = scrub({"note": "x" * 5000})
        assert len(scrubbed["note"]) < 1000

    def test_none_passes_through(self):
        assert scrub(None) is None


class TestLogRedaction:
    def test_password_pairs_are_masked(self):
        assert "hunter2" not in redact("login password=hunter2 ok")

    def test_bearer_tokens_are_masked(self):
        assert "eyJhbGciOi" not in redact("Authorization: Bearer eyJhbGciOi.abc.def")

    def test_bcrypt_digests_are_masked(self):
        digest = hash_password("some-password")
        assert digest not in redact(f"stored {digest} for user")

    def test_ordinary_text_is_untouched(self):
        assert redact("Scored 40000 transactions") == "Scored 40000 transactions"


class TestPermissionMatrix:
    def test_privileges_accumulate_up_the_hierarchy(self):
        analyst = permissions_for(RoleCode.ANALYST)
        senior = permissions_for(RoleCode.SENIOR_ANALYST)
        manager = permissions_for(RoleCode.MANAGER)
        admin = permissions_for(RoleCode.ADMIN)
        assert analyst < senior < manager < admin

    @pytest.mark.parametrize(
        ("role", "permission", "expected"),
        [
            (RoleCode.ANALYST, Permission.VIEW_TRANSACTIONS, True),
            (RoleCode.ANALYST, Permission.CREATE_CASE, True),
            (RoleCode.ANALYST, Permission.ASSIGN_CASE, False),
            (RoleCode.ANALYST, Permission.CLOSE_CASE, False),
            (RoleCode.ANALYST, Permission.VIEW_AUDIT_LOGS, False),
            (RoleCode.ANALYST, Permission.MANAGE_USERS, False),
            (RoleCode.SENIOR_ANALYST, Permission.ASSIGN_CASE, True),
            (RoleCode.SENIOR_ANALYST, Permission.RESOLVE_CASE, True),
            (RoleCode.SENIOR_ANALYST, Permission.CLOSE_CASE, False),
            (RoleCode.SENIOR_ANALYST, Permission.MANAGE_USERS, False),
            (RoleCode.MANAGER, Permission.CLOSE_CASE, True),
            (RoleCode.MANAGER, Permission.VIEW_AUDIT_LOGS, True),
            (RoleCode.MANAGER, Permission.RUN_DETECTION, True),
            (RoleCode.MANAGER, Permission.MANAGE_USERS, False),
            (RoleCode.MANAGER, Permission.TRAIN_MODEL, False),
            (RoleCode.ADMIN, Permission.MANAGE_USERS, True),
            (RoleCode.ADMIN, Permission.TRAIN_MODEL, True),
            (RoleCode.ADMIN, Permission.IMPORT_DATA, True),
        ],
    )
    def test_matrix(self, role, permission, expected):
        assert has_permission(role, permission) is expected

    def test_only_administrators_manage_users(self):
        for role in RoleCode:
            expected = role is RoleCode.ADMIN
            assert has_permission(role, Permission.MANAGE_USERS) is expected
            assert has_permission(role, Permission.IMPORT_DATA) is expected

    def test_role_ordering(self):
        assert role_at_least(RoleCode.MANAGER, RoleCode.ANALYST)
        assert role_at_least(RoleCode.ANALYST, RoleCode.ANALYST)
        assert not role_at_least(RoleCode.ANALYST, RoleCode.MANAGER)

    def test_every_role_can_view_the_dashboard(self):
        for role in RoleCode:
            assert has_permission(role, Permission.VIEW_DASHBOARD)
