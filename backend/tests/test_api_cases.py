"""Case lifecycle, the investigation timeline and audit persistence.

Requires PostgreSQL; skipped cleanly when none is configured.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.db.enums import AuditAction, PaymentMethod, RoleCode, TransactionStatus
from tests.conftest import requires_db

pytestmark = requires_db


@pytest.fixture
def transaction(db_session, sample_customer):
    """One stored transaction, available as case evidence."""
    from app.db.models.transaction import Transaction

    customer, device, beneficiary = sample_customer
    record = Transaction(
        transaction_ref="TX-990001",
        customer_id=customer.id,
        amount=Decimal("125000.00"),
        currency="INR",
        occurred_at=datetime.now(UTC) - timedelta(hours=2),
        location_city="Mumbai",
        location_region="Maharashtra",
        latitude=19.07,
        longitude=72.87,
        device_id=device.id,
        beneficiary_id=beneficiary.id,
        payment_method=PaymentMethod.IMPS,
        channel="MOBILE_APP",
        status=TransactionStatus.FLAGGED,
        is_demo=True,
    )
    db_session.add(record)
    db_session.flush()
    return record


def audit_actions(db_session):
    from sqlalchemy import select

    from app.db.models.audit import AuditLog

    return [row[0] for row in db_session.execute(select(AuditLog.action)).all()]


class TestCaseCreation:
    def test_an_analyst_can_open_a_case(self, client, auth_headers, transaction):
        response = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={
                "title": "Unusual high-value transfer requires review",
                "summary": "Flagged by detection.",
                "priority": "HIGH",
                "transaction_ids": [transaction.id],
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["case_ref"].startswith("CASE-")
        assert body["status"] == "NEW"
        assert body["priority"] == "HIGH"
        assert len(body["transactions"]) == 1

    def test_the_case_is_persisted_and_readable(self, client, auth_headers, transaction):
        created = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Persisted investigation case", "transaction_ids": [transaction.id]},
        ).json()

        fetched = client.get(
            f"/api/cases/{created['case_ref']}", headers=auth_headers(RoleCode.ANALYST)
        )
        assert fetched.status_code == 200
        assert fetched.json()["title"] == "Persisted investigation case"

    def test_creation_writes_an_audit_entry(self, client, auth_headers, transaction, db_session):
        client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Audited case creation", "transaction_ids": [transaction.id]},
        )
        assert AuditAction.CASE_CREATED in audit_actions(db_session)

    def test_a_too_short_title_is_rejected(self, client, auth_headers):
        response = client.post(
            "/api/cases", headers=auth_headers(RoleCode.ANALYST), json={"title": "ab"}
        )
        assert response.status_code == 422

    def test_an_analyst_cannot_assign_on_creation(self, client, auth_headers, users):
        response = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={
                "title": "Analyst attempts to assign",
                "assigned_to": users[RoleCode.SENIOR_ANALYST].id,
            },
        )
        assert response.status_code == 403

    def test_linking_an_unknown_transaction_is_rejected(self, client, auth_headers):
        response = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Case with a bad evidence link", "transaction_ids": [999_999]},
        )
        assert response.status_code == 404


class TestCaseLifecycle:
    @pytest.fixture
    def case_ref(self, client, auth_headers, transaction) -> str:
        return client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Lifecycle test investigation", "transaction_ids": [transaction.id]},
        ).json()["case_ref"]

    def test_a_senior_analyst_can_assign(self, client, auth_headers, case_ref, users):
        response = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"assigned_to": users[RoleCode.ANALYST].id},
        )
        assert response.status_code == 200
        assert response.json()["assignee"]["id"] == users[RoleCode.ANALYST].id

    def test_an_analyst_cannot_assign(self, client, auth_headers, case_ref, users):
        response = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.ANALYST),
            json={"assigned_to": users[RoleCode.MANAGER].id},
        )
        assert response.status_code == 403

    def test_an_analyst_cannot_change_priority(self, client, auth_headers, case_ref):
        response = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.ANALYST),
            json={"priority": "CRITICAL"},
        )
        assert response.status_code == 403

    def test_an_analyst_cannot_resolve(self, client, auth_headers, case_ref):
        response = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.ANALYST),
            json={"status": "RESOLVED"},
        )
        assert response.status_code == 403

    def test_a_senior_analyst_can_resolve(self, client, auth_headers, case_ref):
        response = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"status": "RESOLVED"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "RESOLVED"
        assert response.json()["closed_at"] is not None

    def test_a_senior_analyst_cannot_close(self, client, auth_headers, case_ref):
        response = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"status": "CLOSED"},
        )
        assert response.status_code == 403

    def test_a_manager_can_close_and_reopen(self, client, auth_headers, case_ref):
        closed = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.MANAGER),
            json={"status": "CLOSED"},
        )
        assert closed.status_code == 200
        assert closed.json()["status"] == "CLOSED"

        reopened = client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.MANAGER),
            json={"status": "UNDER_REVIEW"},
        )
        assert reopened.status_code == 200
        assert reopened.json()["status"] == "UNDER_REVIEW"
        assert reopened.json()["closed_at"] is None

    def test_notes_cannot_be_added_to_a_closed_case(self, client, auth_headers, case_ref):
        client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.MANAGER),
            json={"status": "CLOSED"},
        )
        response = client.post(
            f"/api/cases/{case_ref}/notes",
            headers=auth_headers(RoleCode.ANALYST),
            json={"body": "A late note"},
        )
        assert response.status_code == 409


class TestNotesAndEvidence:
    @pytest.fixture
    def case_ref(self, client, auth_headers, transaction) -> str:
        return client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Notes and evidence investigation"},
        ).json()["case_ref"]

    def test_a_note_is_persisted_with_its_author(self, client, auth_headers, case_ref, users):
        response = client.post(
            f"/api/cases/{case_ref}/notes",
            headers=auth_headers(RoleCode.ANALYST),
            json={"body": "Checked the customer's baseline.", "evidence_ref": "TICKET-42"},
        )
        assert response.status_code == 201
        assert response.json()["body"] == "Checked the customer's baseline."
        assert response.json()["evidence_ref"] == "TICKET-42"
        assert response.json()["author"]["id"] == users[RoleCode.ANALYST].id

        detail = client.get(
            f"/api/cases/{case_ref}", headers=auth_headers(RoleCode.ANALYST)
        ).json()
        assert len(detail["notes"]) == 1

    def test_an_empty_note_is_rejected(self, client, auth_headers, case_ref):
        response = client.post(
            f"/api/cases/{case_ref}/notes",
            headers=auth_headers(RoleCode.ANALYST),
            json={"body": ""},
        )
        assert response.status_code == 422

    def test_evidence_can_be_linked_and_unlinked(
        self, client, auth_headers, case_ref, transaction
    ):
        linked = client.post(
            f"/api/cases/{case_ref}/transactions",
            headers=auth_headers(RoleCode.ANALYST),
            json={"transaction_ids": [transaction.id]},
        )
        assert linked.status_code == 200
        assert len(linked.json()["transactions"]) == 1

        removed = client.delete(
            f"/api/cases/{case_ref}/transactions/{transaction.id}",
            headers=auth_headers(RoleCode.ANALYST),
        )
        assert removed.status_code == 200

        detail = client.get(
            f"/api/cases/{case_ref}", headers=auth_headers(RoleCode.ANALYST)
        ).json()
        assert detail["transactions"] == []

    def test_linking_the_same_transaction_twice_is_idempotent(
        self, client, auth_headers, case_ref, transaction
    ):
        for _ in range(2):
            client.post(
                f"/api/cases/{case_ref}/transactions",
                headers=auth_headers(RoleCode.ANALYST),
                json={"transaction_ids": [transaction.id]},
            )
        detail = client.get(
            f"/api/cases/{case_ref}", headers=auth_headers(RoleCode.ANALYST)
        ).json()
        assert len(detail["transactions"]) == 1


class TestInvestigationTimeline:
    def test_every_action_appears_on_the_timeline(
        self, client, auth_headers, transaction, users
    ):
        """The timeline is a read of stored events, never reconstructed."""
        case_ref = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Full lifecycle timeline", "transaction_ids": [transaction.id]},
        ).json()["case_ref"]

        client.post(
            f"/api/cases/{case_ref}/notes",
            headers=auth_headers(RoleCode.ANALYST),
            json={"body": "Opened and reviewed the detection factors."},
        )
        client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"assigned_to": users[RoleCode.SENIOR_ANALYST].id},
        )
        client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"priority": "CRITICAL"},
        )
        client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"status": "RESOLVED"},
        )

        timeline = client.get(
            f"/api/cases/{case_ref}/timeline", headers=auth_headers(RoleCode.ANALYST)
        ).json()
        recorded = {event["event_type"] for event in timeline}
        assert {
            "CREATED",
            "TRANSACTION_LINKED",
            "NOTE_ADDED",
            "ASSIGNED",
            "PRIORITY_CHANGED",
            "CLOSED",
        } <= recorded

        # Ordered oldest-first, and each entry names who did it.
        timestamps = [event["occurred_at"] for event in timeline]
        assert timestamps == sorted(timestamps)
        assert all(event["description"] for event in timeline)

    def test_status_transitions_record_both_sides(self, client, auth_headers):
        case_ref = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Status transition recording"},
        ).json()["case_ref"]
        client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.SENIOR_ANALYST),
            json={"status": "ESCALATED"},
        )
        timeline = client.get(
            f"/api/cases/{case_ref}/timeline", headers=auth_headers(RoleCode.ANALYST)
        ).json()
        change = next(e for e in timeline if e["event_type"] == "STATUS_CHANGED")
        assert change["from_value"] == "NEW"
        assert change["to_value"] == "ESCALATED"


class TestAuditTrail:
    def test_case_mutations_are_audited(self, client, auth_headers, db_session):
        case_ref = client.post(
            "/api/cases",
            headers=auth_headers(RoleCode.ANALYST),
            json={"title": "Audited case mutations"},
        ).json()["case_ref"]
        client.post(
            f"/api/cases/{case_ref}/notes",
            headers=auth_headers(RoleCode.ANALYST),
            json={"body": "A note that should be audited."},
        )
        client.patch(
            f"/api/cases/{case_ref}",
            headers=auth_headers(RoleCode.MANAGER),
            json={"status": "CLOSED"},
        )
        actions = audit_actions(db_session)
        assert AuditAction.CASE_CREATED in actions
        assert AuditAction.CASE_NOTE_ADDED in actions
        assert AuditAction.CASE_CLOSED in actions

    def test_a_manager_can_read_the_audit_log(self, client, auth_headers):
        response = client.get("/api/audit-logs", headers=auth_headers(RoleCode.MANAGER))
        assert response.status_code == 200
        assert "items" in response.json()

    def test_the_audit_api_exposes_no_write_operations(self, client, auth_headers):
        """Append-only means append-only: there is nothing to call."""
        headers = auth_headers(RoleCode.ADMIN)
        assert client.post("/api/audit-logs", headers=headers, json={}).status_code in (
            404,
            405,
        )
        assert client.delete("/api/audit-logs/1", headers=headers).status_code in (404, 405)


class TestTransactionViewing:
    def test_opening_a_transaction_records_who_looked(
        self, client, auth_headers, transaction, db_session
    ):
        response = client.get(
            f"/api/transactions/{transaction.transaction_ref}",
            headers=auth_headers(RoleCode.ANALYST),
        )
        assert response.status_code == 200
        assert AuditAction.TRANSACTION_VIEWED in audit_actions(db_session)

    def test_customer_details_are_masked_in_the_response(
        self, client, auth_headers, transaction, sample_customer
    ):
        customer, _, _ = sample_customer
        response = client.get(
            f"/api/transactions/{transaction.transaction_ref}",
            headers=auth_headers(RoleCode.ANALYST),
        )
        body = response.text
        assert customer.phone not in body
        assert customer.account_number not in body
        assert customer.full_name not in body

    def test_an_unknown_reference_returns_a_clean_404(self, client, auth_headers):
        response = client.get(
            "/api/transactions/TX-DOES-NOT-EXIST", headers=auth_headers(RoleCode.ANALYST)
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"
        assert "Traceback" not in response.text
