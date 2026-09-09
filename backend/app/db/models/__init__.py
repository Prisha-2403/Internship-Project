"""ORM model registry.

Importing this package registers every mapper on ``Base.metadata``, which is
what Alembic autogenerate and ``create_all`` rely on. Import models from here
so no module can accidentally see a half-populated metadata.
"""

from app.db.base import Base
from app.db.models.alert import Alert
from app.db.models.audit import AuditLog
from app.db.models.case import Case, CaseEvent, CaseNote, CaseTransaction
from app.db.models.customer import Beneficiary, Customer, Device
from app.db.models.ops import ImportJob, ModelVersion
from app.db.models.transaction import RiskScore, Transaction, TransactionFeature
from app.db.models.user import Role, User

__all__ = [
    "Base",
    "Role",
    "User",
    "Customer",
    "Device",
    "Beneficiary",
    "Transaction",
    "TransactionFeature",
    "RiskScore",
    "Alert",
    "Case",
    "CaseTransaction",
    "CaseNote",
    "CaseEvent",
    "AuditLog",
    "ModelVersion",
    "ImportJob",
]
