"""Synthetic transaction generator.

Everything produced here is fabricated. No real person, account, device or
payment is represented, and none of the names, phone numbers or account numbers
correspond to anything outside this database. Rows are tagged ``is_demo=True``
and the UI carries a persistent DEMO / SYNTHETIC DATA banner.

The generator gives each customer a stable behavioural profile - a typical
spend, a daily pace, a set of active hours, a home city, known devices and known
payees - and then draws ordinary activity from it. Six anomaly scenarios are
injected on top of that baseline so the detection stack has something real to
find.

The injected scenario is recorded on each transaction as demo bookkeeping only.
It is a record of what was generated, *not* a fraud label, and nothing in the
application treats it as ground truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.db.enums import AnomalyScenario, PaymentMethod

# --- Reference data (entirely fictional) ----------------------------------
CITIES: tuple[tuple[str, str, float, float], ...] = (
    ("Delhi", "Delhi NCR", 28.6139, 77.2090),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777),
    ("Bangalore", "Karnataka", 12.9716, 77.5946),
    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Pune", "Maharashtra", 18.5204, 73.8567),
    ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Surat", "Gujarat", 21.1702, 72.8311),
)

# Weighted so the dashboard's location chart has a realistic long tail.
CITY_WEIGHTS = (22, 20, 16, 11, 9, 8, 5, 4, 3, 2)

FIRST_NAMES = (
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna",
    "Ishaan", "Rudra", "Ananya", "Diya", "Priya", "Aadhya", "Kavya", "Anika",
    "Navya", "Riya", "Meera", "Saanvi", "Rohan", "Kabir", "Nikhil", "Rahul",
    "Neha", "Pooja", "Sneha", "Divya", "Farhan", "Zoya", "Imran", "Tanvi",
)
LAST_NAMES = (
    "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Menon", "Kapoor",
    "Chopra", "Malhotra", "Bose", "Ghosh", "Rao", "Joshi", "Desai", "Kulkarni",
    "Banerjee", "Chatterjee", "Khan", "Sheikh", "Gupta", "Agarwal", "Singh",
)
BANKS = (
    "Meridian Bank", "Northgate Financial", "Crestline Bank", "Silverpine Bank",
    "Anchor Commercial", "Bluestone Bank", "Ironwood Savings",
)
BENEFICIARY_KINDS = (
    "Rentals", "Utilities", "Traders", "Services", "Enterprises", "Supplies",
    "Logistics", "Solutions", "Retail", "Consulting",
)
DEVICE_TYPES = (("MOBILE", "Android"), ("MOBILE", "iOS"), ("DESKTOP", "Windows"), ("TABLET", "iPadOS"))
CHANNELS = ("MOBILE_APP", "WEB", "BRANCH", "ATM")
SEGMENTS = ("RETAIL", "RETAIL", "RETAIL", "PREMIUM", "BUSINESS")

PAYMENT_METHODS = tuple(PaymentMethod)
PAYMENT_WEIGHTS = (34, 22, 14, 12, 9, 4, 5)  # UPI-heavy, matching the reference data

#: Weekend activity relative to a weekday.
WEEKEND_FACTOR = 0.65


@dataclass
class GeneratedDevice:
    device_ref: str
    device_type: str
    operating_system: str
    first_seen_at: datetime
    last_seen_at: datetime
    is_trusted: bool
    usage_count: int = 0


@dataclass
class GeneratedBeneficiary:
    beneficiary_ref: str
    display_name: str
    account_number: str
    bank_name: str
    first_seen_at: datetime
    payment_count: int = 0


@dataclass
class GeneratedTransaction:
    transaction_ref: str
    customer_index: int
    amount: float
    occurred_at: datetime
    location_city: str
    location_region: str
    latitude: float
    longitude: float
    device_ref: str | None
    beneficiary_ref: str | None
    payment_method: PaymentMethod
    channel: str
    injected_scenario: AnomalyScenario | None = None


@dataclass
class GeneratedCustomer:
    customer_ref: str
    full_name: str
    email: str
    phone: str
    account_number: str
    home_city: str
    home_region: str
    home_latitude: float
    home_longitude: float
    segment: str
    kyc_level: str
    onboarded_at: datetime
    devices: list[GeneratedDevice] = field(default_factory=list)
    beneficiaries: list[GeneratedBeneficiary] = field(default_factory=list)
    # Behavioural profile driving this customer's ordinary activity.
    typical_amount: float = 8500.0
    amount_sigma: float = 0.45
    daily_rate: float = 2.4
    active_hours: tuple[int, int] = (9, 20)


@dataclass
class GeneratedDataset:
    customers: list[GeneratedCustomer]
    transactions: list[GeneratedTransaction]
    scenario_counts: dict[str, int]

    @property
    def anomaly_count(self) -> int:
        return sum(self.scenario_counts.values())


class SyntheticDataGenerator:
    """Produces a coherent, entirely fabricated transaction history."""

    def __init__(
        self,
        *,
        customer_count: int = 120,
        target_transactions: int = 40_000,
        history_days: int = 90,
        seed: int = 42,
        anomaly_customer_share: float = 0.30,
        scenarios_per_customer: tuple[int, int] = (2, 4),
    ) -> None:
        self.customer_count = customer_count
        self.target_transactions = target_transactions
        self.history_days = history_days
        self.anomaly_customer_share = anomaly_customer_share
        self.scenarios_per_customer = scenarios_per_customer
        self.random = random.Random(seed)
        self.end_at = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        self.start_at = self.end_at - timedelta(days=history_days)
        self._txn_counter = 0

    # --- Customers --------------------------------------------------------
    def _make_customer(self, index: int) -> GeneratedCustomer:
        rnd = self.random
        city, region, lat, lon = rnd.choices(CITIES, weights=CITY_WEIGHTS, k=1)[0]
        first, last = rnd.choice(FIRST_NAMES), rnd.choice(LAST_NAMES)
        ref = f"CUST-{1000 + index}"
        segment = rnd.choice(SEGMENTS)

        # Premium and business accounts move more money, more often.
        base_amount = {"RETAIL": 8500.0, "PREMIUM": 26000.0, "BUSINESS": 48000.0}[segment]
        typical = base_amount * rnd.uniform(0.55, 1.7)
        daily_rate = {"RETAIL": 2.2, "PREMIUM": 3.1, "BUSINESS": 4.4}[segment] * rnd.uniform(
            0.6, 1.5
        )
        start_hour = rnd.randint(7, 11)
        end_hour = rnd.randint(18, 22)
        onboarded = self.start_at - timedelta(days=rnd.randint(120, 1500))

        customer = GeneratedCustomer(
            customer_ref=ref,
            full_name=f"{first} {last}",
            email=f"{first.lower()}.{last.lower()}{index}@example.invalid",
            phone=f"9{rnd.randint(100000000, 999999999)}",
            account_number=f"{rnd.randint(10**11, 10**12 - 1)}",
            home_city=city,
            home_region=region,
            home_latitude=lat,
            home_longitude=lon,
            segment=segment,
            kyc_level=rnd.choice(("FULL", "FULL", "FULL", "MINIMAL")),
            onboarded_at=onboarded,
            typical_amount=typical,
            amount_sigma=rnd.uniform(0.30, 0.60),
            daily_rate=daily_rate,
            active_hours=(start_hour, end_hour),
        )

        for d in range(rnd.randint(1, 3)):
            device_type, os_name = rnd.choice(DEVICE_TYPES)
            customer.devices.append(
                GeneratedDevice(
                    device_ref=f"DEV-{rnd.getrandbits(40):010x}",
                    device_type=device_type,
                    operating_system=os_name,
                    first_seen_at=onboarded + timedelta(days=d * rnd.randint(5, 60)),
                    last_seen_at=self.end_at,
                    is_trusted=d == 0,
                )
            )

        for _ in range(rnd.randint(2, 7)):
            kind = rnd.choice(BENEFICIARY_KINDS)
            name = f"{rnd.choice(LAST_NAMES)} {kind}"
            customer.beneficiaries.append(
                GeneratedBeneficiary(
                    beneficiary_ref=f"BEN-{rnd.getrandbits(36):09x}",
                    display_name=name,
                    account_number=f"{rnd.randint(10**11, 10**12 - 1)}",
                    bank_name=rnd.choice(BANKS),
                    first_seen_at=onboarded + timedelta(days=rnd.randint(1, 200)),
                )
            )
        return customer

    # --- Transactions -----------------------------------------------------
    def _next_ref(self) -> str:
        self._txn_counter += 1
        return f"TX-{80000 + self._txn_counter}"

    def _normal_amount(self, customer: GeneratedCustomer) -> float:
        """Log-normal draw: mostly modest payments with a plausible upper tail."""
        value = self.random.lognormvariate(math.log(customer.typical_amount), customer.amount_sigma)
        return round(min(max(value, 50.0), 2_500_000.0), 2)

    def _normal_time(self, customer: GeneratedCustomer, day: datetime) -> datetime:
        low, high = customer.active_hours
        hour = self.random.randint(low, high)
        return day.replace(
            hour=hour,
            minute=self.random.randint(0, 59),
            second=self.random.randint(0, 59),
            microsecond=0,
        )

    def _normal_transaction(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> GeneratedTransaction:
        rnd = self.random
        # Established customers overwhelmingly transact from home on a known device.
        device = rnd.choice(customer.devices[:1] * 4 + customer.devices)
        beneficiary = rnd.choice(customer.beneficiaries)
        return GeneratedTransaction(
            transaction_ref=self._next_ref(),
            customer_index=index,
            amount=self._normal_amount(customer),
            occurred_at=when,
            location_city=customer.home_city,
            location_region=customer.home_region,
            latitude=customer.home_latitude + rnd.uniform(-0.05, 0.05),
            longitude=customer.home_longitude + rnd.uniform(-0.05, 0.05),
            device_ref=device.device_ref,
            beneficiary_ref=beneficiary.beneficiary_ref,
            payment_method=rnd.choices(PAYMENT_METHODS, weights=PAYMENT_WEIGHTS, k=1)[0],
            channel=rnd.choices(CHANNELS, weights=(62, 26, 6, 6), k=1)[0],
        )

    # --- Anomaly scenarios ------------------------------------------------
    #
    # Each scenario drives its own primary signal hard, and then - with some
    # probability - a correlated secondary one. That correlation is the point:
    # genuinely suspicious activity rarely trips a single indicator in
    # isolation. An account takeover shows a new device *and* an odd hour; funds
    # being moved out show a large amount *and* an unfamiliar payee.
    #
    # Scenarios that trip one signal land in MEDIUM (requires review) and ones
    # that stack several reach HIGH or CRITICAL. That spread is deliberate: it
    # is what the scoring engine should do, not a dial turned to make the demo
    # look impressive.

    def _new_device_ref(self) -> str:
        return f"DEV-{self.random.getrandbits(40):010x}"

    def _new_beneficiary_ref(self) -> str:
        return f"BEN-{self.random.getrandbits(36):09x}"

    def _send_to_new_payee(self, txn: GeneratedTransaction) -> None:
        txn.beneficiary_ref = self._new_beneficiary_ref()

    def _move_to_far_city(self, customer: GeneratedCustomer, txn: GeneratedTransaction) -> None:
        rnd = self.random
        options = [c for c in CITIES if c[0] != customer.home_city]
        city, region, lat, lon = rnd.choice(options)
        txn.location_city, txn.location_region = city, region
        txn.latitude = lat + rnd.uniform(-0.05, 0.05)
        txn.longitude = lon + rnd.uniform(-0.05, 0.05)

    def _move_to_small_hours(self, txn: GeneratedTransaction) -> None:
        txn.occurred_at = txn.occurred_at.replace(
            hour=self.random.choice((1, 2, 3, 4)),
            minute=self.random.randint(0, 59),
            second=0,
            microsecond=0,
        )

    def _scenario_amount_spike(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> list[GeneratedTransaction]:
        """Scenario 1 - a transaction far above the customer's norm.

        Large sums usually leave for somewhere new, so most of these also go to
        an unfamiliar payee.
        """
        rnd = self.random
        txn = self._normal_transaction(customer, index, when)
        txn.amount = round(customer.typical_amount * rnd.uniform(14, 40), 2)
        if rnd.random() < 0.75:
            self._send_to_new_payee(txn)
        if rnd.random() < 0.35:
            self._move_to_small_hours(txn)
        txn.injected_scenario = AnomalyScenario.AMOUNT_SPIKE
        return [txn]

    def _scenario_new_device(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> list[GeneratedTransaction]:
        """Scenario 2 - activity from a device never seen on the account.

        A device change often comes with a location change, as in an account
        accessed from somewhere the customer has never been.
        """
        rnd = self.random
        txn = self._normal_transaction(customer, index, when)
        txn.device_ref = self._new_device_ref()
        txn.amount = round(customer.typical_amount * rnd.uniform(2.5, 7.0), 2)
        if rnd.random() < 0.55:
            self._move_to_far_city(customer, txn)
        if rnd.random() < 0.45:
            self._send_to_new_payee(txn)
        txn.injected_scenario = AnomalyScenario.NEW_DEVICE
        return [txn]

    def _scenario_unusual_location(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> list[GeneratedTransaction]:
        """Scenario 3 - a far-away city the customer has never transacted from."""
        rnd = self.random
        txn = self._normal_transaction(customer, index, when)
        self._move_to_far_city(customer, txn)
        txn.amount = round(customer.typical_amount * rnd.uniform(2.0, 6.0), 2)
        if rnd.random() < 0.50:
            txn.device_ref = self._new_device_ref()
        if rnd.random() < 0.35:
            self._send_to_new_payee(txn)
        txn.injected_scenario = AnomalyScenario.UNUSUAL_LOCATION
        return [txn]

    def _scenario_unusual_time(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> list[GeneratedTransaction]:
        """Scenario 4 - activity in the small hours, well outside normal."""
        rnd = self.random
        txn = self._normal_transaction(customer, index, when)
        self._move_to_small_hours(txn)
        txn.amount = round(customer.typical_amount * rnd.uniform(2.0, 6.0), 2)
        if rnd.random() < 0.60:
            self._send_to_new_payee(txn)
        if rnd.random() < 0.30:
            txn.device_ref = self._new_device_ref()
        txn.injected_scenario = AnomalyScenario.UNUSUAL_TIME
        return [txn]

    def _scenario_velocity_spike(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> list[GeneratedTransaction]:
        """Scenario 5 - a rapid burst of transactions within minutes.

        Modelled on an account being emptied: the amounts climb through the
        burst and the later transfers go to payees the account has never used.
        """
        rnd = self.random
        burst: list[GeneratedTransaction] = []
        moment = when.replace(minute=rnd.randint(0, 40), second=0, microsecond=0)
        count = rnd.randint(4, 7)
        drain_to_new_payees = rnd.random() < 0.7
        for step in range(count):
            txn = self._normal_transaction(customer, index, moment + timedelta(minutes=step * 2))
            # Amounts escalate as the burst proceeds.
            txn.amount = round(
                customer.typical_amount * rnd.uniform(1.5, 3.0) * (1 + step * 0.6), 2
            )
            if drain_to_new_payees and step >= 1:
                self._send_to_new_payee(txn)
            txn.injected_scenario = AnomalyScenario.VELOCITY_SPIKE
            burst.append(txn)
        return burst

    def _scenario_multi_signal(
        self, customer: GeneratedCustomer, index: int, when: datetime
    ) -> list[GeneratedTransaction]:
        """Scenario 6 - several signals at once: the clearest cases in the demo."""
        rnd = self.random
        options = [c for c in CITIES if c[0] != customer.home_city]
        city, region, lat, lon = rnd.choice(options)
        txn = self._normal_transaction(customer, index, when)
        txn.occurred_at = when.replace(hour=rnd.choice((2, 3, 4)), minute=rnd.randint(0, 59))
        txn.location_city, txn.location_region = city, region
        txn.latitude, txn.longitude = lat, lon
        txn.device_ref = f"DEV-{rnd.getrandbits(40):010x}"
        txn.beneficiary_ref = f"BEN-{rnd.getrandbits(36):09x}"
        txn.amount = round(customer.typical_amount * rnd.uniform(18, 45), 2)
        txn.payment_method = rnd.choice((PaymentMethod.IMPS, PaymentMethod.RTGS, PaymentMethod.NEFT))
        txn.injected_scenario = AnomalyScenario.MULTI_SIGNAL
        return [txn]

    # --- Orchestration ----------------------------------------------------
    def generate(self) -> GeneratedDataset:
        """Build the full dataset."""
        rnd = self.random
        customers = [self._make_customer(i) for i in range(self.customer_count)]

        # Scale each customer's pace so the totals land near the target. The
        # weekend damping below removes ~10% of expected volume, so it is
        # divided back out here rather than left to undershoot.
        weekend_days = sum(
            1
            for d in range(self.history_days)
            if (self.start_at + timedelta(days=d)).weekday() >= 5
        )
        mean_day_factor = (
            (self.history_days - weekend_days) + weekend_days * WEEKEND_FACTOR
        ) / self.history_days
        weight_total = sum(c.daily_rate for c in customers) * self.history_days * mean_day_factor
        scale = self.target_transactions / weight_total if weight_total else 1.0

        transactions: list[GeneratedTransaction] = []
        for index, customer in enumerate(customers):
            per_day = customer.daily_rate * scale
            for day_offset in range(self.history_days):
                day = self.start_at + timedelta(days=day_offset)
                # Weekends are quieter.
                weekday_factor = WEEKEND_FACTOR if day.weekday() >= 5 else 1.0
                count = _poisson(rnd, per_day * weekday_factor)
                for _ in range(count):
                    transactions.append(
                        self._normal_transaction(customer, index, self._normal_time(customer, day))
                    )

        anomalies = self._inject_anomalies(customers, transactions)
        transactions.extend(anomalies)
        transactions.sort(key=lambda t: t.occurred_at)

        scenario_counts: dict[str, int] = {}
        for txn in transactions:
            if txn.injected_scenario is not None:
                key = txn.injected_scenario.value
                scenario_counts[key] = scenario_counts.get(key, 0) + 1

        self._update_reference_counts(customers, transactions)
        return GeneratedDataset(
            customers=customers, transactions=transactions, scenario_counts=scenario_counts
        )

    def _inject_anomalies(
        self, customers: list[GeneratedCustomer], baseline: list[GeneratedTransaction]
    ) -> list[GeneratedTransaction]:
        """Place scenarios on a subset of customers, late in the window.

        Anomalies land in the last third of the history so each affected
        customer already has an established baseline to deviate from - exactly
        the situation the detection stack is meant to catch.
        """
        rnd = self.random
        scenarios = (
            self._scenario_amount_spike,
            self._scenario_new_device,
            self._scenario_unusual_location,
            self._scenario_unusual_time,
            self._scenario_velocity_spike,
            self._scenario_multi_signal,
        )

        affected_count = max(6, int(len(customers) * self.anomaly_customer_share))
        affected = rnd.sample(range(len(customers)), min(affected_count, len(customers)))

        earliest = self.start_at + timedelta(days=int(self.history_days * 0.66))
        window_days = max((self.end_at - earliest).days, 1)

        injected: list[GeneratedTransaction] = []
        for position, customer_index in enumerate(affected):
            customer = customers[customer_index]
            # Rotate the first pick so every scenario is represented regardless
            # of sample size, then add a few at random.
            low, high = self.scenarios_per_customer
            chosen = [scenarios[position % len(scenarios)]]
            chosen += [rnd.choice(scenarios) for _ in range(rnd.randint(low, high) - 1)]
            for scenario in chosen:
                when = self._normal_time(
                    customer, earliest + timedelta(days=rnd.randint(0, window_days - 1))
                )
                injected.extend(scenario(customer, customer_index, when))
        return injected

    def _update_reference_counts(
        self, customers: list[GeneratedCustomer], transactions: list[GeneratedTransaction]
    ) -> None:
        """Backfill device usage and beneficiary payment counts.

        Anomaly scenarios introduce device and beneficiary references that no
        customer owns yet; those are registered here so the transaction rows can
        resolve their foreign keys.
        """
        by_customer_devices = {
            i: {d.device_ref: d for d in c.devices} for i, c in enumerate(customers)
        }
        by_customer_beneficiaries = {
            i: {b.beneficiary_ref: b for b in c.beneficiaries} for i, c in enumerate(customers)
        }

        for txn in transactions:
            customer = customers[txn.customer_index]
            devices = by_customer_devices[txn.customer_index]
            beneficiaries = by_customer_beneficiaries[txn.customer_index]

            if txn.device_ref:
                device = devices.get(txn.device_ref)
                if device is None:
                    device = GeneratedDevice(
                        device_ref=txn.device_ref,
                        device_type="MOBILE",
                        operating_system="Android",
                        first_seen_at=txn.occurred_at,
                        last_seen_at=txn.occurred_at,
                        is_trusted=False,
                    )
                    devices[txn.device_ref] = device
                    customer.devices.append(device)
                device.usage_count += 1
                device.last_seen_at = max(device.last_seen_at, txn.occurred_at)
                device.first_seen_at = min(device.first_seen_at, txn.occurred_at)

            if txn.beneficiary_ref:
                beneficiary = beneficiaries.get(txn.beneficiary_ref)
                if beneficiary is None:
                    beneficiary = GeneratedBeneficiary(
                        beneficiary_ref=txn.beneficiary_ref,
                        display_name=f"{self.random.choice(LAST_NAMES)} "
                        f"{self.random.choice(BENEFICIARY_KINDS)}",
                        account_number=f"{self.random.randint(10**11, 10**12 - 1)}",
                        bank_name=self.random.choice(BANKS),
                        first_seen_at=txn.occurred_at,
                    )
                    beneficiaries[txn.beneficiary_ref] = beneficiary
                    customer.beneficiaries.append(beneficiary)
                beneficiary.payment_count += 1
                beneficiary.first_seen_at = min(beneficiary.first_seen_at, txn.occurred_at)


def _poisson(rnd: random.Random, lam: float) -> int:
    """Draw from a Poisson distribution (Knuth's method).

    Transaction counts per day are counts of independent events, so Poisson
    gives a far more natural spread than a uniform integer would.
    """
    if lam <= 0:
        return 0
    if lam > 30:  # Normal approximation; Knuth's loop gets slow up here.
        return max(0, int(rnd.gauss(lam, math.sqrt(lam)) + 0.5))
    target = math.exp(-lam)
    count, product = 0, 1.0
    while True:
        product *= rnd.random()
        if product <= target:
            return count
        count += 1
