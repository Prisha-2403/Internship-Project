# Sentinel Finance — Fraud & Anomaly Detection Platform

An internal transaction monitoring and investigation platform for finance-company
analysts: it scores every transaction against the customer's own established
behaviour, explains each score in full, and carries the resulting investigation
through to a recorded outcome.

> **This is a decision-support system.** A risk score means a transaction
> **requires review**. It is not a determination that fraud has occurred, and
> nothing in the product presents it as one.
>
> **All demo data is synthetic.** No real person, account, device or payment is
> represented anywhere in this repository.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Setup](#setup)
- [How scoring works](#how-scoring-works)
- [The machine learning pipeline](#the-machine-learning-pipeline)
- [On model metrics](#on-model-metrics)
- [Security](#security)
- [Roles and permissions](#roles-and-permissions)
- [API](#api)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Deployment](#deployment)
- [Deliberate omissions](#deliberate-omissions)

---

## What it does

| Capability | Where |
|---|---|
| Risk overview: KPIs, volume, distribution, trends, leading signals | `/` |
| Transaction monitoring with server-side filter, sort, paginate, export | `/transactions` |
| Investigation view with a full "why was this flagged?" breakdown | `/transactions/:ref` |
| Customer risk profile contrasting baseline against recent behaviour | `/customers/:ref` |
| Anomaly worklist and on-demand detection runs | `/anomalies` |
| Alert centre with acknowledge / assign / investigate / dismiss | `/alerts` |
| Case management with evidence, notes and a stored timeline | `/cases` |
| Analytics including false-positive analysis from real outcomes | `/analytics` |
| Period risk report with CSV and Excel export | `/reports` |
| Model status, feature dictionary and version history | `/model` |
| CSV ingestion with per-row validation | `/import` |
| Append-only audit trail | `/audit` |
| User management and the live permission matrix | `/users` |
| Effective configuration and password change | `/settings` |

The sidebar currently shows only the core detection-to-investigation flow:
Overview, Transactions, Alert Centre, Anomaly Detection, Customer Risk,
Investigations and Analytics. Reports, Model Performance, Data Import, Audit Log
and User Management are built and working but hidden from navigation — their
routes stay registered, so a direct URL still resolves. Model Performance also
remains linked from the Anomaly Detection header, and System Settings from the
account menu. Set `SHOW_SECONDARY_NAV` in `frontend/src/layouts/AppShell.tsx` to
`true` to restore the full sidebar.

---

## Architecture

```
sentinel-finance/
├─ backend/     FastAPI · SQLAlchemy · Alembic · scikit-learn   (Python 3.11)
└─ frontend/    Vite · React 19 · TypeScript · Tailwind · TanStack Query · Recharts
```

The backend is strictly layered, and each layer only talks to the one below it:

```
api/          HTTP only - parse, authorise, delegate, serialise
  ↓
services/     business logic (risk scoring, cases, analytics, import/export)
  ↓
repositories/ querying
  ↓
db/models/    SQLAlchemy ORM
```

Risk scoring lives in `services/risk_scoring_service.py` and knows nothing about
HTTP or the database, so it is directly testable and behaves identically whether
it runs in a batch backfill, during a CSV import, or behind an API call.

### Database

Sixteen tables with foreign keys, indexes, unique and check constraints. Three
are worth calling out:

- **`transactions`** — the immutable financial record as ingested.
- **`transaction_features`** — engineered behavioural signals, recomputable.
- **`risk_scores`** — the scoring verdict, recomputable and versioned.

Keeping them apart means re-running detection rewrites features and scores
without ever touching the source record.

**`audit_logs` is append-only at the database level.** Migration `0001` installs
PostgreSQL rules that turn `UPDATE` and `DELETE` on that table into no-ops. An
application bug — or a compromised application role — cannot rewrite history.

---

## Setup

**Prerequisites:** Python 3.11+, Node 20+, PostgreSQL 14+.

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env               # from the repository root
```

### 3. Database

`bootstrap_db.py` needs your PostgreSQL superuser password **once**. It creates a
least-privilege `sentinel` role (no `SUPERUSER`, no `CREATEDB`), creates the
database, generates a strong application password and a `SECRET_KEY`, and writes
both into the gitignored `.env`. The superuser password is never stored.

```bash
cd backend
python -m scripts.bootstrap_db     # prompts for the superuser password
alembic upgrade head
```

### 4. Seed the demo dataset

```bash
python -m scripts.seed_demo_data
```

This generates roles, four demo users, ~120 customers, ~40,000 transactions
across 90 days with six injected anomaly scenarios, then runs the full pipeline —
feature engineering, model training, scoring — and derives alerts, cases with
realistic timelines, and audit history.

Options: `--reset`, `--transactions N`, `--customers N`, `--history-days N`,
`--seed N`, `--skip-training`.

**The seeder prints the demo sign-in accounts and generates their shared
password into `.env` as `DEMO_USER_PASSWORD`.** No credential is hardcoded
anywhere in this repository.

### 5. Run

```bash
# terminal 1
cd backend && uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend && npm install && npm run dev
```

- Application → <http://localhost:5173>
- API docs → <http://localhost:8000/docs>

---

## How scoring works

Two numbers are produced per transaction, and they are deliberately kept apart.

### Business risk score (0–100) — the analyst-facing number

The sum of named rules that fired. Each contributes explicit points:

| Rule | Signal | Max |
|---|---|---|
| `AMOUNT_DEVIATION` | Amount z-score, or plain multiple of the customer's average — whichever is stronger | 28 |
| `NEW_DEVICE` | Device not seen on this account (12 if seen once or twice) | 20 |
| `NEW_LOCATION` | City never used before (+5 beyond 500 km) | 20 |
| `NEW_BENEFICIARY` | First payment to this payee | 15 |
| `BEHAVIOR_CHANGE` | Drift in spend size and pace versus baseline | 15 |
| `HIGH_FREQUENCY` | ≥3 in 10 min, ≥6 in an hour, or ≥3× the daily norm | 14 |
| `UNUSUAL_TIME` | Hour outside the customer's established pattern | 12 |
| `AMOUNT_ABSOLUTE` | Beyond the customer's own 99th percentile | 10 |
| `ML_ANOMALY` | Bounded contribution from the Isolation Forest | 15 |

**Bands:** 0–29 Low · 30–59 Medium · 60–79 High · 80–100 Critical.

The invariant the whole design protects:

```
business_score == sum(points of the factors shown to the analyst)
```

Raw signal strength can exceed 100 when many rules fire together. Rather than
clipping — which would leave the displayed factors summing to more than the
displayed score — contributions are scaled proportionally onto the 0–100 scale
with largest-remainder rounding, and the assessment records that it happened.
The panel never shows numbers that do not add up.

The `AMOUNT_DEVIATION` rule takes the *stronger* of a z-score band and a
ratio band. That matters for customers with near-constant spending: their
standard deviation is zero, the z-score is undefined, and a 50× transaction
would otherwise score nothing — precisely the case most worth catching.

### ML anomaly score (0–1) — a separate signal

How unusual a transaction looks to an Isolation Forest, relative to the wider
population. It is **an anomaly measure, not a probability of fraud**, and it is
never labelled as one. Its contribution to the risk score is capped
(`ML_MAX_UPLIFT_POINTS`, default 15) so an unexplainable signal can never
dominate a decision an analyst has to justify to the customer it affects.

---

## The machine learning pipeline

```
load → clean → engineer features → scale → fit → score → normalise →
combine with business rules → persist
```

**Feature engineering** (`app/ml/features.py`) produces 24 documented features
per transaction, every one computed from the customer's history *strictly
before* that transaction. There is no lookahead, which means a score computed
during a backfill is the same score the transaction would have received live.
The test suite asserts this directly: it recomputes features with later rows
removed and requires every value to be unchanged.

Features cover amount deviation (z-score, ratio to average, percentile), velocity
(10-minute, hourly, daily counts and pace versus the customer's norm), timing
(hour, day, whether the hour is unusual for this customer), novelty (new device,
location, beneficiary, distance from their usual area, 30-day distinct counts)
and drift (spend and frequency trend, combined behavioural change).

**Normalisation** (`app/ml/normalization.py`) maps the unbounded
`decision_function` onto 0–1 using percentile bounds captured at fit time and
stored on the model version. Reusing the training bounds is the point: a
transaction scored next week lands on the same scale as one scored during
training, so stored scores stay comparable and the threshold does not silently
move. The anomaly threshold is the normalised position of the model's own
decision boundary, so "the model considers this anomalous" and "the score is at
or above the threshold" are the same statement.

**Without a trained model the platform still works.** Business rules alone
produce a complete, fully explainable score; the ML factor simply does not fire.

---

## On model metrics

The model page reports version, algorithm, training date, record count, feature
count and names, contamination, threshold and observed anomaly rate — all real,
all read from the registry.

It reports **no precision, recall, F1 or confusion matrix**, and says so:

> "Model performance metrics requiring labeled ground truth are unavailable."

Those metrics require confirmed outcomes for each transaction — a verified record
of which were genuinely fraudulent. This platform has none, so any such figure
would be fabricated. It is omitted rather than estimated.

What *is* measurable is on the Analytics page: how many investigations analysts
resolved as confirmed suspicious versus false positive. Those are real human
decisions. Alerts nobody has judged are reported as **pending**, not folded into
either bucket.

A separate, clearly fenced panel reports **synthetic scenario coverage** —
how often the detection stack surfaced transactions the generator injected.
That describes generated demo data and is not a measure of accuracy against real
fraud; the page says so on its face.

---

## Security

| Control | Implementation |
|---|---|
| Password hashing | bcrypt, cost 12, used directly (avoids passlib's bcrypt 4.x breakage) |
| Over-72-byte passwords | Rejected rather than silently truncated, so two long passwords cannot collide |
| Tokens | PyJWT HS256. Short-lived access token in the response body; refresh token in an httpOnly cookie |
| Token type confusion | `type` claim compared in constant time — a refresh token cannot be replayed as an access token |
| Session storage | Access token held in memory only, never `localStorage`, so XSS cannot read it |
| Immediate revocation | Every request re-reads the user and role, so deactivation or demotion takes effect at once |
| Brute force | Sliding-window limiter keyed on **IP *and* account**, so one attacker cannot lock out a real user |
| Account enumeration | Identical response and timing for unknown address and wrong password |
| RBAC | Enforced server-side per endpoint; the UI mirrors the same matrix as a courtesy |
| SQL injection | SQLAlchemy ORM throughout; no query text is ever assembled from user input |
| XSS | React escaping; the API sets a restrictive CSP and `nosniff` |
| CSV formula injection | Export cells beginning `=`, `+`, `-`, `@` are neutralised |
| PII | Masked in the Pydantic response schemas, so no endpoint can leak it by omission |
| Audit integrity | Append-only enforced by PostgreSQL rules, not by convention |
| Secrets in logs | Redacting filter masks credential patterns, bearer tokens and bcrypt digests |
| Secrets in audit | Known-sensitive keys scrubbed before any state snapshot is persisted |
| Error handling | Professional messages to the user; stack traces and database detail to the server log only |
| Secrets in git | `.env` gitignored; `.env.example` holds placeholders only |

Masking examples: `9876543210` → `9876******`, account → `XXXX XXXX 4821`,
`priya.sharma@example.com` → `p***********@e******.com`.

---

## Roles and permissions

|  | Analyst | Senior | Manager | Admin |
|---|:--:|:--:|:--:|:--:|
| View dashboards, transactions, customers, alerts, cases, analytics | ✔ | ✔ | ✔ | ✔ |
| Create case, add note, link evidence, export, acknowledge alert | ✔ | ✔ | ✔ | ✔ |
| Assign · change priority · escalate · resolve · dismiss alert | | ✔ | ✔ | ✔ |
| Close & reopen cases · audit log · run detection · reports | | | ✔ | ✔ |
| Manage users · change roles · settings · train model · import | | | | ✔ |

---

## API

Interactive documentation at `/docs`; the OpenAPI schema at `/openapi.json`.
48 operations, every one of them authenticated except `/health` and
`/api/auth/login`.

```
POST   /api/auth/login                          GET    /api/dashboard/summary
POST   /api/auth/refresh                        GET    /api/transactions
POST   /api/auth/logout                         GET    /api/transactions/{ref}
GET    /api/auth/me                             GET    /api/transactions/export
                                                POST   /api/transactions/import
GET    /api/customers                           PATCH  /api/transactions/{ref}/status
GET    /api/customers/{ref}
GET    /api/customers/{ref}/risk                GET    /api/alerts
                                                POST   /api/alerts/{ref}/acknowledge
GET    /api/cases                               POST   /api/alerts/{ref}/assign
POST   /api/cases                               POST   /api/alerts/{ref}/dismiss
GET    /api/cases/{ref}
PATCH  /api/cases/{ref}                         GET    /api/analytics
POST   /api/cases/{ref}/notes                   GET    /api/analytics/outcomes
POST   /api/cases/{ref}/transactions            GET    /api/reports/risk-summary
GET    /api/cases/{ref}/timeline                GET    /api/model/status
                                                POST   /api/model/train
GET    /api/audit-logs                          POST   /api/detection/run
GET    /api/users                               GET    /api/anomalies
POST   /api/users                               GET    /api/settings
PATCH  /api/users/{id}                          GET    /api/settings/roles
POST   /api/users/me/password                   GET    /api/search
```

---

## Testing

```bash
cd backend
python -m pytest tests/ -q
```

265 tests, all passing. 205 run anywhere; 60 exercise the API and persistence
and skip cleanly with a clear reason when no database is configured.

Coverage: the score/factor invariant (including a randomised fuzz), risk band
boundaries, every individual rule, the ML contribution cap, feature engineering
including the no-lookahead property, Isolation Forest training, scoring,
determinism and persistence, score normalisation, the synthetic generator,
password hashing, token forgery and type-confusion (including `alg: none`), rate
limiting, masking, log redaction, audit scrubbing, the permission matrix, CSV
parsing and validation, the full case lifecycle, timeline persistence, audit
writes and RBAC denials for every role against every privileged endpoint.

Six real bugs were found by these tests during development and fixed:

1. **Lookahead leak** — `customer_avg_daily_txns` was computed over each
   customer's entire history including future transactions, so a backfilled
   score would not have matched what the transaction received live.
2. **Raw PII in responses** — `CustomerIdentity` and `TransactionCustomerOut`
   serialised `full_name`, `email`, `phone` and `account_number` alongside their
   masked versions, defeating the entire masking layer. The source fields are
   now `Field(exclude=True)`: readable by the masking logic, never serialised.
3. **Log redaction ordering** — the key/value rule ran before the bearer rule,
   so `Authorization: Bearer <token>` masked the word "Bearer" and left the
   token exposed.
4. **Invalid SQL cast** — casting a boolean to `Float` for conditional counts is
   rejected by PostgreSQL, 500-ing the customer list. Replaced with `CASE`.
5. **Stale relationship after assignment** — the identity map returned a cached
   `assignee`, so a just-assigned case still read as unassigned until reload.
6. **Missing timeline event** — evidence linked while opening a case wrote no
   `TRANSACTION_LINKED` event, so the primary flow (opening an investigation
   from a flagged transaction) produced a timeline with no record of the
   evidence being attached.

Frontend:

```bash
cd frontend
npx tsc --noEmit && npm run build
```

---

## Project layout

```
backend/
  app/
    api/v1/          route modules, one per domain
    auth/            dependencies, permission matrix, auth router
    core/            config, security, rate limiting, errors, logging, middleware
    db/              models, enums, session
    ml/              features, isolation_forest, normalization, pipeline, registry
    repositories/    query layer
    schemas/         Pydantic request/response models (masking applied here)
    services/        risk_scoring_service, case, analytics, customer,
                     import, export, alert, audit, masking, synthetic_data
  alembic/versions/  migrations
  scripts/           bootstrap_db, seed_demo_data, train_model
  tests/
frontend/
  src/
    charts/          Recharts wrappers with one shared set of chart decisions
    components/      ui primitives · risk components
    hooks/           useAuth, useToast, useDebounce
    layouts/         AppShell
    lib/             api client, formatters, risk presentation, cn
    pages/           16 routes
    types/           API types mirroring the backend schemas
```

### Design notes

A restrained enterprise palette: white surfaces on a light grey plane,
navy/charcoal ink, hairline borders, almost no shadow. Colour is spent almost
entirely on risk state, so that when something is coloured it means something.

The risk ramp — `#0f9d58` → `#eda100` → `#e8590c` → `#a51f1f` — was validated for
colour-vision deficiency and normal-vision separation against the white chart
surface (worst adjacent pair: CVD ΔE 9.9, normal-vision ΔE 16.0). Severity is
carried by **lightness as well as hue**, so the four levels stay correctly
ordered in greyscale. Amber falls below 3:1 contrast against white by nature of
the hue, which is why every risk colour in the product is always paired with its
text label and, where space allows, an icon — a risk colour never carries meaning
alone.

Red is reserved for Critical only.

---

## Deployment

Free-tier across three services: **Supabase** (PostgreSQL), **Render** (the
FastAPI service) and **Netlify** (the static frontend). Because the three sit on
different origins, the session cookie is `SameSite=None; Secure` in production
and the API is given an explicit CORS origin — a wildcard is not usable when
credentials are involved.

Step-by-step instructions, including the ordering constraint between Render and
Netlify and a table of the usual failure modes, are in
**[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## Deliberate omissions

These were scoped out by decision, not overlooked:

- **Docker / docker-compose** — Docker is not installed on the target machine, so
  a compose file could not be verified to run. The native setup above is the
  supported and tested path.
- **PDF report generation** — Reports renders on screen and exports CSV and Excel.
- **Frontend unit tests** — the frontend is typechecked and built in CI-style
  verification; behavioural tests are backend-side.
- **WebSockets** — the alert centre polls on an interval. For an internal console
  this is far simpler to make correct, and it survives reconnects, sleeping
  laptops and proxy timeouts without any reconnect logic of its own.
- **Redis / Celery** — the deployment runs a single API process, so the in-memory
  rate limiter is sufficient and a broker would add operational weight for
  nothing. `SlidingWindowLimiter` can be swapped for a shared store without any
  change to its callers.
