"""
llm/credits.py
==============
Per-user prepaid credit tracking for MISVector.

BUSINESS MODEL:
  Users pay upfront (Stripe / Razorpay) → credit balance stored in PostgreSQL.
  Before every LLM job: check balance → reject if insufficient.
  After every LLM job: deduct actual tokens used.
  Platform carries no float — revenue arrives before cost is incurred.

DATABASE SETUP (run once):
  CREATE TABLE user_credits (
      user_id         TEXT        PRIMARY KEY,
      balance_credits INTEGER     NOT NULL DEFAULT 0,
      updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
  );

  CREATE TABLE credit_transactions (
      id              BIGSERIAL   PRIMARY KEY,
      user_id         TEXT        NOT NULL,
      delta           INTEGER     NOT NULL,   -- positive = top-up, negative = deduction
      reason          TEXT        NOT NULL,   -- e.g. "top_up", "query", "dashboard"
      tokens_used     INTEGER,                -- populated on deduction
      balance_after   INTEGER     NOT NULL,
      created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
  );

  CREATE INDEX idx_credit_tx_user ON credit_transactions(user_id, created_at DESC);

CREDIT COSTS (approximate, tunable via env):
  GENERAL query    → 1 credit   (~500 tokens)
  DASHBOARD        → 2 credits  (~1200 tokens)
  DECISION         → 2 credits  (~1200 tokens)
  EMBEDDING        → 0 credits  (local, no cost)
  Top-up rate      → 100 credits = ₹10 (configurable)

LIBRARY:
  Uses asyncpg for async PostgreSQL access.
  Install: pip install asyncpg
  Set env:  DATABASE_URL=postgresql://user:pass@localhost:5432/misvector

WIRING INTO FASTAPI (when ready):
  1. Call await credits.check_and_reserve(user_id, cost) before the LLM call.
  2. Call await credits.deduct(user_id, cost, reason, tokens_used) after.
  3. If check_and_reserve raises InsufficientCreditsError, return HTTP 402.

CELERY INTEGRATION (future):
  In your Celery task: wrap the LLM call between check_and_reserve / deduct.
  Pass user_id and job_type through the task kwargs.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Cost table — credits per operation type
# Adjust these as you learn your actual token usage in production.
# ---------------------------------------------------------------------------

CREDIT_COSTS: dict[str, int] = {
    "general":   1,
    "dashboard": 2,
    "decision":  2,
    "forecast":  2,
    "summary":   1,
    "upload":    1,   # embedding cost for ingest
    "embed":     0,   # individual embed calls are free (covered by upload)
}

# Minimum balance required to make any request
MIN_BALANCE = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InsufficientCreditsError(Exception):
    """Raised when a user's balance is too low to cover the requested operation."""
    def __init__(self, user_id: str, required: int, available: int):
        self.user_id   = user_id
        self.required  = required
        self.available = available
        super().__init__(
            f"User {user_id} has {available} credits but {required} are required."
        )


class CreditServiceError(Exception):
    """Raised when the credit service cannot reach the database."""
    pass


# ---------------------------------------------------------------------------
# CreditService
# ---------------------------------------------------------------------------

class CreditService:
    """
    Async credit tracking service backed by PostgreSQL (asyncpg).

    Usage pattern:
        cost = credits.cost_for("dashboard")
        await credits.check_and_reserve(user_id, cost)       # raises on low balance
        try:
            result = await dashboard_engine.generate(...)
            await credits.deduct(user_id, cost, "dashboard", tokens_used=800)
        except Exception:
            await credits.refund(user_id, cost, "dashboard_failed")
            raise
    """

    def __init__(self):
        self._pool = None   # asyncpg connection pool — lazy init

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def _get_pool(self):
        """Lazy-init the asyncpg connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise CreditServiceError(
                    "asyncpg is not installed. Run: pip install asyncpg"
                )

            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise CreditServiceError(
                    "DATABASE_URL env var is not set. "
                    "Example: postgresql://user:pass@localhost:5432/misvector"
                )

            self._pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        return self._pool

    async def close(self):
        """Close the connection pool. Wire to app shutdown in main.py."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_balance(self, user_id: str) -> int:
        """
        Return the current credit balance for a user.
        Returns 0 if the user has no row yet (new user, not topped up).
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT balance_credits FROM user_credits WHERE user_id = $1",
                user_id,
            )
        return row["balance_credits"] if row else 0

    def cost_for(self, operation: str) -> int:
        """
        Return the credit cost for an operation type.
        Defaults to 1 for unknown operation types.

        Example:
            cost = credits.cost_for("dashboard")  # → 2
        """
        return CREDIT_COSTS.get(operation.lower(), 1)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def check_and_reserve(self, user_id: str, cost: int) -> int:
        """
        Check the user has enough credits and atomically reserve them.

        This is a SELECT FOR UPDATE to prevent race conditions when multiple
        requests arrive simultaneously for the same user.

        Returns the new balance after reservation.
        Raises InsufficientCreditsError if balance < cost.
        Raises CreditServiceError if the database is unreachable.
        """
        if cost <= 0:
            return await self.get_balance(user_id)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT balance_credits
                    FROM user_credits
                    WHERE user_id = $1
                    FOR UPDATE
                    """,
                    user_id,
                )

                current = row["balance_credits"] if row else 0

                if current < cost:
                    raise InsufficientCreditsError(
                        user_id=user_id,
                        required=cost,
                        available=current,
                    )

                new_balance = current - cost

                if row:
                    await conn.execute(
                        """
                        UPDATE user_credits
                        SET balance_credits = $1, updated_at = now()
                        WHERE user_id = $2
                        """,
                        new_balance, user_id,
                    )
                else:
                    # First time this user appears — shouldn't happen if
                    # top-up flow is enforced, but handle it gracefully.
                    await conn.execute(
                        """
                        INSERT INTO user_credits (user_id, balance_credits)
                        VALUES ($1, $2)
                        """,
                        user_id, new_balance,
                    )

                await conn.execute(
                    """
                    INSERT INTO credit_transactions
                        (user_id, delta, reason, balance_after)
                    VALUES ($1, $2, $3, $4)
                    """,
                    user_id, -cost, "reserved", new_balance,
                )

                return new_balance

    async def deduct(
        self,
        user_id:     str,
        cost:        int,
        reason:      str,
        tokens_used: Optional[int] = None,
    ) -> int:
        """
        Record a completed deduction after a successful LLM call.

        In the simple flow, check_and_reserve already deducted the credits.
        This call just updates the transaction record with actual token usage
        for billing analytics.

        Returns the current balance.
        """
        if cost <= 0:
            return await self.get_balance(user_id)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Update the most recent "reserved" transaction for this user
            # to record actual tokens used and mark it as the real operation.
            await conn.execute(
                """
                UPDATE credit_transactions
                SET reason = $1, tokens_used = $2
                WHERE id = (
                    SELECT id FROM credit_transactions
                    WHERE user_id = $3 AND reason = 'reserved'
                    ORDER BY created_at DESC
                    LIMIT 1
                )
                """,
                reason, tokens_used, user_id,
            )

            row = await conn.fetchrow(
                "SELECT balance_credits FROM user_credits WHERE user_id = $1",
                user_id,
            )
            return row["balance_credits"] if row else 0

    async def refund(self, user_id: str, cost: int, reason: str) -> int:
        """
        Refund credits when a job fails after being reserved.

        Call this in the except block if the LLM call throws after
        check_and_reserve has already deducted.

        Returns the new balance after refund.
        """
        if cost <= 0:
            return await self.get_balance(user_id)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT balance_credits FROM user_credits
                    WHERE user_id = $1 FOR UPDATE
                    """,
                    user_id,
                )
                current     = row["balance_credits"] if row else 0
                new_balance = current + cost

                await conn.execute(
                    """
                    UPDATE user_credits
                    SET balance_credits = $1, updated_at = now()
                    WHERE user_id = $2
                    """,
                    new_balance, user_id,
                )

                await conn.execute(
                    """
                    INSERT INTO credit_transactions
                        (user_id, delta, reason, balance_after)
                    VALUES ($1, $2, $3, $4)
                    """,
                    user_id, cost, reason, new_balance,
                )

                return new_balance

    async def top_up(self, user_id: str, amount: int, reason: str = "top_up") -> int:
        """
        Add credits to a user's balance (called after Stripe/Razorpay payment confirmed).

        Returns the new balance.
        """
        if amount <= 0:
            raise ValueError(f"top_up amount must be positive, got {amount}")

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO user_credits (user_id, balance_credits)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id) DO UPDATE
                    SET balance_credits = user_credits.balance_credits + $2,
                        updated_at      = now()
                    RETURNING balance_credits
                    """,
                    user_id, amount,
                )

                new_balance = row["balance_credits"]

                await conn.execute(
                    """
                    INSERT INTO credit_transactions
                        (user_id, delta, reason, balance_after)
                    VALUES ($1, $2, $3, $4)
                    """,
                    user_id, amount, reason, new_balance,
                )

                return new_balance

    async def get_transaction_history(
        self,
        user_id: str,
        limit:   int = 20,
    ) -> list[dict]:
        """
        Return recent transactions for a user — for the account/billing UI.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT delta, reason, tokens_used, balance_after, created_at
                FROM credit_transactions
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id, limit,
            )
        return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Module-level singleton
# Usage:  from llm.credits import credits, InsufficientCreditsError
# ------------------------------------------------------------------
credits = CreditService()
