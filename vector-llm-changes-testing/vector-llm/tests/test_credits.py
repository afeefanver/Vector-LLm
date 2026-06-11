"""
tests/test_credits.py
Tests for CreditService — all database calls are mocked (no PostgreSQL needed).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm.credits import CreditService, InsufficientCreditsError, CreditServiceError, CREDIT_COSTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pool():
    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))
    pool = AsyncMock()
    pool.acquire = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=conn),
        __aexit__=AsyncMock(return_value=False),
    ))
    return pool, conn


@pytest.fixture
def service(mock_pool):
    pool, conn = mock_pool
    s = CreditService()
    s._pool = pool
    return s, conn


# ---------------------------------------------------------------------------
# cost_for()
# ---------------------------------------------------------------------------

class TestCostFor:

    def test_dashboard_costs_2(self):
        s = CreditService()
        assert s.cost_for("dashboard") == 2

    def test_decision_costs_2(self):
        s = CreditService()
        assert s.cost_for("decision") == 2

    def test_general_costs_1(self):
        s = CreditService()
        assert s.cost_for("general") == 1

    def test_summary_costs_1(self):
        s = CreditService()
        assert s.cost_for("summary") == 1

    def test_unknown_operation_defaults_to_1(self):
        s = CreditService()
        assert s.cost_for("unknown_op") == 1

    def test_case_insensitive(self):
        s = CreditService()
        assert s.cost_for("DASHBOARD") == s.cost_for("dashboard")

    def test_embed_is_free(self):
        s = CreditService()
        assert s.cost_for("embed") == 0

    def test_all_operations_defined(self):
        s = CreditService()
        for op in ["general", "dashboard", "decision", "forecast", "summary", "upload"]:
            assert s.cost_for(op) >= 0


# ---------------------------------------------------------------------------
# get_balance()
# ---------------------------------------------------------------------------

class TestGetBalance:

    async def test_returns_balance_for_existing_user(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 42})
        result = await s.get_balance("u_001")
        assert result == 42

    async def test_returns_zero_for_new_user(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value=None)
        result = await s.get_balance("u_new")
        assert result == 0


# ---------------------------------------------------------------------------
# check_and_reserve()
# ---------------------------------------------------------------------------

class TestCheckAndReserve:

    async def test_sufficient_balance_deducts_and_returns_new_balance(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 10})
        conn.execute  = AsyncMock()
        result = await s.check_and_reserve("u_001", cost=2)
        assert result == 8

    async def test_insufficient_balance_raises(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 1})
        with pytest.raises(InsufficientCreditsError) as exc_info:
            await s.check_and_reserve("u_001", cost=2)
        assert exc_info.value.required  == 2
        assert exc_info.value.available == 1
        assert exc_info.value.user_id   == "u_001"

    async def test_zero_cost_skips_db_write(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 5})
        await s.check_and_reserve("u_001", cost=0)
        conn.execute.assert_not_called()

    async def test_exact_balance_succeeds(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 2})
        conn.execute  = AsyncMock()
        result = await s.check_and_reserve("u_001", cost=2)
        assert result == 0

    async def test_new_user_with_zero_balance_raises(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value=None)
        with pytest.raises(InsufficientCreditsError):
            await s.check_and_reserve("u_new", cost=1)

    async def test_transaction_logged(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 10})
        conn.execute  = AsyncMock()
        await s.check_and_reserve("u_001", cost=2)
        # At least one INSERT into credit_transactions
        insert_calls = [
            c for c in conn.execute.call_args_list
            if "INSERT INTO credit_transactions" in str(c)
        ]
        assert len(insert_calls) >= 1


# ---------------------------------------------------------------------------
# refund()
# ---------------------------------------------------------------------------

class TestRefund:

    async def test_refund_adds_credits_back(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 8})
        conn.execute  = AsyncMock()
        result = await s.refund("u_001", cost=2, reason="job_failed")
        assert result == 10

    async def test_refund_zero_cost_is_noop(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 5})
        await s.refund("u_001", cost=0, reason="noop")
        conn.execute.assert_not_called()

    async def test_refund_logs_transaction(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 5})
        conn.execute  = AsyncMock()
        await s.refund("u_001", cost=2, reason="failed")
        insert_calls = [
            c for c in conn.execute.call_args_list
            if "INSERT INTO credit_transactions" in str(c)
        ]
        assert len(insert_calls) >= 1


# ---------------------------------------------------------------------------
# top_up()
# ---------------------------------------------------------------------------

class TestTopUp:

    async def test_top_up_returns_new_balance(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 110})
        conn.execute  = AsyncMock()
        result = await s.top_up("u_001", amount=100)
        assert result == 110

    async def test_top_up_zero_raises(self, service):
        s, conn = service
        with pytest.raises(ValueError):
            await s.top_up("u_001", amount=0)

    async def test_top_up_negative_raises(self, service):
        s, conn = service
        with pytest.raises(ValueError):
            await s.top_up("u_001", amount=-50)

    async def test_top_up_logs_transaction(self, service):
        s, conn = service
        conn.fetchrow = AsyncMock(return_value={"balance_credits": 100})
        conn.execute  = AsyncMock()
        await s.top_up("u_001", amount=100)
        insert_calls = [
            c for c in conn.execute.call_args_list
            if "INSERT INTO credit_transactions" in str(c)
        ]
        assert len(insert_calls) >= 1


# ---------------------------------------------------------------------------
# InsufficientCreditsError
# ---------------------------------------------------------------------------

class TestInsufficientCreditsError:

    def test_error_message_contains_user_id(self):
        err = InsufficientCreditsError("u_001", required=5, available=2)
        assert "u_001" in str(err)

    def test_error_has_correct_attributes(self):
        err = InsufficientCreditsError("u_001", required=5, available=2)
        assert err.required  == 5
        assert err.available == 2
        assert err.user_id   == "u_001"

    def test_is_exception(self):
        err = InsufficientCreditsError("u_001", required=1, available=0)
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# CreditServiceError — raised when DB unreachable
# ---------------------------------------------------------------------------

class TestCreditServiceError:

    async def test_raises_when_asyncpg_not_installed(self):
        s = CreditService()
        with patch.dict("sys.modules", {"asyncpg": None}):
            with pytest.raises((CreditServiceError, ImportError)):
                await s._get_pool()

    async def test_raises_when_database_url_missing(self):
        s = CreditService()
        with patch("llm.credits.os.getenv", return_value=None):
            with pytest.raises(CreditServiceError):
                await s._get_pool()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:

    async def test_close_closes_pool(self, service):
        s, conn = service
        pool = s._pool   # save ref before close sets it to None
        await s.close()
        pool.close.assert_called_once()

    async def test_close_sets_pool_to_none(self, service):
        s, conn = service
        await s.close()
        assert s._pool is None

    async def test_close_when_no_pool_is_noop(self):
        s = CreditService()
        s._pool = None
        await s.close()  # must not raise
