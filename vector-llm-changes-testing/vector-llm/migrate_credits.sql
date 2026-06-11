-- migrate_credits.sql
-- Run once against your PostgreSQL database to set up credit tracking.
-- psql -U <user> -d misvector -f migrate_credits.sql

CREATE TABLE IF NOT EXISTS user_credits (
    user_id         TEXT        PRIMARY KEY,
    balance_credits INTEGER     NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT balance_non_negative CHECK (balance_credits >= 0)
);

CREATE TABLE IF NOT EXISTS credit_transactions (
    id              BIGSERIAL   PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    delta           INTEGER     NOT NULL,
    reason          TEXT        NOT NULL,
    tokens_used     INTEGER,
    balance_after   INTEGER     NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_credit_tx_user
    ON credit_transactions(user_id, created_at DESC);

-- Optional: foreign key if you have a users table
-- ALTER TABLE user_credits ADD CONSTRAINT fk_user
--     FOREIGN KEY (user_id) REFERENCES users(id);
