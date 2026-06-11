"""
auth.py
=======
API key authentication for all endpoints.

HOW IT WORKS:
  Every request must include a header:
      X-API-Key: <api_key>

  The key is looked up in the API_KEYS dict (loaded from the environment).
  If valid, the corresponding user_id is returned and injected into the
  route handler via FastAPI dependency injection.

  The user_id then drives:
    - Which ChromaDB collection is read/written (rag.py)
    - Credit balance checks (future: credits.py)
    - Audit logging

SETUP:
  In your .env file, define keys as a JSON dict:
      API_KEYS={"key_abc123": "u_001", "key_xyz789": "u_002"}

  In production, replace this dict lookup with a PostgreSQL query:
      SELECT user_id FROM api_keys WHERE key_hash = sha256($1) AND active = true

USAGE IN ROUTES:
    from auth import get_user_id

    @app.post("/query")
    async def query(req: QueryRequest, user_id: str = Depends(get_user_id)):
        context = await rag.retrieve(req.query, user_id=user_id)
        ...
"""

import os
import json
from fastapi import Header, HTTPException, status


def _load_api_keys() -> dict[str, str]:
    """
    Load the API key → user_id mapping from the environment.

    Expected format (JSON string in env var):
        API_KEYS={"key_abc123": "u_001", "key_xyz789": "u_002"}

    Falls back to a default dev key when running locally so you don't
    need to set up keys just to test. REMOVE the fallback before deploying.
    """
    raw = os.getenv("API_KEYS", "")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(
                "API_KEYS env var is not valid JSON. "
                'Expected format: \'{"key_abc": "user_001"}\''
            )

    # Dev fallback — one key maps to a dev user
    # !! Remove this block before deploying to production !!
    print("[auth] WARNING: API_KEYS not set — using dev fallback key. "
          "Set API_KEYS in .env before deploying.")
    return {"dev-key-do-not-use-in-prod": "u_dev"}


# Loaded once at startup — dict is immutable after this point
_API_KEYS: dict[str, str] = _load_api_keys()


async def get_user_id(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    FastAPI dependency. Validates the API key and returns the user_id.

    Raises HTTP 401 if the key is missing or invalid.

    Add to any route with:
        user_id: str = Depends(get_user_id)

    The header name is X-API-Key (case-insensitive in HTTP, FastAPI normalises it).
    """
    user_id = _API_KEYS.get(x_api_key)
    if not user_id:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or missing API key.",
            headers     = {"WWW-Authenticate": "ApiKey"},
        )
    return user_id