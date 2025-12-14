import os
import time
import jwt
from passlib.context import CryptContext
from fastapi import Depends, Request, HTTPException
from sqlalchemy.orm import Session

from db import get_db, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("AF3_GUI_JWT_SECRET", "CHANGE_ME_IN_ENV")
JWT_ALG = "HS256"
COOKIE_NAME = "af3_token"
TOKEN_TTL_SECONDS = int(os.getenv("AF3_GUI_TOKEN_TTL", "86400"))


def _bcrypt_safe_password(pw: str) -> str:
    b = (pw or "").encode("utf-8")[:72]
    return b.decode("utf-8", errors="ignore")


def hash_password(password: str) -> str:
    password = _bcrypt_safe_password(password)
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    password = _bcrypt_safe_password(password)
    return pwd_context.verify(password, password_hash)


def create_token(user_id: int, username: str, is_admin: bool) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "is_admin": bool(is_admin),
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        return None


def get_current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    user_id = int(payload["sub"])
    return db.query(User).filter(User.id == user_id).first()


def require_user(user: User | None = Depends(get_current_user_optional)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user
