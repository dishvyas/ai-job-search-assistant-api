# Database session wiring — engine creation and the FastAPI dependency that
# hands a session to each request handler. Lives separately from base.py so
# models can import Base without pulling in the engine at import time.
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# SQLite requires check_same_thread=False for use with FastAPI's thread pool.
# PostgreSQL doesn't need this — the guard keeps the connection args correct for both.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)

# autocommit=False: transactions must be committed explicitly — prevents accidental
# partial writes from being persisted silently.
# autoflush=False: prevents SQLAlchemy from issuing extra SQL before every query;
# we control flushes explicitly via db.commit().
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        # Always close — returns the connection to the pool even if the handler raised.
        db.close()
