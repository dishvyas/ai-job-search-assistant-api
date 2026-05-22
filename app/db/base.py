# Isolated module for the SQLAlchemy declarative base.
# Keeping Base here — separate from session.py and from the models themselves —
# avoids circular imports: models import Base, session imports models for Alembic,
# and nothing creates a cycle.
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base. All ORM models inherit from this."""

    pass
