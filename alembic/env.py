# Alembic migration runner.
# Alembic is used instead of SQLAlchemy's create_all() because it provides an
# auditable, versioned migration history. create_all() is convenient but can't
# track what changed, can't roll back, and diverges from the real DB schema over time.
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

# noqa: F401 — this import has no runtime effect here, but it registers
# ApplicationTailoringRun with Base.metadata so Alembic's autogenerate can see it.
import app.models.application  # noqa: F401 — registers model with Base.metadata
import app.models.job_description  # noqa: F401 — registers JobDescription with Base.metadata
from alembic import context
from app.core.config import settings
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at the same metadata the ORM uses so autogenerate can diff
# model definitions against the live schema.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL scripts)."""
    # Offline mode is useful for generating a SQL script to review or apply manually —
    # common in production deployments where Alembic doesn't have DB access at deploy time.
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    cfg = config.get_section(config.config_ini_section, {})
    # Override alembic.ini's sqlalchemy.url with the value from app settings so
    # the same DATABASE_URL env var drives both the app and migrations.
    cfg["sqlalchemy.url"] = settings.database_url

    # NullPool prevents connection pooling during migrations — each migration gets
    # a fresh connection, which avoids stale state across multiple revision steps.
    connectable = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
