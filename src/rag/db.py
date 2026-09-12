"""Synchronous SQLAlchemy setup and transaction helpers."""

from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from rag.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all persisted entities."""


def build_engine(database_url: str | None = None, **overrides: object) -> Engine:
    """Build an engine, allowing tests and maintenance scripts to supply a URL."""

    settings = get_settings()
    url = database_url or settings.database_url
    options: dict[str, object] = {
        "pool_pre_ping": True,
        "echo": settings.database_echo,
    }
    if not url.startswith("sqlite"):
        options.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
    options.update(overrides)
    return create_engine(url, **options)


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session with request-owned lifetime."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# A more explicit alias for callers that do not use FastAPI's dependency system.
get_session = get_db


@contextmanager
def session_scope(session_factory: sessionmaker[Session] = SessionLocal) -> Iterator[Session]:
    """Run a unit of work in one transaction, rolling back on any exception."""

    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database(connection_engine: Engine = engine) -> None:
    """Fail quickly if the database cannot execute a trivial query."""

    with connection_engine.connect() as connection:
        connection.execute(text("SELECT 1"))
