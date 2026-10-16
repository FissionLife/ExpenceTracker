import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)

IS_SQLITE = settings.database_url.startswith("sqlite")


def _build_engine():
    """SQLite for local/single-user use, PostgreSQL for production; chosen by config."""
    if IS_SQLITE:
        database = make_url(settings.database_url).database
        if database and database != ":memory:":
            Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True,
        )

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # off by default: needed for ON DELETE CASCADE
            cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        return eng

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
    )


engine = _build_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def describe_database() -> str:
    """Safe-to-log description of the active backend (never includes the password)."""
    if IS_SQLITE:
        return f"SQLite ({make_url(settings.database_url).database})"
    return f"PostgreSQL ({engine.url.render_as_string(hide_password=True)})"


# Columns added after the first release. `create_all` never alters an existing table, and the project has
# no migration tool, so a database made by an older version gets these added here (safe to repeat).
_ADDED_COLUMNS = [
    ("users", "default_account", "VARCHAR(80)"),
    ("transactions", "entered_at", "TIMESTAMP"),
    ("transactions", "affects_balance", "BOOLEAN DEFAULT TRUE"),
    ("transactions", "reference", "VARCHAR(80)"),
]
# Indexes added after the first release (CREATE INDEX IF NOT EXISTS works on SQLite and PostgreSQL).
_ADDED_INDEXES = [
    ("ix_transactions_user_created", "transactions", "user_id, created_at"),
    ("ix_chat_messages_user", "chat_messages", "user_id"),
    ("ix_accounts_user", "accounts", "user_id"),
    ("ix_categories_user", "categories", "user_id"),
    ("ix_merchants_user", "merchants", "user_id"),
    ("ix_api_keys_user", "api_keys", "user_id"),
]


def _add_missing_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    for table, column, ddl_type in _ADDED_COLUMNS:
        if table not in inspector.get_table_names():
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
        except Exception:
            # another process (a second worker) may have added it a moment ago; fine if it is there now
            if column not in {c["name"] for c in inspect(engine).get_columns(table)}:
                raise
            continue
        logger.info("Database upgraded: added %s.%s", table, column)


def _add_missing_indexes() -> None:
    from sqlalchemy import inspect, text

    tables = set(inspect(engine).get_table_names())
    for name, table, columns in _ADDED_INDEXES:
        if table not in tables:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({columns})"))


def _move_aliased_budgets() -> None:
    """One-time: a budget set on an alias category ("Food") moves to the canonical one
    ("Food & Dining") that expenses actually file under, so spend counts against the budget.

    Only moves the budget_limit (never merges transactions). Safe to run on every start.
    """
    from sqlalchemy import func, select

    from app.bot.categories import canonical_category
    from app.models import Category

    with SessionLocal() as db:
        budgeted = list(db.scalars(select(Category).where(Category.budget_limit.is_not(None))))
        moved = 0
        for cat in budgeted:
            canon = canonical_category(cat.name)
            if canon.lower() == cat.name.lower():
                continue  # already canonical
            target = db.scalar(
                select(Category).where(
                    Category.user_id == cat.user_id,
                    Category.type == cat.type,
                    func.lower(Category.name) == canon.lower(),
                )
            )
            if target is None or target.budget_limit is not None:
                continue  # no canonical twin, or it already has its own budget — leave as is
            target.budget_limit, cat.budget_limit = cat.budget_limit, None
            moved += 1
        if moved:
            db.commit()
            logger.info("Moved %d budget(s) onto their canonical category", moved)


def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _add_missing_indexes()
    try:
        _move_aliased_budgets()
    except Exception:  # a convenience migration must never block startup
        logger.warning("Could not reconcile aliased budgets", exc_info=True)
    logger.info("Database ready: %s", describe_database())
