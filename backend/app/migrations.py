from pathlib import Path

from sqlalchemy import text

from app.database import engine


def run_migrations() -> None:
    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    files = sorted(migrations_dir.glob("*.sql"))
    if not files:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
        )
        for file in files:
            version = file.name
            exists = conn.execute(
                text("SELECT 1 FROM schema_migrations WHERE version = :version"),
                {"version": version},
            ).first()
            if exists:
                continue
            conn.execute(text(file.read_text(encoding="utf-8")))
            conn.execute(text("INSERT INTO schema_migrations(version) VALUES (:version)"), {"version": version})
