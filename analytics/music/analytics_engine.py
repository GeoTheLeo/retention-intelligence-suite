from __future__ import annotations

from pathlib import Path

from rich.console import Console

from shared.utils.duckdb_manager import DuckDBManager


console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW = (
    PROJECT_ROOT
    / "data"
    / "warehouse"
    / "music"
    / "raw"
)


def load_tables(db: DuckDBManager) -> None:

    db.execute(f"""
        CREATE OR REPLACE TABLE users AS
        SELECT *
        FROM read_csv_auto('{RAW / "users.csv"}')
    """)

    db.execute(f"""
        CREATE OR REPLACE TABLE subscriptions AS
        SELECT *
        FROM read_csv_auto('{RAW / "subscriptions.csv"}')
    """)


def executive_summary(db: DuckDBManager):

    return db.dataframe(
        """
        SELECT

        COUNT(*) AS customers,

        AVG(monthly_fee) AS arpu,

        SUM(active) AS active_subscribers

        FROM subscriptions
        """
    )


def main():

    console.rule("[bold cyan]Retention Intelligence Suite[/bold cyan]")

    db = DuckDBManager()

    load_tables(db)

    summary = executive_summary(db)

    console.print(summary)

    db.close()


if __name__ == "__main__":

    main()