from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAREHOUSE = PROJECT_ROOT / "data" / "warehouse"

DATABASE = WAREHOUSE / "retention.duckdb"


class DuckDBManager:
    """
    Central warehouse manager.

    Every module in the Retention Intelligence Suite
    uses this single database connection.
    """

    def __init__(self) -> None:

        self.connection = duckdb.connect(DATABASE)

    def execute(self, sql: str):

        return self.connection.execute(sql)

    def dataframe(self, sql: str):

        return self.connection.execute(sql).df()

    def close(self):

        self.connection.close()