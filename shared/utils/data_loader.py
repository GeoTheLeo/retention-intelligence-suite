from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


class DataLoader:
    """
    Generic dataset loader driven by dataset_manifest.yaml.
    """

    def __init__(self, manifest_path: str | Path):

        self.manifest_path = Path(manifest_path)

        self.project_root = self.manifest_path.parents[2]

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:

        with open(self.manifest_path, "r", encoding="utf-8") as file:

            return yaml.safe_load(file)

    def show_available_datasets(self) -> None:

        table = Table(title="Datasets Defined in Manifest")

        table.add_column("Dataset")

        table.add_column("Source")

        table.add_column("Update")

        table.add_column("Grain")

        for dataset_name, config in self.manifest["datasets"].items():

            table.add_row(
                dataset_name,
                config["source"],
                config["update_frequency"],
                config["grain"],
            )

        console.print(table)

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:

        csv_path = Path(csv_path)

        console.print(f"\nLoading: {csv_path.name}")

        dataframe = pd.read_csv(csv_path)

        console.print(
            f"Loaded {len(dataframe):,} rows × "
            f"{len(dataframe.columns)} columns"
        )

        return dataframe

    @staticmethod
    def validate_columns(
        dataframe: pd.DataFrame,
        required_columns: list[str],
    ) -> bool:

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:

            console.print("\nMissing columns:")

            for column in missing:

                console.print(f"   • {column}")

            return False

        console.print("\nColumn validation passed.")

        return True

    @staticmethod
    def dataset_summary(dataframe: pd.DataFrame) -> None:

        table = Table(title="Dataset Summary")

        table.add_column("Metric")

        table.add_column("Value")

        table.add_row("Rows", f"{len(dataframe):,}")

        table.add_row("Columns", str(len(dataframe.columns)))

        table.add_row(
            "Memory (MB)",
            f"{dataframe.memory_usage(deep=True).sum()/1024**2:.2f}",
        )

        console.print(table)