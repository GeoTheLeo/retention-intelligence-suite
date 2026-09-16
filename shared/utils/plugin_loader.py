from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUSINESS_DIR = PROJECT_ROOT / "business"


@dataclass(slots=True)
class Plugin:
    """Represents a single industry plugin."""

    id: str
    name: str
    version: str
    enabled: bool
    path: Path


class PluginLoader:
    """Loads every enabled business plugin."""

    REQUIRED_FILES = (
        "business_problem.yaml",
        "dataset_manifest.yaml",
        "dashboard.yaml",
        "recommendations.yaml",
        "prompts.yaml",
    )

    def __init__(self, plugin_directory: Path = BUSINESS_DIR) -> None:
        self.plugin_directory = plugin_directory

    def discover(self) -> list[Plugin]:
        """Return all enabled plugins."""

        plugins: list[Plugin] = []

        for directory in sorted(self.plugin_directory.iterdir()):

            if not directory.is_dir():
                continue

            plugin_file = directory / "plugin.yaml"

            if not plugin_file.exists():
                raise FileNotFoundError(
                    f"Missing plugin descriptor:\n{plugin_file}"
                )

            self._validate(directory)

            with plugin_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                config = yaml.safe_load(file)

            if not config["enabled"]:
                continue

            plugins.append(
                Plugin(
                    id=config["id"],
                    name=config["name"],
                    version=config["version"],
                    enabled=config["enabled"],
                    path=directory,
                )
            )

        return plugins

    def _validate(self, directory: Path) -> None:
        """Ensure every required file exists."""

        missing = [
            file
            for file in self.REQUIRED_FILES
            if not (directory / file).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"{directory.name} is missing:\n"
                + "\n".join(missing)
            )


def main() -> None:

    print("=" * 50)
    print("Retention Intelligence Suite")
    print("Plugin Discovery")
    print("=" * 50)
    print()

    loader = PluginLoader()

    plugins = loader.discover()

    for plugin in plugins:
        print(f"✓ {plugin.name}")

    print()
    print(f"{len(plugins)} plugins successfully loaded.")


if __name__ == "__main__":
    main()