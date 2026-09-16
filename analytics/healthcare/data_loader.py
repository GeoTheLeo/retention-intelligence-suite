from pathlib import Path

from shared.utils.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "business"
    / "healthcare"
    / "dataset_manifest.yaml"
)

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "healthcare_patients.csv"
)


def main() -> None:

    loader = DataLoader(MANIFEST)

    dataframe = loader.load_csv(CSV_FILE)

    loader.dataset_summary(dataframe)

    print()
    print(dataframe.head())


if __name__ == "__main__":
    main()
