from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "N1": ("paracetamol", "reference"),
    "K12": ("paracetamol", "reference"),
    "N29": ("paracetamol", "adulterated"),
    "N31": ("paracetamol", "adulterated"),
    "N36": ("paracetamol", "adulterated"),
    "N18": ("aspirin", "reference"),
    "N10": ("vitamin_c", "reference"),
}

BANDS = {
    "ch450": (445, 455),
    "ch500": (495, 505),
    "ch550": (545, 555),
    "ch570": (565, 575),
    "ch600": (595, 605),
    "ch650": (645, 655),
}


def find_raw_dir() -> Path:
    root = Path(__file__).resolve().parent
    candidates = [root / "data" / "raw", root / "data" / "public"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No ASD data directory found. Expected data/raw or data/public.")


def file_dataset_name(path: Path) -> str:
    name = path.name.replace("_Avg_ASD.xlsx", "")
    return name


def load_asd_spectra(raw_dir: Path):
    files = sorted(raw_dir.glob("*_Avg_ASD.xlsx"))
    if not files:
        raise FileNotFoundError(f"No ASD Excel files found in {raw_dir}.")

    frames = []
    for file_path in files:
        df = pd.read_excel(file_path)
        if df.empty:
            continue

        first_col = df.columns[0]
        df = df.rename(columns={first_col: "sample_code"})
        df["sample_code"] = df["sample_code"].astype(str).str.strip()
        df["dataset_source"] = file_dataset_name(file_path)

        numeric_cols = [
            c for c in df.columns[1:]
            if str(c).replace(".", "", 1).isdigit()
        ]

        for channel, (low, high) in BANDS.items():
            band_cols = [
                c for c in numeric_cols
                if low <= float(c) <= high
            ]
            if band_cols:
                df[channel] = df[band_cols].mean(axis=1, skipna=True)

        frames.append(df)

    if not frames:
        raise ValueError("No valid ASD spectra could be loaded.")

    return pd.concat(frames, ignore_index=True)


def load_metadata(raw_dir: Path):
    meta_path = raw_dir / "metadata.xlsx"
    if not meta_path.exists():
        return None
    meta = pd.read_excel(meta_path)
    if "code" in meta.columns:
        meta["code"] = meta["code"].astype(str).str.strip()
    return meta


def main():
    raw_dir = find_raw_dir()
    spectra = load_asd_spectra(raw_dir)
    metadata = load_metadata(raw_dir)

    target_order = ["N1", "K12", "N29", "N31", "N36", "N18", "N10"]
    filtered = spectra[spectra["sample_code"].isin(target_order)].copy()

    if metadata is not None:
        meta_targets = metadata[metadata["code"].isin(target_order)].copy()
        print("Metadata verification:")
        print(meta_targets[["code", "component", "set"]].to_string(index=False))
        print()

    if len(filtered) != len(target_order):
        missing = [code for code in target_order if code not in filtered["sample_code"].unique()]
        raise ValueError(f"Target samples missing from ASD files: {missing}")

    filtered["medicine"] = filtered["sample_code"].map(lambda x: TARGETS[x][0])
    filtered["status"] = filtered["sample_code"].map(lambda x: TARGETS[x][1])

    output_dir = raw_dir.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "medicines_as7262.csv"

    result = filtered[[
        "sample_code",
        "dataset_source",
        "medicine",
        "status",
        "ch450",
        "ch500",
        "ch550",
        "ch570",
        "ch600",
        "ch650",
    ]].copy()

    print("Number of rows:", len(result))
    print("Number of columns:", len(result.columns))
    print("Selected sample codes:", result["sample_code"].tolist())
    print("Dataset source for each sample:")
    print(result[["sample_code", "dataset_source"]].to_string(index=False))
    print("Medicine/status for each sample:")
    print(result[["sample_code", "medicine", "status"]].to_string(index=False))

    nan_channels = {channel: bool(result[channel].isna().any()) for channel in BANDS}
    print("Any AS7262 channel contains NaN:", nan_channels)
    print("Six simulated channel values:")
    print(result[["sample_code", "ch450", "ch500", "ch550", "ch570", "ch600", "ch650"]].to_string(index=False))
    print("Final output path:", output_path)

    if any(result[channel].isna().any() for channel in BANDS):
        raise ValueError("At least one AS7262 channel contains NaN values.")

    result.to_csv(output_path, index=False)
    print(f"\nSaved dataset to: {output_path}")


if __name__ == "__main__":
    main()
