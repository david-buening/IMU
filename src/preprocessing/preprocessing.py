import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_FILE  = DATA_DIR / "preprocessed.csv"

# Order matters: check longer/more-specific keys before shorter ones
# to avoid partial-match issues (e.g. "left_hand" before "lefthand")
CARRY_MAP = [
    ("both_hands",  "both"),
    ("bothhands",   "both"),
    ("amidextrous", "both"),
    ("left_hand",   "left"),
    ("lefthand",    "left"),
    ("right_hand",  "right"),
    ("righthand",   "right"),
]


def parse_metadata(csv_path: Path) -> dict:
    """Extract all metadata from file path and folder name."""
    stem   = csv_path.stem.lower()          # e.g. "versuch_david_both_hands_big_floor_l"
    folder = csv_path.parent.name.lower()   # e.g. "big box david"  (handles "smal box david" typo)

    # subject
    subject = "David" if "david" in stem else "Viktor"

    # sensor side: always the last character of the stem, uppercase
    sensor_hand = csv_path.stem[-1]  # "L" or "R"

    # box size from folder name ("smal" also contains "smal" but not "big", safe)
    box_size = "big" if "big" in folder else "small"

    # surface ("ground" falls into else → "floor", unifying Viktor/David naming)
    surface = "table" if "table" in stem else "floor"

    # carrying hand via keyword lookup
    try:
        carrying_hand = next(val for key, val in CARRY_MAP if key in stem)
    except StopIteration:
        raise ValueError(f"Cannot parse carrying_hand from filename: {csv_path.name}")

    return {
        "subject":          subject,
        "box_size":         box_size,
        "sensor_hand":      sensor_hand,
        "surface":          surface,
        "one_handed_carry": carrying_hand != "both",
    }


def load_all_data(data_dir: Path) -> pd.DataFrame:
    dfs, skipped = [], []

    for csv_path in sorted(data_dir.glob("**/*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            skipped.append(csv_path.name)
            continue
        meta = parse_metadata(csv_path)
        for col, val in meta.items():
            df[col] = val
        dfs.append(df)

    if skipped:
        print(f"Skipped {len(skipped)} empty file(s):")
        for name in skipped:
            print(f"  - {name}")

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    df = load_all_data(DATA_DIR)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT_FILE}")

    print("\nRow counts per metadata group:")
    summary = (
        df.groupby(["subject", "box_size", "sensor_hand", "surface", "one_handed_carry"])
        .size()
        .reset_index(name="n_rows")
    )
    print(summary.to_string(index=False))
