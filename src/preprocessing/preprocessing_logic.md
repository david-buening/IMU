# Preprocessing Logic

## Purpose
Merges all 64 raw IMU CSV files into a single unified dataset at `data/preprocessed.csv`.
All experiment metadata is extracted from file and folder names and added as structured columns.

## Input
```
data/
├── Big Box David/       versuch_David_*_{L|R}.csv
├── Big Box Viktor/      versuch_Viktor_*_{L|R}.csv
├── Smal Box David/      versuch_David_*_{L|R}.csv   ← folder name typo, handled
└── Small Box Viktor/    versuch_Viktor_*_{L|R}.csv
```

Each raw CSV has 9 columns recorded at ~100 Hz:
`time_s, server_time_s, GX, GY, GZ, AX, AY, AZ, phase`

## Output Schema (`data/preprocessed.csv`)

| Column | Type | Description |
|---|---|---|
| `time_s` | float | Milliseconds since Start signal, converted to seconds (watch-local clock) |
| `server_time_s` | float | Elapsed seconds on the server since recording started |
| `GX, GY, GZ` | float | Gyroscope readings in °/s |
| `AX, AY, AZ` | float | Accelerometer readings in g |
| `phase` | str | Movement phase: `Aufheben` (pick up) / `Laufen` (carry) / `Absetzen` (set down) |
| `subject` | str | Participant: `David` or `Viktor` |
| `box_size` | str | Object size: `big` or `small` |
| `sensor_hand` | str | Wrist the sensor was worn on: `L` (left) or `R` (right) |
| `surface` | str | Where the box was picked up/set down: `table` or `floor` |
| `one_handed_carry` | bool | `True` if carried with one hand only, `False` if both hands used |

## Metadata Parsing Rules

### Subject
Detected from the filename: presence of `"david"` → `David`, otherwise → `Viktor`.

### Sensor hand
Always the last character of the filename stem (before `.csv`): `L` or `R`.

### Box size
Detected from the **parent folder name**: `"big"` → `big`, anything else → `small`.
This also handles the `"Smal Box David"` folder name typo.

### Surface
Detected from the filename:
- `"table"` → `table`
- `"floor"` or `"ground"` → `floor`  ← unified (David used "floor", Viktor used "ground")

### Carrying hand / one_handed_carry
Detected from the filename via keyword matching (order matters to avoid partial matches):

| Filename keyword | `one_handed_carry` |
|---|---|
| `both_hands`, `bothhands`, `amidextrous` | `False` |
| `left_hand`, `lefthand` | `True` |
| `right_hand`, `righthand` | `True` |

## Known Data Issues
- **4 empty files** skipped automatically (header-only, no sensor data):
  - `versuch_David_both_hands_big_floor_L.csv`
  - `versuch_David_both_hands_big_floor_R.csv`
  - `versuch_David_both_hands_big_floor_2_L.csv`
  - `versuch_David_both_hands_big_floor_2_R.csv`
  - These trials failed during data collection (David, big box, both hands, floor condition).

## How to Run
From the project root:
```bash
python src/preprocessing/preprocessing.py
```
Output is written to `data/preprocessed.csv`. A summary of row counts per group is printed to stdout.
