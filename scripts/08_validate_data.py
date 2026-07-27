from pathlib import Path

import pandas as pd


DATA_FILE = Path(
    "data/processed/"
    "wildfire_training_2024.parquet"
)


print("Reading final training dataset...")

data = pd.read_parquet(
    DATA_FILE
)


required_columns = [
    "date",
    "grid_id",
    "centroid_lat",
    "centroid_lon",
    "temperature_max",
    "temperature_min",
    "humidity_mean",
    "humidity_min",
    "precipitation_total",
    "rain_7d",
    "rain_30d",
    "wind_speed_max",
    "fire_today",
    "fire_next_day",
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


duplicate_count = data.duplicated(
    subset=[
        "grid_id",
        "date",
    ]
).sum()


invalid_labels = data[
    ~data["fire_next_day"].isin(
        [
            0,
            1,
        ]
    )
]


print()
print("Validation summary")
print("------------------")

print(
    f"Rows: "
    f"{len(data):,}"
)

print(
    f"Grid cells: "
    f"{data['grid_id'].nunique():,}"
)

print(
    f"Start date: "
    f"{data['date'].min()}"
)

print(
    f"End date: "
    f"{data['date'].max()}"
)

print(
    f"Duplicate grid-date rows: "
    f"{duplicate_count}"
)

print(
    f"Invalid labels: "
    f"{len(invalid_labels)}"
)

print(
    f"Positive next-day labels: "
    f"{data['fire_next_day'].sum():,}"
)

print(
    f"Positive rate: "
    f"{data['fire_next_day'].mean():.6f}"
)

print()
print("Missing-value percentages")
print("-------------------------")

print(
    data[
        required_columns
    ]
    .isna()
    .mean()
    .sort_values(
        ascending=False
    )
)


if duplicate_count > 0:
    raise ValueError(
        "Duplicate grid-date rows were found."
    )

if len(invalid_labels) > 0:
    raise ValueError(
        "Invalid fire labels were found."
    )

print()
print("Validation passed.")