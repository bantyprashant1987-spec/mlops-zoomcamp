import json
from pathlib import Path

# Path to the notebook
notebook_path = Path("03-training/experiment_tracking/trainamodel.ipynb")

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replacements to make
replacements = [
    # Initial data loading cell
    ('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet', 'data/yellow_tripdata_2023-01.parquet'),
    ('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet', 'data/yellow_tripdata_2023-02.parquet'),
    ('lpep_dropoff_datetime', 'tpep_dropoff_datetime'),
    ('lpep_pickup_datetime', 'tpep_pickup_datetime'),
    ("df = df[df.trip_type == 2]", "# Removed trip_type filter for yellow taxi data"),
    ("'objective': 'reg:linear'", "'objective': 'reg:squarederror'"),
    ('"objective": "reg:linear"', '"objective": "reg:squarederror"'),
    ("./data/green_tripdata_2021-01.parquet", "data/yellow_tripdata_2023-01.parquet"),
    ("./data/green_tripdata_2021-02.parquet", "data/yellow_tripdata_2023-02.parquet"),
    ("#numerical = ['trip_distance']", "numerical = ['trip_distance']"),
    ("df[((df.duration >= 1) & (df.duration <= 60))]", "df = df[((df.duration >= 1) & (df.duration <= 60))]"),
]

# Apply replacements to all cells
count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        original = source
        
        for old, new in replacements:
            source = source.replace(old, new)
        
        if source != original:
            cell['source'] = source.split('\n')
            # Add back newlines
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            count += 1
            print(f"Updated cell: {cell.get('metadata', {}).get('id', 'unknown')}")

# Write the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Updated {count} cells")
print(f"✅ Notebook saved to {notebook_path}")
