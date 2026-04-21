import requests
import os
from pathlib import Path

# Create data directory
data_dir = Path("03-training/experiment_tracking/data")
data_dir.mkdir(parents=True, exist_ok=True)

# URLs for yellow taxi 2023 data
urls = {
    "yellow_tripdata_2023-01.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    "yellow_tripdata_2023-02.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet"
}

print("Starting download of yellow taxi 2023 data...")
for filename, url in urls.items():
    filepath = data_dir / filename
    
    # Skip if already exists
    if filepath.exists():
        print(f"✓ {filename} already exists")
        continue
    
    try:
        print(f"⏳ Downloading {filename}...")
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded {filename} ({file_size_mb:.1f} MB)")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

print("\nDownload complete!")
print(f"Files saved in: {data_dir.absolute()}")
