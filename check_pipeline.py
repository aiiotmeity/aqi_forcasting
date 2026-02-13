import os
import sys
import pandas as pd
import datetime

print("=== Air Quality Pipeline Diagnostic ===")
print(f"Current time: {datetime.datetime.now()}")
print(f"Current directory: {os.getcwd()}")

# Check last_training.txt
if os.path.exists('last_training.txt'):
    with open('last_training.txt', 'r') as f:
        timestamp = f.read().strip()
    print(f"Last training timestamp: {timestamp}")
else:
    print("last_training.txt file not found")

# Check latest_forecast.csv
if os.path.exists('latest_forecast.csv'):
    try:
        df = pd.read_csv('latest_forecast.csv')
        print(f"latest_forecast.csv exists, shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        
        # Check file modification time
        mtime = os.path.getmtime('latest_forecast.csv')
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Last modified: {mtime_str}")
    except Exception as e:
        print(f"Error reading latest_forecast.csv: {e}")
else:
    print("latest_forecast.csv file not found")

print("=== Diagnostic Complete ===")