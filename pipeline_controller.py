import boto3
import datetime
import pandas as pd
import json
import os
from automated_pipeline import AirQualityForecastingPipeline

# AWS Configuration
AWS_REGION = 'us-east-1'
DYNAMO_TABLE = 'ttn_aws_db'
DEVICES = ['lora-v1', 'loradev2','lora-v3']

# Device configurations for forecasting
DEVICE_CONFIGS = {
    'lora-v1': {
        'csv_file': 'updated_real4_lora_v1.csv',
        'station_name': 'Station 1'
    },
    'loradev2': {
        'csv_file': 'updated_real4_loradev2.csv', 
        'station_name': 'Station 2'
    },
    'lora-v3': {
        'csv_file': 'updated_real4_lora_v3.csv',
        'station_name': 'Station 3'
    },
}

def force_fetch_today_data(device_id):
    """Force fetch today's data using multiple strategies"""
    print(f" FORCE FETCHING TODAY'S DATA FOR: {device_id}")
    
    dynamo_client = boto3.client('dynamodb', region_name=AWS_REGION)
    today = datetime.datetime.now()
    
    # Strategy 1: Get ALL recent data and filter locally
    print(f"Strategy 1: Scanning ALL recent data...")
    
    scan_params = {
        'TableName': DYNAMO_TABLE,
        'FilterExpression': 'device_id = :device_id',
        'ExpressionAttributeValues': {
            ':device_id': {'S': device_id}
        },
        'Limit': 1000  # Get more data
    }
    
    all_items = []
    try:
        response = dynamo_client.scan(**scan_params)
        all_items.extend(response.get('Items', []))
        
        # Handle pagination to get more recent data
        while 'LastEvaluatedKey' in response and len(all_items) < 5000:
            scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = dynamo_client.scan(**scan_params)
            all_items.extend(response.get('Items', []))
        
        print(f"   Fetched {len(all_items)} total items")
        
    except Exception as e:
        print(f" Error fetching data: {e}")
        return []
    
    if not all_items:
        print(f" No data found for {device_id}")
        return []
    
    # Parse and filter for recent dates (last 7 days)
    recent_items = []
    today_items = []
    
    cutoff_date = today - datetime.timedelta(days=7)
    
    for item in all_items:
        try:
            received_at = item.get('received_at', {}).get('S', '')
            if received_at:
                # Parse timestamp
                timestamp = parse_timestamp_enhanced(received_at)
                if timestamp and timestamp >= cutoff_date:
                    recent_items.append(item)
                    
                    # Check if it's today's data
                    if timestamp.date() == today.date():
                        today_items.append(item)
        except:
            continue
    
    print(f"   Found {len(recent_items)} recent items (last 7 days)")
    print(f"   Found {len(today_items)} items from TODAY")
    
    if today_items:
        print(f"SUCCESS: Today's data exists!")
        
        # Show sample of today's data
        sample_item = today_items[0]
        payload = sample_item.get('payload', {}).get('M', {})
        pm25 = payload.get('pm25', {}).get('N', 'N/A')
        temp = payload.get('temp', {}).get('N', 'N/A')
        print(f"   Sample today data: PM2.5={pm25}, Temp={temp}")
    else:
        print(f"No today's data found, using most recent available")
    
    return recent_items

def parse_timestamp_enhanced(timestamp_str):
    """Enhanced timestamp parsing for various formats"""
    if not timestamp_str:
        return None
    
    timestamp_str = timestamp_str.strip()
    
    try:
        # Handle the nanosecond format like "2025-06-06T16:42:59.183621702Z"
        if timestamp_str.endswith('Z') and '.' in timestamp_str:
            parts = timestamp_str[:-1].split('.')
            if len(parts) == 2:
                # Truncate to microseconds (6 digits max)
                fractional = parts[1][:6].ljust(6, '0')
                timestamp_str = f"{parts[0]}.{fractional}Z"
        
        # Parse with timezone
        if timestamp_str.endswith('Z'):
            return datetime.datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
        else:
            return datetime.datetime.fromisoformat(timestamp_str)
            
    except Exception as e1:
        try:
            # Fallback: Remove fractional seconds
            if '.' in timestamp_str:
                timestamp_str = timestamp_str.split('.')[0] + 'Z'
            return datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
        except Exception as e2:
            print(f"Failed to parse: {timestamp_str}")
            return None

def process_and_update_csv(items, device_id):
    """Process items and update CSV with current data prioritized"""
    if not items:
        print(f"No items to process for {device_id}")
        return False
    
    print(f"Processing {len(items)} items for {device_id}")
    
    rows = []
    today = datetime.datetime.now()
    
    for item in items:
        try:
            if 'payload' not in item or 'M' not in item['payload']:
                continue
                
            payload = item['payload']['M']
            
            # Extract timestamp
            received_at = item.get('received_at', {}).get('S', '')
            if not received_at:
                continue
            
            timestamp = parse_timestamp_enhanced(received_at)
            if timestamp is None:
                continue
            
            # Extract date from payload
            payload_date = payload.get('date', {}).get('S', '')
            if payload_date:
                date_formatted = payload_date.replace(':', '-')
            else:
                date_formatted = timestamp.strftime('%d-%m-%Y')
            
            # Create row
            row = {
                'device_id': device_id,
                'timestamp': timestamp,
                'DATE': date_formatted,
                'PM2.5': float(payload.get('pm25', {}).get('N', 0)),
                'PM10': float(payload.get('pm10', {}).get('N', 0)),
                'O3': float(payload.get('o3', {}).get('N', 0)),
                'CO': float(payload.get('co', {}).get('N', 0)),
                'NO2': float(payload.get('no2', {}).get('N', 0)),
                'SO2': float(payload.get('so2', {}).get('N', 0)),
                'NH3': float(payload.get('nh3', {}).get('N', 0)),
                'Temperature': float(payload.get('temp', {}).get('N', 0)),
                'Humidity': float(payload.get('hum', {}).get('N', 0)),
                'Pressure': float(payload.get('pre', {}).get('N', 0)),
                'AQI': float(payload.get('aqi', {}).get('N', 0)),
            }
            rows.append(row)
            
        except Exception as e:
            continue
    
    if not rows:
        print(f"No valid rows processed for {device_id}")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    df['SampleCount'] = 1
    df = df.sort_values('timestamp')
    df['date_only'] = df['timestamp'].dt.date
    
    # Group by date and calculate averages
    daily_averages = df.groupby('date_only').agg({
        'PM2.5': 'mean',
        'PM10': 'mean',
        'O3': 'mean',
        'CO': 'mean',
        'NO2': 'mean',
        'SO2': 'mean',
        'NH3': 'mean',
        'Temperature': 'mean',
        'Humidity': 'mean',
        'Pressure': 'mean',
        'AQI': 'mean',
        'SampleCount': 'sum'
    }).round(2)
    
    # Get the most recent 4 days
    recent_dates = sorted(daily_averages.index, reverse=True)[:4]
    daily_averages = daily_averages.loc[recent_dates].sort_index()
    
    print(f"Most recent 4 days for {device_id}:")
    for date in daily_averages.index:
        sample_count = daily_averages.loc[date, 'SampleCount']
        pm25 = daily_averages.loc[date, 'PM2.5']
        temp = daily_averages.loc[date, 'Temperature']
        is_today = date == today.date()
        today_marker = "TODAY!" if is_today else ""
        print(f"   {date.strftime('%d-%m-%Y')}: {int(sample_count)} samples, PM2.5={pm25:.2f}, Temp={temp:.1f}C{today_marker}")
    
    # Create CSV data
    csv_rows = []
    for date, row in daily_averages.iterrows():
        csv_row = {
            'DATE': date.strftime('%d-%m-%Y'),
            'PM2.5': row['PM2.5'],
            'PM10': row['PM10'],
            'O3': row['O3'],
            'CO': row['CO'],
            'NO2': row['NO2'],
            'SO2': row['SO2'],
            'NH3': row['NH3'],
            'Temperature': row['Temperature'],
            'Humidity': row['Humidity'],
            'Pressure': row['Pressure'],
            'AQI': row['AQI'],
            'SampleCount': int(row['SampleCount'])
        }
        csv_rows.append(csv_row)
    
    # Save CSV
    final_df = pd.DataFrame(csv_rows)
    filename = f'updated_real4_{device_id.replace("-", "_")}.csv'
    final_df.to_csv(filename, index=False)
    
    print(f"Updated {filename} with {len(final_df)} days of data")
    
    # Show what's in the CSV
    print(f"\n CSV Contents:")
    for _, row in final_df.iterrows():
        print(f"   {row['DATE']}: PM2.5={row['PM2.5']:.2f}, Temp={row['Temperature']:.1f}C, Samples={row['SampleCount']}")
    
    return True

def run_device_forecasting(device_id, config):
    """NEW: Run ML forecasting for a device"""
    print(f"\n Starting ML forecasting for {device_id} ({config['station_name']})")
    
    try:
        # Check if dataset exists
        if not os.path.exists(config['csv_file']):
            print(f"Dataset not found: {config['csv_file']}")
            print(f"   Skipping forecasting for {device_id}")
            return False
        
        # Show dataset info
        try:
            with open(config['csv_file'], 'r') as f:
                line_count = sum(1 for line in f) - 1  # Subtract header
            file_size = os.path.getsize(config['csv_file'])
            print(f"Dataset: {config['csv_file']} ({line_count} days, {file_size:,} bytes)")
        except Exception as e:
            print(f"Could not read dataset info: {e}")
        
        # Initialize pipeline with device-specific config
        pipeline_config = {
            'device_id': device_id,
            'station_name': config['station_name'],
            'csv_file': config['csv_file']
        }
        
        # Create and run pipeline
        pipeline = AirQualityForecastingPipeline(pipeline_config)
        pipeline.run_pipeline()
        
        # Check if forecast files were generated
        forecast_csv = f'forecast_{device_id.replace("-", "_")}.csv'
        forecast_json = f'forecast_{device_id.replace("-", "_")}.json'
        
        csv_exists = os.path.exists(forecast_csv)
        json_exists = os.path.exists(forecast_json)
        
        if csv_exists and json_exists:
            print(f"{device_id}: Forecasting completed successfully")
            print(f"Generated: {forecast_csv}")
            print(f"Generated: {forecast_json}")
            return True
        else:
            print(f"{device_id}: Forecasting completed but files missing")
            print(f"CSV exists: {csv_exists}, JSON exists: {json_exists}")
            return False
            
    except Exception as e:
        print(f"Error in forecasting for {device_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_combined_dashboard():
    """Generate combined dashboard"""
    print(f"\n Generating combined dashboard...")
    
    try:
        dashboard_data = {
            "updated_at": datetime.datetime.now().isoformat(),
            "pipeline_version": "3.0-INTEGRATED",
            "total_stations": len(DEVICES),
            "stations": {}
        }
        
        success_count = 0
        
        for device_id in DEVICES:
            config = DEVICE_CONFIGS[device_id]
            forecast_csv = f'forecast_{device_id.replace("-", "_")}.csv'
            forecast_json = f'forecast_{device_id.replace("-", "_")}.json'
            
            csv_exists = os.path.exists(forecast_csv)
            json_exists = os.path.exists(forecast_json)
            
            if csv_exists and json_exists:
                success_count += 1
                status = 'success'
            else:
                status = 'failed'
            
            station_data = {
                "station_name": config['station_name'],
                "device_id": device_id,
                "status": status,
                "csv_file": config['csv_file'],
                "forecast_days": 4 if status == 'success' else 0,
                "files_exist": {
                    "csv": csv_exists,
                    "json": json_exists
                }
            }
            
            # Try to load forecast data if available
            if status == 'success' and json_exists:
                try:
                    with open(forecast_json, 'r') as f:
                        forecast_data = json.load(f)
                        station_data.update({
                            "gases": forecast_data.get("gases", {}),
                            "dates": forecast_data.get("dates", [])
                        })
                except Exception as e:
                    print(f" Could not load forecast data for {device_id}: {e}")
            
            dashboard_data["stations"][device_id] = station_data
        
        dashboard_data["successful_stations"] = success_count
        dashboard_data["failed_stations"] = len(DEVICES) - success_count
        
        # Save combined dashboard
        with open('multi_station_dashboard.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f" Combined dashboard saved: multi_station_dashboard.json")
        print(f" Success rate: {success_count}/{len(DEVICES)} devices")
        
        return True
        
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return False

def main():
    """UPDATED: Force update all devices with current data AND run forecasting"""
    print("INTEGRATED PIPELINE - DATA UPDATE + ML FORECASTING")
    print("=" * 60)
    
    # Phase 1: Data Update (existing functionality)
    print("\n PHASE 1: DATA UPDATE")
    print("-" * 30)
    
    data_results = {}
    for device_id in DEVICES:
        print(f"\n Processing data for {device_id}...")
        
        # Force fetch recent data
        items = force_fetch_today_data(device_id)
        
        # Process and save
        if items:
            success = process_and_update_csv(items, device_id)
            if success:
                print(f" {device_id}: Data update SUCCESS")
                data_results[device_id] = True
            else:
                print(f"{device_id}: Data update FAILED")
                data_results[device_id] = False
        else:
            print(f"{device_id}: Data fetch FAILED")
            data_results[device_id] = False
    
    # Phase 2: ML Forecasting (NEW functionality)
    print("\n PHASE 2: ML FORECASTING")
    print("-" * 30)
    
    forecast_results = {}
    for device_id in DEVICES:
        config = DEVICE_CONFIGS[device_id]
        success = run_device_forecasting(device_id, config)
        forecast_results[device_id] = success
    
    # Phase 3: Combined Dashboard
    print("\n PHASE 3: DASHBOARD GENERATION")
    print("-" * 30)
    generate_combined_dashboard()
    
    # Final Summary
    print(f"\n FINAL RESULTS:")
    print("=" * 30)
    
    for device_id in DEVICES:
        data_status = " " if data_results[device_id] else ""
        forecast_status = " " if forecast_results[device_id] else ""
        station_name = DEVICE_CONFIGS[device_id]['station_name']
        
        print(f"{device_id} ({station_name}):")
        print(f"   Data Update: {data_status}")
        print(f"   ML Forecast: {forecast_status}")
    
    # Show generated files
    print(f"\n GENERATED FILES:")
    for device_id in DEVICES:
        if data_results[device_id]:
            dataset_file = f'updated_real4_{device_id.replace("-", "_")}.csv'
            print(f"{dataset_file}")
        
        if forecast_results[device_id]:
            forecast_csv = f'forecast_{device_id.replace("-", "_")}.csv'
            forecast_json = f'forecast_{device_id.replace("-", "_")}.json'
            print(f"   {forecast_csv}")
            print(f"   {forecast_json}")
    
    if os.path.exists('multi_station_dashboard.json'):
        print(f"multi_station_dashboard.json")
    
    print(f"\n INTEGRATED PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
