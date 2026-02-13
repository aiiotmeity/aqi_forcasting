import boto3
import datetime
import pandas as pd
import json
import numpy as np
import os

# AWS Configuration
AWS_REGION = 'us-east-1'
DYNAMO_TABLE = 'ttn_aws_db'

# Device configurations
DEVICES = ['lora-v1', 'loradev2','lora-v3']

def get_today_all_data(device_id, target_date=None):
    """Get ALL 10-second interval data for today from DynamoDB"""
    
    dynamo_client = boto3.client('dynamodb', region_name=AWS_REGION)
    
    if target_date:
        today_str = target_date
        print(f" Fetching ALL data for specific date: {target_date} (device: {device_id})")
    else:
        today = datetime.datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        print(f" Fetching ALL today's 10s interval data for: {device_id}")
    
    print(f" Looking for data from: {today_str}")
    
    # Scan for today's data - handle both with and without leading space
    scan_params = {
        'TableName': DYNAMO_TABLE,
        'FilterExpression': 'device_id = :device_id AND (begins_with(received_at, :date_prefix) OR begins_with(received_at, :date_prefix_space))',
        'ExpressionAttributeValues': {
            ':device_id': {'S': device_id},
            ':date_prefix': {'S': today_str},
            ':date_prefix_space': {'S': f' {today_str}'}  # Handle leading space
        },
        'Limit': 1000
    }
    
    all_today_items = []
    try:
        response = dynamo_client.scan(**scan_params)
        all_today_items.extend(response.get('Items', []))
        
        # Handle pagination to get ALL today's data
        while 'LastEvaluatedKey' in response:
            scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = dynamo_client.scan(**scan_params)
            all_today_items.extend(response.get('Items', []))
        
        print(f" Found {len(all_today_items)} 10s-interval records for today ({today_str})")
        
        if len(all_today_items) > 0:
            # Show time range of today's data
            sample_item = all_today_items[0]
            received_at = sample_item.get('received_at', {}).get('S', '')
            print(f" Sample timestamp: {received_at}")
            print(f" Expected ~{8640} records per day (24h * 60min * 6 per min)")
        
        return all_today_items
        
    except Exception as e:
        print(f" Error fetching today's data: {e}")
        return []

def get_recent_days_data(device_id, days_back=7):
    """Get recent days data if today's data is not available"""
    print(f" Fetching recent {days_back} days data for: {device_id}")
    
    dynamo_client = boto3.client('dynamodb', region_name=AWS_REGION)
    today = datetime.datetime.now()
    
    # Get recent dates
    target_dates = []
    for i in range(days_back):
        date = today - datetime.timedelta(days=i)
        target_dates.append(date.strftime('%Y-%m-%d'))
    
    all_items = []
    for target_date in target_dates:
        scan_params = {
            'TableName': DYNAMO_TABLE,
            'FilterExpression': 'device_id = :device_id AND (begins_with(received_at, :date_prefix) OR begins_with(received_at, :date_prefix_space))',
            'ExpressionAttributeValues': {
                ':device_id': {'S': device_id},
                ':date_prefix': {'S': target_date},
                ':date_prefix_space': {'S': f' {target_date}'}  # Handle leading space
            },
            'Limit': 1000
        }
        
        try:
            response = dynamo_client.scan(**scan_params)
            date_items = response.get('Items', [])
            
            while 'LastEvaluatedKey' in response:
                scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = dynamo_client.scan(**scan_params)
                date_items.extend(response.get('Items', []))
            
            print(f"   {target_date}: {len(date_items)} records")
            all_items.extend(date_items)
            
        except Exception as e:
            print(f"   Error for {target_date}: {e}")
            continue
    
    print(f" Total recent data: {len(all_items)} records")
    return all_items

def parse_timestamp_safely(timestamp_str):
    """Parse timestamp from DynamoDB"""
    if not timestamp_str:
        return None
    
    # Remove leading/trailing whitespace
    timestamp_str = timestamp_str.strip()
    
    try:
        # Handle nanosecond format
        if timestamp_str.endswith('Z') and '.' in timestamp_str:
            parts = timestamp_str[:-1].split('.')
            if len(parts) == 2 and len(parts[1]) > 6:
                fractional = parts[1][:6].ljust(6, '0')
                timestamp_str = f"{parts[0]}.{fractional}Z"
        
        if timestamp_str.endswith('Z'):
            return datetime.datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
        else:
            return datetime.datetime.fromisoformat(timestamp_str)
            
    except Exception as e1:
        try:
            if '.' in timestamp_str:
                timestamp_str = timestamp_str.split('.')[0] + 'Z'
            return datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
        except Exception as e2:
            print(f"Could not parse timestamp: '{timestamp_str}' (Original had leading/trailing spaces)")
            return None

def calculate_daily_averages(items, device_id):
    """Calculate daily averages from 10s interval data"""
    if not items:
        print(f" No data to process for {device_id}")
        return pd.DataFrame()
    
    print(f" Processing {len(items)} 10s-interval records for {device_id}")
    
    rows = []
    processed_count = 0
    error_count = 0
    
    for item in items:
        try:
            if 'payload' not in item or 'M' not in item['payload']:
                continue
                
            payload = item['payload']['M']
            
            # Parse timestamp
            received_at = item.get('received_at', {}).get('S', '')
            if not received_at:
                continue
            
            timestamp = parse_timestamp_safely(received_at)
            if timestamp is None:
                error_count += 1
                continue
            
            # Extract date from payload
            payload_date = payload.get('date', {}).get('S', '')
            if payload_date:
                date_formatted = payload_date.replace(':', '-')
            else:
                date_formatted = timestamp.strftime('%d-%m-%Y')
            
            # Create row for each 10s reading
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
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    print(f" Processing summary: {processed_count} successful, {error_count} errors")
    
    if not rows:
        print(f" No valid data processed for {device_id}")
        return pd.DataFrame()
    
    # Create DataFrame from 10s data
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp')
    df['date_only'] = df['timestamp'].dt.date
    
    print(f" Data time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Add SampleCount column (count of 10s readings per day)
    df['SampleCount'] = 1
    
    # Calculate DAILY AVERAGES (one average per day)
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
        'SampleCount': 'sum'  # Sum of readings per day (count)
    }).round(2)
    
    # Show daily averages
    print(f"\n Daily averages calculated:")
    today = datetime.date.today()
    for date, row in daily_averages.iterrows():
        sample_count = row['SampleCount']
        pm25 = row['PM2.5']
        temp = row['Temperature']
        is_today = date == today
        today_marker = "  TODAY!" if is_today else ""
        print(f"   {date.strftime('%d-%m-%Y')}: {int(sample_count)} readings, PM2.5={pm25:.2f}, Temp={temp:.1f}C{today_marker}")
    
    return daily_averages

def load_existing_dataset(device_id):
    """Load existing cumulative dataset"""
    filename = f'updated_real4_{device_id.replace("-", "_")}.csv'
    
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            existing_df['date_parsed'] = pd.to_datetime(existing_df['DATE'], format='%d-%m-%Y').dt.date
            
            print(f" Loaded existing dataset for {device_id}: {len(existing_df)} days")
            print(f"    Date range: {existing_df['DATE'].min()} to {existing_df['DATE'].max()}")
            
            return existing_df
        except Exception as e:
            print(f" Error loading existing dataset: {e}")
            return pd.DataFrame()
    else:
        print(f" No existing dataset found for {device_id}")
        return pd.DataFrame()

def update_cumulative_dataset(device_id, new_daily_averages, existing_df):
    """Update cumulative dataset with new daily averages"""
    print(f"\n Updating cumulative dataset for {device_id}...")
    
    if new_daily_averages.empty:
        print(f" No new daily averages to add")
        return existing_df
    
    # Convert new averages to rows
    new_rows = []
    for date, row in new_daily_averages.iterrows():
        new_row = {
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
        new_rows.append(new_row)
    
    # Check what dates are new
    if not existing_df.empty:
        existing_dates = set(existing_df['date_parsed'])
        new_dates = set(new_daily_averages.index)
        
        # Only add truly new dates
        dates_to_add = new_dates - existing_dates
        
        if dates_to_add:
            print(f" Adding {len(dates_to_add)} new days: {sorted(dates_to_add)}")
            
            # Filter new_rows to only include new dates
            new_rows = [row for row in new_rows 
                       if pd.to_datetime(row['DATE'], format='%d-%m-%Y').date() in dates_to_add]
            
            # Combine existing + new
            all_rows = existing_df.drop('date_parsed', axis=1).to_dict('records') + new_rows
            
        else:
            print(f" No new dates to add - all dates already exist in dataset")
            all_rows = existing_df.drop('date_parsed', axis=1).to_dict('records')
    else:
        print(f" Creating new dataset with {len(new_rows)} days")
        all_rows = new_rows
    
    # Create final DataFrame
    final_df = pd.DataFrame(all_rows)
    
    # Sort by date
    final_df['sort_date'] = pd.to_datetime(final_df['DATE'], format='%d-%m-%Y')
    final_df = final_df.sort_values('sort_date').drop('sort_date', axis=1)
    
    return final_df

def save_dataset(device_id, final_df):
    """Save the updated cumulative dataset"""
    filename = f'updated_real4_{device_id.replace("-", "_")}.csv'
    final_df.to_csv(filename, index=False)
    
    print(f" Saved cumulative dataset: {filename}")
    print(f" Total days in dataset: {len(final_df)}")
    print(f" Date range: {final_df['DATE'].min()} to {final_df['DATE'].max()}")
    
    # Show recent entries
    print(f"\n Recent entries (last 5 days):")
    recent_entries = final_df.tail(5)
    today = datetime.date.today()
    
    for _, row in recent_entries.iterrows():
        row_date = pd.to_datetime(row['DATE'], format='%d-%m-%Y').date()
        is_today = row_date == today
        today_marker = "  TODAY!" if is_today else ""
        print(f"   {row['DATE']}: PM2.5={row['PM2.5']:.2f}, Temp={row['Temperature']:.1f}C, Readings={row['SampleCount']}{today_marker}")

def process_device_daily_update(device_id, target_date=None):
    """Process daily update for one device"""
    print(f"\n{'='*20} PROCESSING {device_id} {'='*20}")
    
    try:
        # Step 1: Try to get today's data first (or specific date)
        today_items = get_today_all_data(device_id, target_date)
        
        # If no today's data, get recent days
        if not today_items:
            print(f" No data found for target date, getting recent days...")
            all_items = get_recent_days_data(device_id, days_back=7)
        else:
            all_items = today_items
        
        if not all_items:
            print(f" No data found for {device_id}")
            return False
        
        # Step 2: Calculate daily averages from 10s data
        daily_averages = calculate_daily_averages(all_items, device_id)
        
        if daily_averages.empty:
            print(f" Could not calculate daily averages for {device_id}")
            return False
        
        # Step 3: Load existing cumulative dataset
        existing_df = load_existing_dataset(device_id)
        
        # Step 4: Update cumulative dataset
        updated_df = update_cumulative_dataset(device_id, daily_averages, existing_df)
        
        # Step 5: Save updated dataset
        save_dataset(device_id, updated_df)
        
        print(f" Successfully updated {device_id}")
        return True
        
    except Exception as e:
        print(f" Error processing {device_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for daily dataset update"""
    import sys
    
    # Check if specific date provided as argument
    target_date = None
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
        print(f" Using specific date: {target_date}")
    
    print("===== DAILY DATASET UPDATE (ONE DAY = ONE AVERAGE) =====")
    print(f" Execution time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing devices: {DEVICES}")
    print(f" Logic: Calculate daily averages from 10s interval data")
    if target_date:
        print(f"Target date: {target_date}")
    print("=" * 60)
    
    results = {}
    
    # Process each device
    for device_id in DEVICES:
        success = process_device_daily_update(device_id, target_date)
        results[device_id] = "SUCCESS" if success else "FAILED"
    
    # Summary
    print(f"\n{'='*20} UPDATE SUMMARY {'='*20}")
    success_count = 0
    
    for device_id, status in results.items():
        if status == "SUCCESS":
            success_count += 1
            print(f"{device_id}: {status}")
        else:
            print(f"{device_id}: {status}")
    
    print(f"\n INAL RESULTS:")
    print(f"Successful: {success_count}/{len(DEVICES)}")
    print(f"Failed: {len(DEVICES) - success_count}/{len(DEVICES)}")
    
    if success_count == len(DEVICES):
        print(f"\n ALL DEVICES UPDATED SUCCESSFULLY!")
        print(f" Cumulative datasets now contain daily averages")
        print(f"Ready for ML model training and forecasting")
    elif success_count > 0:
        print(f"\n PARTIAL SUCCESS: {success_count} devices updated")
    else:
        print(f"\n ALL DEVICES FAILED - Check connections and data")
    
    # Show generated files
    print(f"\n UPDATED FILES:")
    for device_id in DEVICES:
        if results[device_id] == "SUCCESS":
            filename = f'updated_real4_{device_id.replace("-", "_")}.csv'
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                with open(filename, 'r') as f:
                    line_count = sum(1 for line in f) - 1
                print(f" {filename} ({file_size:,} bytes, {line_count} days)")

if __name__ == "__main__":
    main()
