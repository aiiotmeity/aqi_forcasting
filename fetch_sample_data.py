import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# AWS Configuration
AWS_REGION = 'us-east-1'
DYNAMO_TABLE = 'ttn_aws_db'

def debug_pm_values_for_device(device_id='lora-v1'):
    """Debug individual PM2.5 and PM10 values to understand the high average"""
    print(f"🔍 DEBUGGING PM VALUES FOR DEVICE: {device_id}")
    print("=" * 60)
    
    dynamo_client = boto3.client('dynamodb', region_name=AWS_REGION)
    
    # Get last few days data
    today = datetime.now()
    start_date = today - timedelta(days=4)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    scan_params = {
        'TableName': DYNAMO_TABLE,
        'FilterExpression': 'device_id = :device_id AND begins_with(received_at, :start_date)',
        'ExpressionAttributeValues': {
            ':device_id': {'S': device_id},
            ':start_date': {'S': start_date_str}
        }
    }
    
    try:
        response = dynamo_client.scan(**scan_params)
        items = response.get('Items', [])
        
        print(f"Found {len(items)} items for {device_id}")
        
        if not items:
            print("No data found!")
            return
        
        # Extract PM values
        pm25_values = []
        pm10_values = []
        timestamps = []
        all_data = []
        
        for item in items:
            try:
                payload = item.get('payload', {}).get('M', {})
                received_at = item.get('received_at', {}).get('S', '')
                
                pm25 = float(payload.get('pm25', {}).get('N', 0))
                pm10 = float(payload.get('pm10', {}).get('N', 0))
                temp = float(payload.get('temp', {}).get('N', 0))
                aqi = float(payload.get('aqi', {}).get('N', 0))
                
                pm25_values.append(pm25)
                pm10_values.append(pm10)
                timestamps.append(received_at)
                
                all_data.append({
                    'timestamp': received_at,
                    'PM2.5': pm25,
                    'PM10': pm10,
                    'Temperature': temp,
                    'AQI': aqi
                })
                
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        if not pm25_values:
            print("No valid PM2.5 values found!")
            return
        
        # Convert to arrays for analysis
        pm25_array = np.array(pm25_values)
        pm10_array = np.array(pm10_values)
        
        print(f"\n📊 PM2.5 STATISTICS:")
        print(f"Count: {len(pm25_values)}")
        print(f"Average: {np.mean(pm25_array):.2f}")
        print(f"Min: {np.min(pm25_array):.2f}")
        print(f"Max: {np.max(pm25_array):.2f}")
        print(f"Median: {np.median(pm25_array):.2f}")
        print(f"Standard Deviation: {np.std(pm25_array):.2f}")
        
        print(f"\n📊 PM10 STATISTICS:")
        print(f"Count: {len(pm10_values)}")
        print(f"Average: {np.mean(pm10_array):.2f}")
        print(f"Min: {np.min(pm10_array):.2f}")
        print(f"Max: {np.max(pm10_array):.2f}")
        print(f"Median: {np.median(pm10_array):.2f}")
        print(f"Standard Deviation: {np.std(pm10_array):.2f}")
        
        # Show value distribution
        print(f"\n📈 PM2.5 VALUE DISTRIBUTION:")
        ranges = [
            (0, 25, "Good"),
            (25, 50, "Moderate"), 
            (50, 100, "Unhealthy for Sensitive"),
            (100, 200, "Unhealthy"),
            (200, 300, "Very Unhealthy"),
            (300, float('inf'), "Hazardous")
        ]
        
        for min_val, max_val, category in ranges:
            count = np.sum((pm25_array >= min_val) & (pm25_array < max_val))
            percentage = (count / len(pm25_array)) * 100
            print(f"  {category} ({min_val}-{max_val}): {count} values ({percentage:.1f}%)")
        
        # Show first and last 10 readings
        print(f"\n🔍 FIRST 10 READINGS:")
        for i in range(min(10, len(all_data))):
            data = all_data[i]
            print(f"  {data['timestamp'][:19]} - PM2.5: {data['PM2.5']:6.2f}, PM10: {data['PM10']:6.2f}, Temp: {data['Temperature']:5.1f}°C")
        
        if len(all_data) > 10:
            print(f"\n🔍 LAST 10 READINGS:")
            for i in range(max(0, len(all_data)-10), len(all_data)):
                data = all_data[i]
                print(f"  {data['timestamp'][:19]} - PM2.5: {data['PM2.5']:6.2f}, PM10: {data['PM10']:6.2f}, Temp: {data['Temperature']:5.1f}°C")
        
        # Check for outliers
        print(f"\n⚠️  POTENTIAL OUTLIERS (PM2.5 > 150):")
        outliers = [(data['timestamp'], data['PM2.5']) for data in all_data if data['PM2.5'] > 150]
        for timestamp, value in outliers[:10]:  # Show first 10 outliers
            print(f"  {timestamp[:19]} - PM2.5: {value:.2f}")
        
        if len(outliers) > 10:
            print(f"  ... and {len(outliers) - 10} more outliers")
        
        # Check if values are realistic
        print(f"\n🤔 DATA QUALITY ANALYSIS:")
        very_high_pm25 = np.sum(pm25_array > 200)
        very_high_pm10 = np.sum(pm10_array > 200)
        
        if very_high_pm25 > 0:
            print(f"⚠️  Warning: {very_high_pm25} PM2.5 readings > 200 (very high)")
        
        if very_high_pm10 > 0:
            print(f"⚠️  Warning: {very_high_pm10} PM10 readings > 200 (very high)")
        
        # Check if values might be in wrong units
        if np.mean(pm25_array) > 100:
            print(f"\n💡 POSSIBLE ISSUES:")
            print(f"1. Values might be in different units (e.g., ppb instead of μg/m³)")
            print(f"2. Sensor might be malfunctioning")
            print(f"3. Extremely polluted environment")
            print(f"4. Data corruption during transmission")
        
        return pm25_values, pm10_values, all_data
        
    except Exception as e:
        print(f"Error debugging PM values: {e}")
        return [], [], []

def check_data_units():
    """Check if PM values might be in wrong units"""
    print(f"\n🔬 CHECKING DATA UNITS AND PATTERNS")
    print("=" * 50)
    
    pm25_values, pm10_values, all_data = debug_pm_values_for_device('lora-v1')
    
    if pm25_values and pm10_values:
        pm25_array = np.array(pm25_values)
        pm10_array = np.array(pm10_values)
        
        # Check PM2.5 vs PM10 relationship
        print(f"\n🔄 PM2.5 vs PM10 RELATIONSHIP:")
        pm25_higher = np.sum(pm25_array > pm10_array)
        total_readings = len(pm25_array)
        percentage_higher = (pm25_higher / total_readings) * 100
        
        print(f"PM2.5 > PM10 in {pm25_higher}/{total_readings} readings ({percentage_higher:.1f}%)")
        
        if percentage_higher > 20:
            print(f"⚠️  WARNING: PM2.5 should usually be < PM10")
            print(f"   This suggests possible data issues")
        
        # Check for consistent patterns
        print(f"\n📋 SAMPLE COMPARISONS:")
        for i in range(min(5, len(all_data))):
            data = all_data[i]
            ratio = data['PM2.5'] / data['PM10'] if data['PM10'] > 0 else 0
            print(f"  PM2.5: {data['PM2.5']:6.2f}, PM10: {data['PM10']:6.2f}, Ratio: {ratio:.2f}")

if __name__ == "__main__":
    debug_pm_values_for_device('lora-v1')
    check_data_units()