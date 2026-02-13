"""
DynamoDB Connection Test Script

This script tests the connection to DynamoDB, retrieves the most recent air quality 
sensor data, and validates that it can be processed correctly before implementing 
the full pipeline.

Usage:
    python test_dynamodb_connection.py

Author: Claude
Date: April 22, 2025
"""

import boto3
import pandas as pd
import json
from datetime import datetime, timedelta
import os

# AWS Configuration
AWS_REGION = 'us-east-1'
DYNAMO_TABLE = 'ttn_aws_db'

def test_dynamodb_connection():
    """Test connection to DynamoDB and retrieve recent items"""
    try:
        # Initialize DynamoDB client
        dynamo_client = boto3.client('dynamodb', region_name=AWS_REGION)
        
        print(f"Connecting to DynamoDB table '{DYNAMO_TABLE}' in region '{AWS_REGION}'...")
        
        # Test connection by describing the table
        table_info = dynamo_client.describe_table(TableName=DYNAMO_TABLE)
        print(f"Successfully connected to table. Item count: {table_info['Table']['ItemCount']}")
        
        # Get recent items (last 24 hours)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        # Get recent items (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()

        scan_params = {
            'TableName': DYNAMO_TABLE,
            'FilterExpression': 'received_at > :ts',
            'ExpressionAttributeValues': {
                ':ts': {'S': yesterday}
            },
            'Limit': 10  # Retrieve only 10 items for testing
        }        
        # Scan DynamoDB table
        response = dynamo_client.scan(**scan_params)
        items = response.get('Items', [])
        
        print(f"Retrieved {len(items)} items from the last 24 hours")
        
        # Return items for further processing
        return items
        
    except Exception as e:
        print(f"Error connecting to DynamoDB: {e}")
        return []

def process_sample_items(items):
    """Process and validate DynamoDB items"""
    if not items:
        print("No items to process.")
        return
    
    print("\n--- Sample Item Analysis ---")
    
    # Take the first item for detailed analysis
    first_item = items[0]
    print(f"First item structure:")
    print(json.dumps(first_item, indent=2))
    
    # Check if the item has the expected structure
    if 'payload' not in first_item or 'M' not in first_item['payload']:
        print("WARNING: Item does not have the expected structure (missing 'payload.M')")
        return
    
    # Extract payload
    payload = first_item['payload']['M']
    
    # Check required fields
    required_fields = ['date', 'pm25', 'pm10', 'o3', 'co', 'no2', 'so2', 'nh3', 'temp', 'hum', 'pre', 'aqi']
    missing_fields = [field for field in required_fields if field not in payload]
    
    if missing_fields:
        print(f"WARNING: Missing required fields in payload: {missing_fields}")
    else:
        print("All required fields are present in the payload")
    
    # Test mapping to CSV format
    try:
        # Extract date and format it
        date_str = payload.get('date', {}).get('S', '')
        if not date_str:
            print("WARNING: Date field is empty or not found")
        else:
            # Convert date format from "DD:MM:YYYY" to "DD-MM-YYYY"
            date_str = date_str.replace(':', '-')
            print(f"Formatted date: {date_str}")
        
        # Create row dict matching CSV format
        row = {
            'DATE': date_str,
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
            'SampleCount': 1,  # Assuming 1 sample per record
            'AQI': float(payload.get('aqi', {}).get('N', 0)),
        }
        
        print("\nMapped CSV row:")
        print(json.dumps(row, indent=2))
        
        # Process all items into a DataFrame
        rows = []
        for item in items:
            try:
                payload = item.get('payload', {}).get('M', {})
                date_str = payload.get('date', {}).get('S', '')
                if not date_str:
                    continue
                
                date_str = date_str.replace(':', '-')
                
                row = {
                    'DATE': date_str,
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
                    'SampleCount': 1,
                    'AQI': float(payload.get('aqi', {}).get('N', 0)),
                }
                rows.append(row)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
            
            print("\nCreated DataFrame:")
            print(f"Shape: {df.shape}")
            print(df.head())
            
            # Save to CSV for inspection
            df.to_csv('test_dynamodb_data.csv', index=False)
            print("\nSaved DataFrame to 'test_dynamodb_data.csv' for inspection")
        else:
            print("No valid rows could be processed.")
        
    except Exception as e:
        print(f"Error mapping data: {e}")

def test_csv_compatibility():
    """Test compatibility with existing CSV"""
    try:
        # Check if real4.csv exists
        if not os.path.exists('real4.csv'):
            print("\nWARNING: 'real4.csv' not found in current directory")
            return
        
        # Load the CSV
        df_orig = pd.read_csv('real4.csv')
        print("\n--- CSV Compatibility Check ---")
        print(f"Original CSV shape: {df_orig.shape}")
        print(f"Columns: {df_orig.columns.tolist()}")
        
        # Check if we created a test CSV
        if not os.path.exists('test_dynamodb_data.csv'):
            print("No test DynamoDB data CSV found for comparison")
            return
        
        # Load the test CSV
        df_test = pd.read_csv('test_dynamodb_data.csv')
        print(f"Test CSV shape: {df_test.shape}")
        print(f"Columns: {df_test.columns.tolist()}")
        
        # Compare columns
        orig_cols = set(df_orig.columns)
        test_cols = set(df_test.columns)
        
        missing_cols = orig_cols - test_cols
        extra_cols = test_cols - orig_cols
        
        if missing_cols:
            print(f"WARNING: Missing columns in test data: {missing_cols}")
        
        if extra_cols:
            print(f"NOTE: Extra columns in test data: {extra_cols}")
        
        if not missing_cols and not extra_cols:
            print("Column sets match perfectly!")
        
        # Try to combine datasets
        print("\nAttempting to combine datasets...")
        combined_df = pd.concat([df_orig, df_test.drop('timestamp', axis=1)], ignore_index=True)
        print(f"Combined shape: {combined_df.shape}")
        print("Datasets can be combined successfully")
        
    except Exception as e:
        print(f"Error checking CSV compatibility: {e}")

if __name__ == "__main__":
    print("===== DynamoDB Connection Test =====")
    # Test DynamoDB connection and retrieve items
    items = test_dynamodb_connection()
    
    if items:
        # Process and validate the items
        process_sample_items(items)
        
        # Test compatibility with existing CSV
        test_csv_compatibility()
    
    print("\n===== Test Complete =====")