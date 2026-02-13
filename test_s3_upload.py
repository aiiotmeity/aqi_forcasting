"""
Test script to verify S3 uploads are working correctly
"""
import boto3
import pandas as pd
import json
import os
from datetime import datetime

def test_s3_connection():
    """Test S3 connectivity"""
    print("Testing S3 connection...")
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.list_buckets()
        print("S3 connection successful")
        return True
    except Exception as e:
        print(f"S3 connection error: {e}")
        return False

def upload_test_file():
    """Upload a test file to S3"""
    print("Creating and uploading a test file...")
    
    s3_bucket = 'ai-model-bucket-output'
    test_key = 'data/air_quality/test_file.txt'
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.put_object(
            Body=f"Test file created at {datetime.now().isoformat()}".encode(),
            Bucket=s3_bucket,
            Key=test_key
        )
        print(f"Test file uploaded to s3://{s3_bucket}/{test_key}")
        return True
    except Exception as e:
        print(f"Test upload error: {e}")
        return False

def upload_forecast_files():
    """Upload existing forecast files if they exist"""
    if not os.path.exists('latest_forecast.csv'):
        print("No forecast file found!")
        return False
    
    print("Uploading existing forecast file...")
    s3_bucket = 'ai-model-bucket-output'
    data_path = 'data/air_quality'
    
    try:
        # Read the CSV
        df = pd.read_csv('latest_forecast.csv')
        print(f"Read forecast with shape: {df.shape}")
        
        # Create S3 client
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Upload CSV
        with open('latest_forecast.csv', 'rb') as f:
            s3_client.upload_fileobj(
                f, 
                s3_bucket, 
                f"{data_path}/latest_forecast.csv"
            )
        print("CSV uploaded successfully")
        
        # Create and upload JSON version
        dashboard_data = {
            "updated_at": datetime.now().isoformat(),
            "forecast_days": len(df),
            "gases": {},
            "dates": df['Date'].tolist()
        }
        
        for column in df.columns:
            if column.startswith('Forecasted_'):
                gas_name = column.replace('Forecasted_', '')
                dashboard_data["gases"][gas_name] = {
                    "values": df[column].tolist(),
                    "unit": "μg/m³"
                }
        
        json_data = json.dumps(dashboard_data).encode()
        s3_client.put_object(
            Body=json_data,
            Bucket=s3_bucket,
            Key=f"{data_path}/latest_forecast.json",
            ContentType='application/json'
        )
        print("JSON uploaded successfully")
        
        return True
    except Exception as e:
        print(f"Error uploading forecast: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=== S3 Upload Test ===")
    
    if test_s3_connection():
        upload_test_file()
        upload_forecast_files()
    
    print("=== Test Complete ===")