import pandas as pd
import boto3
from io import StringIO

def upload_to_s3(df: pd.DataFrame, bucket_name: str, s3_key: str):
    try:
        s3 = boto3.client("s3")
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
    except Exception as e:
        print(f"An exception has occured during upload: {e}")

def get_from_s3(bucket_name: str, file_key: str):
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        if status_code == 200:
            csv_content = response['Body'].read().decode('utf-8')
            return pd.read_csv(StringIO(csv_content))
        else:
            print(f"Error getting object: HTTP status code {status_code}")
            return None
    except Exception as e:
        print(f"An error ocurred: {e}")
        return None