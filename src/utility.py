import pandas as pd
import boto3
from io import BytesIO

def upload_to_s3(df: pd.DataFrame, bucket_name: str, s3_key: str):
    try:
        s3 = boto3.client("s3")
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
    except Exception as e:
        print(f"An exception has occured during upload: {e}")

def get_from_s3(bucket_name: str, file_key: str):
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        if status_code == 200:
            parquet_contents = BytesIO(response['Body'].read())
            return pd.read_parquet(parquet_contents)
        else:
            print(f"Error getting object: HTTP status code {status_code}")
            return None
    except Exception as e:
        print(f"An error ocurred: {e}")
        return None