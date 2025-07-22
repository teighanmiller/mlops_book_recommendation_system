import pandas as pd
import boto3
from io import StringIO

def upload_to_s3(df: pd.DataFrame, bucket_name: str, s3_key: str):
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())