import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from utility import get_from_s3, upload_to_s3

def create_features(input_path: str, output_path: str) -> None:

    if input_path.startswith("s3://"):
        _, bucket, *key_parts = input_path.split("/")
        s3_key = "/".join(key_parts)
        df = get_from_s3(bucket, s3_key)
    else:
        raise ValueError("Input path must start with s3://")

    # Need to pass input path and file_key
    print("Creating Features.....")
    categorical = "Title: " + df.title + " Author: " + df.author + " Description: " + df.description + " Genres: " + df.genres 
    categorical = pd.DataFrame(categorical, columns=['categorical'])

    numerical = pd.DataFrame(df[['likedPercent', 'numRatings']]) 

    # scale the numerical features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numerical)
    numerical = pd.DataFrame(scaled_data, columns=['scaled_percent', 'scaled_ratings'])

    data = pd.concat([categorical, numerical], axis=1)

    if output_path.startswith("s3://"):
        _, bucket, *key_parts = output_path.split("/")
        s3_key = "/".join(key_parts)
        upload_to_s3(data, bucket, s3_key)
    else:
        raise ValueError("Output path must start with s3://")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, required=True, help="s3://bucket-name/path/to/output.csv")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="s3://bucket-name/path/to/output.csv")
    args = parser.parse_args()
    create_features(args.input_path, args.output_path)
