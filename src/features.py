"""
Script to create features for embedding
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utility import get_data, write_data, parse_io_args, check_io_args

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw data and creates features
    """
    print("Creating Features.....")
    categorical = "Title: " + df.title \
                + " Author: " + df.author \
                + " Description: " + df.description \
                + " Genres: " + df.genres
    categorical = pd.DataFrame(categorical, columns=['categorical'])

    numerical = pd.DataFrame(df[['likedPercent', 'numRatings']])

    # scale the numerical features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numerical)
    numerical = pd.DataFrame(scaled_data, columns=['scaled_percent', 'scaled_ratings'])

    data = pd.concat([categorical, numerical], axis=1)

    return data

def get_features(input_path: str, output_path: str) -> None:
    """
    Creates features for embeddings from cleaned data. Gets and uploads data from s3.
    """
    df = get_data(input_path)

    data = create_features(df)

    write_data(data, output_path)

if __name__=="__main__":
    args = parse_io_args()

    check_io_args(args)

    get_features(args.input_path, args.output_path)
