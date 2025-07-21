import pandas as pd
from token_utils import check_tokens
from sklearn.preprocessing import MinMaxScaler

def create_features(df):
    print("Creating Features.....")
    categorical = "Title: " + df.title + " Author: " + df.author + " Description: " + df.description + " Genres: " + df.genres 

    categorical = categorical.dropna()
    categorical = categorical[categorical.apply(check_tokens)]

    numerical = pd.DataFrame(df[['likedPercent', 'numRatings']]) 

    # scale the numerical features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numerical)

    return categorical, pd.DataFrame(scaled_data)