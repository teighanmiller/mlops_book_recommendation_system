import pandas as pd
import argparse
from utility import upload_to_s3

def clean_genre(genre_str):
    genre_str = genre_str.replace("[", "")
    genre_str = genre_str.replace("]", "")
    genre_str = genre_str.replace("'", "")
    return genre_str

def clean_author(author_str):
    pos = author_str.find("(")
    return author_str[:pos]

def read_data(output_path):
    # THIS SHOULD BE REPLACED WITH A WEB DOWNLOAD
    filepath = r"/home/ubuntu/notebooks/mlops_book_recommendation_system/data/books_1.Best_Books_Ever.csv"
    df = pd.read_csv(filepath, on_bad_lines='skip')

    print("Cleaning data.....")

    clean_df = df[["title", "author", "description", "genres", "likedPercent", "numRatings"]]
    clean_df = clean_df[clean_df['description'].notna()]
    clean_df['genres'] = clean_df['genres'].apply(lambda x: clean_genre(x))
    clean_df = clean_df[clean_df['numRatings'] > 500]
    clean_df['author'] = clean_df['author'].apply(lambda x: clean_author(x))
    clean_df = clean_df[clean_df.notna()]
    clean_df = clean_df.reset_index()
    clean_df = clean_df.drop(columns=['index'])

    if output_path.startswith("s3://"):
        _, bucket, *key_parts = output_path.split("/")
        s3_key = "/".join(key_parts)
        upload_to_s3(clean_df, bucket, s3_key)
    else:
        raise ValueError("Ouput path must start with s3://")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", "-o", type=str, required=True, help="s3://bucket-name/path/to/output.csv")
    args = parser.parse_args()
    read_data(args.output_path)

    