"""

This is the script for cleaning data before feature engineering and embedding processes are applied.

"""
from io import StringIO
import requests
import pandas as pd
from src.utility import write_data, parse_io_args
CSV_URL="https://github.com/scostap/goodreads_bbe_dataset/raw/refs/heads/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv" # pylint: disable=C0301

def clean_genre(genre_str: str) -> str:
    """
    Removes unneeded characters from genre string.
    """
    genre_str = genre_str.replace("[", "")
    genre_str = genre_str.replace("]", "")
    genre_str = genre_str.replace("'", "")
    return genre_str

def clean_author(author_str: str) -> str:
    """
    Removes unneeded characters from author data.
    """
    pos = author_str.find("(")
    if pos < 0:
        return author_str
    return author_str[:pos]

def read_data() -> pd.DataFrame:
    """
    Reads the raw url which contains the CSV with book data.
    """
    try:
        response = requests.get(CSV_URL, timeout=20)
        csv_data = StringIO(response.text)
    except requests.exceptions.Timeout:
        print("The request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"An exception occured: {e}")

    return pd.read_csv(csv_data)

def clean_data(output_path: str) -> None:
    """
    Main function for cleaning raw csv book data.
    """
    df = read_data()

    print("Cleaning data.....")

    clean_df = df[["title", "author", "description", "genres", "likedPercent", "numRatings"]]
    clean_df = clean_df[clean_df['description'].notna()]
    clean_df['genres'] = clean_df['genres'].apply(clean_genre)
    clean_df = clean_df[clean_df['numRatings'] > 500]
    clean_df['author'] = clean_df['author'].apply(clean_author)
    clean_df = clean_df[clean_df.notna()]
    clean_df = clean_df.reset_index()
    clean_df = clean_df.drop(columns=['index'])

    write_data(clean_df, output_path)
    print("Finished.")

if __name__=="__main__":
    args = parse_io_args()

    if not args.output_path:
        raise ValueError("You must provide --output_path")

    clean_data(args.output_path)
