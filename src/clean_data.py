import pandas as pd

def clean_genre(genre_str):
    genre_str = genre_str.replace("[", "")
    genre_str = genre_str.replace("]", "")
    genre_str = genre_str.replace("'", "")
    return genre_str

def clean_author(author_str):
    pos = author_str.find("(")
    return author_str[:pos]

def read_data(filepath):
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

    clean_df.to_csv("data/clean_books.csv")
    return clean_df

if __name__=="__main__":
    filepath = r"/home/ubuntu/notebooks/mlops_book_recommendation_system/data/books_1.Best_Books_Ever.csv"
    read_data(filepath=filepath)

    