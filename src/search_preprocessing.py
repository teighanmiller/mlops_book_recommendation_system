"""
Prepares data set for quick searching
"""
import re
import pandas as pd
import nltk
import spacy
from nltk.stem import PorterStemmer
from src.utility import get_data, write_data, parse_io_args, check_io_args


def remove_special_characters(text: str):
    """
    Removes special characters from text.
    """
    return re.sub(r"[^\w\s]", "", text)


def remove_stopwords(text: str, nlp):
    """
    Removes all stopwords from text.
    """
    doc = nlp(text)
    filtered = [token.text for token in doc if not token.is_stop]
    return " ".join(filtered)


def stem_words(text: str):
    """
    Make words in text into their stem words.
    """
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])


def tokenize_text(text: str):
    """
    Turn the text into a list.
    """
    return text.split()


def tokenize_corpus(df: pd.Series):
    """
    Turn data into lists of tokens
    """
    return df.apply(tokenize_text)


def lower_text(text: str):
    """
    Turns all text to lowercase
    """
    return text.lower()


def prepare_corpus(input_file: str, output_file: str):
    """
    Create data set for search
    """
    nltk.download("punkt_tab", quiet=True)
    nlp = spacy.load("en_core_web_sm")

    print("Getting data set....")
    corpus = get_data(input_path=input_file)

    try:
        msg_corpus = corpus["categorical"]
    except Exception as e:
        raise ValueError(f"An error has occured: {e}") from e

    print("Making all text lowercase....")
    msg_corpus.apply(lower_text)

    print("Removing special characters....")
    msg_corpus.apply(remove_special_characters)

    print("Removing stopword....")
    msg_corpus.apply(lambda x: remove_stopwords(x, nlp))

    print("Stemming words....")
    msg_corpus.apply(stem_words)

    print("Tokenizing text....")
    msg_corpus.apply(tokenize_text)
    tokenized_corpus = tokenize_corpus(msg_corpus)

    print("Writing data set....")
    write_data(
        tokenized_corpus.to_frame().rename(columns={"categorical": "corpus"}),
        output_file,
    )
    print("Finished.")


if __name__ == "__main__":
    args = parse_io_args()

    check_io_args(args)

    prepare_corpus(args.input_path, args.output_path)
