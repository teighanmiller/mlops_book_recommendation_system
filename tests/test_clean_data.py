"""
Pytest file for clean_data file
"""

from src.clean_data import clean_author, clean_genre


def test_clean_genre():
    """
    Testing for clean_genre functionality
    """
    assert clean_genre("") == ""
    assert clean_genre("[[[[[string]]]]]") == "string"
    assert clean_genre("'''''so'") == "so"


def test_clean_author():
    """
    Testing for clean_author functionality
    """
    assert clean_author("") == ""
    assert clean_author("alksdjf((((lllll))))") == "alksdjf"
    assert clean_author("ksdjf)") == "ksdjf)"
    assert clean_author("aksdfjs() something()") == "aksdfjs"
