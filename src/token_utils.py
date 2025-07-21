import tiktoken

def count_tokens(text: str | list):
    encoding = tiktoken.encoding_for_model('text-embedding-3-small')
    num_tokens = 0

    if type(text) == str:
        num_tokens = len(encoding.encode(text))
    elif type(text) == list:
        num_tokens = sum([count_tokens(txt) for txt in text])
    else:
        raise ValueError(f"count_tokens does not support input of type {type(text)}. Please ensure your input is a string value or a list of strings.")
        
    return num_tokens 

def check_tokens(text):
    return count_tokens(text) <= 8191