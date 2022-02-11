from .string_indexer import string_indexer
from .label_encoding import label_encoding


def encoding(df, features_list):
    # df = string_indexer(df)
    df = label_encoding(df, features_list)
    return df