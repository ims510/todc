"""
This is the first script to be used in the project.
The purpose of this script is to read the data from a treebank in a conllu format
and remove certain values from certain fields and then save the output as a dataframe.
"""
import pyconll
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def make_df(conll_file_path):
    """
    This function reads a conllu file and returns a pandas dataframe.
    """
    data = pyconll.load_from_file(conll_file_path)
    tokens =[]
    for sentence in data:
        for token in sentence:
            token_info = {
                'id': token.id,
                'form': token.form,
                'lemma': token.lemma,
                'upos': token.upos,
                'xpos': token.xpos,
                'head': token.head,
                'deprel': token.deprel,
                'deps': token.deps,
                'misc': token.misc
            }
            for key, value in token.feats.items():
                token_info[key] = value
            tokens.append(token_info)

    df = pd.DataFrame(tokens)
    return df

def remove_value_from_column(df, column_name, value, placeholder):
    """
    Removes all occurrences of a value from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column from which to remove the value.
    value: The value to be removed from the column.

    Returns:
    pd.DataFrame: The resulting DataFrame with the value removed from the specified column.
    """
    df[column_name] = df[column_name].replace(value, placeholder)
    return df

def encode_categorical_features(df):
    """
    Encodes all categorical features in a DataFrame using label encoding.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The resulting DataFrame with all categorical features encoded using label encoding.
    """
    label_encoder = LabelEncoder()
    for column in df.columns:
        if column not in ['id', 'form', 'lemma', 'head']:
            df[column] = label_encoder.fit_transform(df[column].astype(str))
    return df
    
def encode_text_features_tf_idf(df):
    """
    Encodes all text features in a DataFrame using tf-idf encoding.
    """
    NotImplemented

def encode_text_features_word2vec(df):
    """
    Encodes all text features in a DataFrame using Word2Vec embeddings.
    """
    NotImplemented

def encode_text_features_bert(df):
    """
    Encodes all text features in a DataFrame using BERT embeddings.
    """
    NotImplemented


file_path = "data/input/ro_rrt-ud-train.conllu"
original_df = make_df(file_path)
placeholder = 'REMOVED_VALUE'

new_df = remove_value_from_column(original_df, 'deprel', 'nummod', placeholder)
new_df = remove_value_from_column(new_df, 'deprel', 'amod', placeholder)
new_df = remove_value_from_column(new_df, 'deprel', 'det', placeholder)
new_df = encode_categorical_features(new_df)
new_df.to_csv("data/output/ro_rrt-ud-train_nummod-amod-det_categ.csv", index=False)