import nltk 
import pandas as pd


def lemmatizing(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).split("', '")
    for i in range(len(text_array)):
        if type(text_array.loc[i]) != float:
            text_array.loc[i] = [nltk.WordNetLemmatizer().lemmatize(w) for w in text_array.loc[i]]
    return text_array


tokens_without_stopwords_csv = pd.read_csv('./two_tokens_without_stopwords.csv')

tokens_with_lemmatizing = lemmatizing(tokens_without_stopwords_csv['Outcome_Description'])

tokens_with_lemmatizing_dataframe = pd.DataFrame(tokens_with_lemmatizing)
tokens_with_lemmatizing_dataframe.to_csv('./three_tokens_with_lemmatizing.csv', index=False)
