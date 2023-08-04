import keybert
import pandas as pd


def keyword_extractor(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = (text_array.loc[i]).replace("['","", 1).replace("']","", 1).replace("', '", " ")
    kw_model = keybert.KeyBERT()
    for i in range(len(text_array)):
        if type(text_array.loc[i]) != float:
            # print(text_array.loc[i])
            text_array.loc[i] = kw_model.extract_keywords(text_array.loc[i])
            print(str((i/13741)*100)+"%\r")
    return text_array


tokens_with_lemmatizing_csv = pd.read_csv('./three_tokens_with_lemmatizing.csv')

extracted_key_tokens = keyword_extractor(tokens_with_lemmatizing_csv['Outcome_Description'])

extracted_key_tokens_dataframe = pd.DataFrame(extracted_key_tokens)
extracted_key_tokens_dataframe.to_csv('./four_extracted_key_tokens.csv', index=False)
