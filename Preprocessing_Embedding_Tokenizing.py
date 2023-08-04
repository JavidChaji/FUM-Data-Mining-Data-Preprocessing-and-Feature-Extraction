import re
import nltk 
import pandas as pd




def outcome_description_merger(outcome, description):
    # print(type(outcome.loc[1]))
    for i in range(len(outcome)):
        if not pd.isna(outcome.loc[i]):
            if not pd.isna(description.loc[i]):
                outcome.loc[i] = outcome.loc[i].replace('[<li>',"", 1).replace('</li>]'," " + description.loc[i], 1).replace('</li>, <li>'," ")
            else:
                outcome.loc[i] = outcome.loc[i].replace('[<li>',"", 1).replace('</li>]',"", 1).replace('</li>, <li>'," ")
        else:
            if not pd.isna(description.loc[i]):
                outcome.loc[i] = description.loc[i]
    return outcome

def tokenize(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = re.sub(r"[^a-zA-Z0-9]", " ", text_array.loc[i].lower())
            text_array.loc[i] = nltk.word_tokenize(text_array.loc[i])
            
    return text_array

data_csv = pd.read_csv('./zero_UOSA_Phase0.csv')

data_csv['Outcome'] = outcome_description_merger(data_csv['Outcome'], data_csv['Description'])

data_csv = data_csv.drop(columns="Description")

data_csv = data_csv.rename(columns={'Outcome':'Outcome_Description'})

data_csv.to_csv('./ zero_point_one_UOSA_Phase0.csv')
######################################################################################################
                #Embeded Feature Ready
######################################################################################################

pure_tokens = tokenize(data_csv['Outcome_Description'])

pure_tokens_dataframe = pd.DataFrame(pure_tokens)
pure_tokens_dataframe.to_csv('./one_pure_tokens.csv', index=False)
