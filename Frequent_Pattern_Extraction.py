import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def converter(text_array):
    for i in range(len(text_array)):
        if not pd.isna(text_array.loc[i]):
            text_array.loc[i] = ''.join((x for x in text_array.loc[i] if not x.isdigit()))
            text_array.loc[i] = text_array.loc[i].replace('[',"").replace("', .)", "").replace("('", "").replace(']',"").split(', ')
    return text_array

def convert_to_list(text_array):
    list = []
    for i in range(len(text_array)):
        if type(text_array.loc[i]) != float:
            list.append(text_array.loc[i])
        else:
            list.append('')
    return list

extracted_key_tokens = pd.read_csv('./four_extracted_key_tokens.csv')

extracted_key_tokens['Outcome_Description'] = converter(extracted_key_tokens['Outcome_Description'])

extracted_key_tokens_list = convert_to_list(extracted_key_tokens['Outcome_Description'])

te = TransactionEncoder()
te_ary = te.fit(extracted_key_tokens_list).transform(extracted_key_tokens_list)
df = pd.DataFrame(te_ary, columns=te.columns_)

aprior = apriori(df, min_support=0.004, use_colnames=True)

aprior['length'] = aprior['itemsets'].apply(lambda x: len(x))

# printing the frequntly items 
print(aprior[(aprior['support'] >= 0.05)])
aprior[(aprior['support'] >= 0.05)].to_csv('./Frequnt_Items .csv')

rules = association_rules(aprior, metric="lift", min_threshold=1.2)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift",ascending=False)

print(rules.sort_values("confidence",ascending=False))

rules.to_csv('./Association_Rules.csv')