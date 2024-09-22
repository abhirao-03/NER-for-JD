import pandas as pd
import nltk
import string


dirty_data = pd.read_csv('job_descriptions.csv')

punkt = string.punctuation
punkt = punkt[0:2] + punkt[3:13] + punkt[14:-1]

for i in range(len(dirty_data)):
    dirty_data.loc[i, 'job'] = dirty_data.loc[i, 'job'].lower()

    punkt_str = dirty_data.loc[i, 'job']
    test_str = punkt_str.translate(str.maketrans('', '', punkt))
    dirty_data.loc[i, 'job'] = test_str


clean_data = dirty_data.copy()
clean_data = clean_data.drop(columns=['url'])

for i in range(len(clean_data)):
    clean_data.loc[i, 'job'] = nltk.sent_tokenize(dirty_data['job'][i])


clean_data = clean_data.explode('job').reset_index(drop=False)