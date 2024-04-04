import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from joblib import dump



def get_tokens(str):
  return str.strip("]['").split("', '")


def make_data_binary(path):
  reliable_types = ['clickbait', 'reliable', 'political']
  all_types = ['clickbait', 'reliable', 'political', 'unreliable', 'fake', 'conspiracy', 'bias', 'junksci', 'satire', 'rumor']

  df = pd.read_csv(path)
  df = df.drop(df[~df.type.isin(all_types)].index)
  df['type'] = np.where(df['type'].isin(reliable_types), 'reliable', 'fake')

  return df
  

if __name__ == "__main__":
  #run = wandb.init(project="GDS_simple_log")

  simple_classifier = Pipeline([('cv', CountVectorizer(tokenizer=get_tokens,
                                                       lowercase=False,
                                                       token_pattern=None)),
                                ('tfidf', TfidfTransformer()),
                                ('class', LogisticRegression(max_iter=200))])
  
  fnc = make_data_binary('FNC_lemmatized.csv')
  rnc = make_data_binary('RNC_lemmatized.csv')
  data = pd.concat([fnc, rnc])
  test_sz = int(data.shape[0]/5)
  content_train, content_test, type_train, type_test = train_test_split(data.content, data.type, test_size=test_sz, random_state=47)

  #scores = cross_validate(simple_classifier, content_train, type_train, scoring=('precision_weighted', 'recall_weighted', 'f1_weighted'))
  #precision = scores['test_precision_weighted'].mean()
  #recall = scores['test_recall_weighted'].mean()
  #f1 = scores['test_f1_weighted'].mean()

  simple_classifier.fit(content_train, type_train)
  predicted = simple_classifier.predict(content_test)
  print(np.mean(predicted == type_test))

  #wandb.log({'Lemmatized': 0, 'RNC': 1, 'Precision': precision, 'Recall': recall, 'F1': f1})
  dump(simple_classifier, 'simple.joblib')