import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from zeugma.embeddings import EmbeddingTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from joblib import dump


def clean_string(str):
  lst = str.strip("]['").split("', '")
  res = ''
  for word in lst:
    res += word + ' '
  return res

def get_data(path):
  types = ['clickbait', 'reliable', 'political', 'unreliable', 'fake', 'conspiracy', 'bias', 'junksci', 'satire', 'rumor']

  df = pd.read_csv(path)
  df = df.drop(df[~df.type.isin(types)].index)
  for i in range(df.shape[0]):
    df.iat[i,3] = clean_string(df.iat[i,3])

  return df


if __name__ == "__main__":
  run = wandb.init(project="GDS_advanced")
  
  fnc = get_data('FNC_lemmatized.csv')
  rnc = get_data('RNC_lemmatized.csv')
  data = pd.concat([fnc, rnc])
  test_sz = int(data.shape[0]/5)
  content_train, content_test, type_train, type_test = train_test_split(data.content, data.type, test_size=test_sz, random_state=47)

  advanced_classifier = Pipeline([('glove', EmbeddingTransformer('glove')),
                                  ('scaler', StandardScaler(copy=False)),
                                  ('class', MLPClassifier(hidden_layer_sizes=(500,400,300,200,100,), alpha=0.05, random_state=47))])

  scores = cross_validate(advanced_classifier, content_train, type_train, scoring=('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'))
  accuracy = scores['test_accuracy'].mean()
  precision = scores['test_precision_weighted'].mean()
  recall = scores['test_recall_weighted'].mean()
  f1 = scores['test_f1_weighted'].mean()
  

  wandb.log({'Lemmatized': 1, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1})
  dump(advanced_classifier, 'advanced.joblib')