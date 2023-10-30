from math import nan
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

MODEL = 'model.bin'

# # Reading the data
data  = pd.read_csv('Datasets/Breast_Cancer.csv')

df = data.copy()
df = df.rename(columns={'Reginol Node Positive':'regional_node_positive'})
df.columns = df.columns.str.lower().str.strip(' ').str.replace(' ', '_')

categorical = list(df.dtypes[df.dtypes=='object'].index)
numerical = list(df.dtypes[df.dtypes!='object'].index)

for col in categorical:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
df['status'] = (df['status']=='dead').astype(int)

categorical.remove('status')


y = df['status'].values
df_full, df_test, y_full, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(df_full, y_full, test_size=0.25, stratify=y_full, random_state=1)

df_val = df_val.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_full = df_full.reset_index(drop=True)

del df_train['status']
del df_val['status']

for i in [df_train, df_val]:
    assert 'status' not in list(i.columns)



#@ Feature Engineering
size_mapping = {
    't1': 'Small',
    't2': 'Medium',
    't3': 'Medium',
    't4': 'Large'
}
    
age_group = {
    (0, 30): 'young',
    (30, 59): 'middle_aged',
    (59, 150): 'senior'
}

def map_age_to_group(age):
    for age_range, group in age_group.items():
        if age_range[0] <= age <= age_range[1]:
            return group
    return 'unknown'

def feature_engineer(series):
# classification of patient ages
    series['age_group'] = series['age'].apply(map_age_to_group)
    
# classification of cancer sizes
    series['size_classification'] = series['t_stage'].map(size_mapping)
    
# lymph node positivity rate: extent or severity of lymph node involvement in breast cancer.
    series['lymph_node_positivity_%'] = (series['regional_node_positive'] / series['regional_node_examined'])*100
    
    return series


full_df = feature_engineer(df_full)

categorical = list(df_full.dtypes[df_full.dtypes=='object'].index)
numerical = list(df_full.dtypes[df_full.dtypes!='object'].index)


#@ Baseline Features
base_features = sorted([
    'regional_node_examined',
    'regional_node_positive',
    'survival_months',
    'lymph_node_positivity_%',
    'race',
    'marital_status',
    't_stage',
    'n_stage',
    '6th_stage',
    'differentiate',
    'grade',
    'a_stage',
    'estrogen_status',
    'progesterone_status',
    'age_group',
    'size_classification'
])


def prepare_df(df):
    new_df = df.copy()
    new_df = feature_engineer(new_df)
    new_df = new_df[base_features]
    
    return new_df


train_df = prepare_df(df_train)
val_df = prepare_df(df_val)
test_df = prepare_df(df_test)
full_df = prepare_df(df_full)


#@ function to train model
def train(X, y, model):
    dv = DictVectorizer(sparse=False)
    
    X_dicts = X.to_dict(orient='records')
    X = dv.fit_transform(X_dicts)
   
    model.fit(X, y)
    
    return (model, dv)


#@ function to return precision and recall
def precision_recall(actual, predicted):
    conf_matrix = confusion_matrix(actual, predicted)
    tp = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    tn = conf_matrix[1, 1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (precision, recall)

#! function making predictions
def predict(df, model, dv):
    x_dict = df.to_dict(orient='records')
    X = dv.transform(x_dict)
    
    y_pred = model.best_estimator_.predict(X)
    
    return y_pred

clf = RandomForestClassifier(class_weight='balanced', max_depth=7, min_samples_leaf=2, n_estimators=150)

print('---> Training on train_df...')
clf_model, dv = train(train_df, y_train, clf)
print('---> Finished Training')
print('---> Validating model on val_df...')

val_dict = val_df.to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = clf_model.predict(X_val)

print('Precision, Recall: ')
print(precision_recall(y_val, y_pred))


# ## Testing the model on full train
print('--->Training full train with best model <---')
model, dv = train(full_df, y_full, clf_model)


# # Exporting the model
import pickle

print('exporting the model...')

with open('model.bin', 'wb') as f:
    pickle.dump((model, dv), f)
# model.save_model(MODEL)

print(f"--->Model exported as '{MODEL}'")
print('---------------------> <---------------------')


