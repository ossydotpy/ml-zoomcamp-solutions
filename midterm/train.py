import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

MODEL = 'model.json'
DV = 'dv.bin'


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


# # Model Selection
rf = RandomForestClassifier(class_weight='balanced')
clf = XGBClassifier()


#@ randomforest parameters
rf_params = {
    'n_estimators': np.arange(150, 200, 20),
    'max_depth': np.arange(5, 8, 2),
    'min_samples_leaf': np.arange(1,6,1)
    }
#@ xgbclassifier parameters
xgb_params = {
    'n_estimators': np.arange(160, 220, 20),
    'max_depth': np.arange(3, 10, 2),
    'learning_rate': np.arange(0.1,1.0, 0.1)
    }
#@randomized search
rf_search = GridSearchCV(rf, rf_params, scoring='f1', verbose=1, cv=5, n_jobs=-1)
xgb_search = GridSearchCV(clf, xgb_params, scoring='f1', verbose=1, cv=5, n_jobs=-1)

searches = {
    'Random Forest': rf_search,
    'XGBClassifer': xgb_search
}

#@ custom function to compare models
def model_comparison(df, y, val, val_y):
    for search_name, search in searches.items():
        print(f'---> Tuning {search_name} parameters on training set')
        model, dv = train(df, y, search)
        y_pred = predict(val, model, dv)
        precisonRecall = precision_recall(y_pred, val_y)
        scores.append((model.best_estimator_, precisonRecall[0], precisonRecall[1]))
        
    df_scores = pd.DataFrame(scores, columns=['Model','Precision', 'Recall'])
    df_scores.index = [x for x,y in searches.items()]
    return df_scores


# ## Model Comparison
scores = []

#@ comparing the scores of xgbclassifier against randomforest on df_train
print('')
scores = model_comparison(train_df, y_train, val_df, y_val)
print(scores,'\n','---------------------> <---------------------')


# #### - Selecting XGBClassifier Model for High Recall score
# ### Best Estimator

print(f'\n--->best estimator')
best_model = scores.sort_values(by='Recall', ascending=False).iloc[0].values[0]
print(best_model,'\n')


# ## Testing the model on full train
print('--->Training full train with best model <---')
model, dv = train(full_df, y_full, best_model)


# # Exporting the model
import pickle

print('exporting the model...')

with open(DV, 'wb') as f:
    pickle.dump(dv, f)

model.save_model(MODEL)

print(f"--->Model exported as '{MODEL}' and '{DV}'")
print('---------------------> <---------------------')
