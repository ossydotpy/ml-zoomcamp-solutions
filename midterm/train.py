import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Set up logging
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

# Reading the data
logging.info("Reading the data...")
data = pd.read_csv('Datasets/Breast_Cancer.csv')

# Data cleaning
logging.info("Data cleaning...")
df = data.copy()
df = df.rename(columns={'Reginol Node Positive': 'regional_node_positive'})
df.columns = df.columns.str.lower().str.strip(' ').str.replace(' ', '_')

categorical = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df['status'] = (df['status'] == 'dead').astype(int)

# Splitting the data 60:20:20
logging.info("Splitting the data...")
y = df['status'].values
df_full, df_test, y_full, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(df_full, y_full, test_size=0.25, stratify=y_full, random_state=1)

df_val = df_val.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_full = df_full.reset_index(drop=True)

def prepare_df(df):
    new_df = df.copy()
    new_df = feature_engineer(new_df)
    return new_df

# Feature Engineering
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
    # Classification of patient ages
    series['age_group'] = series['age'].apply(map_age_to_group)
    
    # Classification of cancer sizes
    series['size_classification'] = series['t_stage'].map(size_mapping)
    
    # Lymph node positivity rate: extent or severity of lymph node involvement in breast cancer.
    series['lymph_node_positivity_%'] = (series['regional_node_positive'] / series['regional_node_examined']) * 100
    
    return series

low_MI = ['age_group', 'a_stage', 'race', 'marital_status']

# Defining final Datasets
logging.info("Defining final datasets...")
train_df = prepare_df(df_train)
val_df = prepare_df(df_val)
test_df = prepare_df(df_test)
full_df = prepare_df(df_full)

for ds in [train_df, val_df, full_df, test_df]:
    ds.drop(columns=low_MI, inplace=True)
    del ds['status']
    assert 'status' not in list(ds.columns)

# Function to train model
def train_model(X, y, model):
    dv = DictVectorizer(sparse=False)
    X_dicts = X.to_dict(orient='records')
    X = dv.fit_transform(X_dicts)
    model.fit(X, y)
    return (model, dv)

# Function to return precision and recall
def precision_recall(actual, predicted):
    conf_matrix = confusion_matrix(actual, predicted)
    tp = conf_matrix[1, 1]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (precision, recall)

# Function making predictions
def predict(model, dv, df):
    logging.info("Making predictions...")
    x_dict = df.to_dict(orient='records')
    X = dv.transform(x_dict)
    y_pred = model.best_estimator_.predict(X)
    return y_pred

# Parameter Tuning & Model Selection
rf = RandomForestClassifier(class_weight='balanced', max_depth=7, min_samples_leaf=7, n_estimators=220)

logging.info("Training model on full_train...")
model, dv = train_model(full_df, y_full, rf)

# Exporting the model
logging.info("Exporting the model...")
with open('model.bin', 'wb') as f:
    pickle.dump((model, dv), f)
