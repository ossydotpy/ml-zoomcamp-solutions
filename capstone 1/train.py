
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler      
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV


data = pd.read_csv('datasets/smoking_driking_dataset_Ver01.csv')
# data['DRK_YN'] = (data['DRK_YN'] == 'Y').astype(int)



class Model:
    """
    This class implements a ml model pipeline for regression or classification.
    """

    def __init__(self, dataset, training_function, scaler=StandardScaler(), vectorizer=DictVectorizer(sparse=False)):
        
        self.dataset = dataset
        self.training_function = training_function
        self.vectorizer = vectorizer
        self.model = None
        self.predictions = None
        self.best_estimator = None
        self.target=None
        


        self.setup_data()
        self.Xtrain = self._prepare_dataset(self.train_data, train_data=True)
        self.Xval = self._prepare_dataset(self.val_data, train_data=False)
        self.Xfull = None
        self.Xtest = None

    def setup_data(self):
        print('intial data setup...')

        self.dataset['sex'] = (self.dataset['sex'] == 'Male').astype(int)
        self.dataset['DRK_YN'] = (self.dataset['DRK_YN'] == 'Y').astype(int)
        self.target = self.dataset['DRK_YN']
        del self.dataset['DRK_YN']

        self.dataset.columns = self.dataset.columns.str.lower().str.replace(" ", "_")


        self.train_test_data, self.test_data, self.y_full, self.y_test = train_test_split(self.dataset,self.target, test_size=0.2, random_state=1)
        self.train_data, self.val_data, self.y_train, self.y_val  = train_test_split(self.train_test_data,self.y_full, test_size=0.25, random_state=1)
        
    def _prepare_dataset(self, data, train_data=True):
        data_dict = data.to_dict(orient='records')
        # data_with_dummies = pd.get_dummies(data,columns=["sex"],prefix="sex", drop_first=True)
        
        if train_data:
            vectorized_data = self.vectorizer.fit_transform(data_dict)
            # scaled_data = self.scaler.fit_transform(data_with_dummies)
        else:
            vectorized_data = self.vectorizer.transform(data_dict)
            # scaled_data = self.scaler.transform(data_with_dummies)
        
        return vectorized_data


    def train(self, use_full_train=False):
        if use_full_train:
            print("Training with full data...")
            self.Xfull = self._prepare_dataset(self.train_test_data, train_data=True)
            self.Xtest = self._prepare_dataset(self.test_data, train_data=False)
            self.training_function.fit(self.Xfull, self.y_full)
        else:
            print("Training with split data...")
            self.training_function.fit(self.Xtrain, self.y_train)

    def predict(self, use_full_train=False):
        print("Predicting...")
        if not use_full_train:
            self.predictions = self.training_function.predict(self.Xval)
        else:
            self.predictions = self.training_function.predict(self.Xtest)

    def score_model(self, use_full_train=False):
        print("Scoring...")
        if use_full_train:
            score = accuracy_score(self.y_test, self.predictions)
        else:
            score = accuracy_score(self.y_val, self.predictions)
        return score

    def tune_random_search(self, params, scoring="roc_auc", cv=5, use_full_train=False):
        print("Tuning hyperparameters...")
        random_search = RandomizedSearchCV(
            self.training_function, params, scoring=scoring, verbose=1, cv=cv, n_jobs=-1
        )
        if use_full_train:
            self.Xfull = self._prepare_dataset(self.train_test_data,train_data=True)
            self.Xtest = self._prepare_dataset(self.test_data, train_data=False)
            random_search.fit(self.Xfull, self.y_full)
        else:
            random_search.fit(self.Xtrain, self.y_train)
            
        self.training_function = random_search.best_estimator_
        print(f"Updated training function: {self.training_function}")


if __name__=='__main__':

    xgbclf = XGBClassifier()

    xgb_params = {
        'n_estimators': np.arange(190, 250, 20),
        'max_depth': np.arange(3, 8, 2),
        'learning_rate': np.arange(0.2,0.5, 0.1)
        }


    xgb_model = Model(dataset=data, training_function=xgbclf)
    # xgb_model.tune_random_search(params=xgb_params, cv=5)
    # xgb_model.train()
    # xgb_model.predict()
    # xgb_accuracy = xgb_model.score_model()
    # print('accuracy:',xgb_accuracy)
    # print('exporting model...')
    print(xgb_model.train_data)

    # xgb_model.training_function.save_model(f'model-{xgb_accuracy:.2f}.json')

    # import pickle
    # with open (f'vectorizer-{xgb_accuracy:.2f}.bin', 'wb') as f:
    #     pickle.dump(xgb_model.vectorizer, f)
    