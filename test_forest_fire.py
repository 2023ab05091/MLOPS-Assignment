import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import pickle

# Sample data for testing
@pytest.fixture
def sample_data():
    data = {
        'X': np.random.randint(0, 100, (100, 10)),
        'y': np.random.randint(0, 2, 100)
    }
    return data

# Test data loading and preprocessing
def test_data_loading_and_preprocessing(sample_data):
    X = sample_data['X']
    y = sample_data['y']
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert X.dtype == np.int32
    assert y.dtype == np.int32

# Test train_test_split
def test_train_test_split(sample_data):
    X = sample_data['X']
    y = sample_data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    assert X_train.shape == (70, 10)
    assert X_test.shape == (30, 10)
    assert y_train.shape == (70,)
    assert y_test.shape == (30,)

# Test model training and evaluation
@pytest.mark.parametrize("model, params", [
    (LogisticRegression(), {'solver': ['lbfgs', 'liblinear'], 'max_iter': randint(100, 500), 'C': uniform(0.1, 10.0)}),
    (RandomForestClassifier(), {'n_estimators': randint(10, 200), 'max_depth': [None, 10, 20, 30], 'criterion': ['gini', 'entropy'], 'min_samples_split': randint(2, 10), 'min_samples_leaf': randint(1, 10)}),
    (SVC(), {'C': uniform(0.1, 10.0), 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    (KNeighborsClassifier(), {'n_neighbors': randint(3, 20), 'weights': ['uniform', 'distance']})
])
def test_model_training_and_evaluation(sample_data, model, params):
    X = sample_data['X']
    y = sample_data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=10, cv=5, scoring='accuracy', random_state=42, verbose=0)
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    assert accuracy > 0
    assert f1 > 0

# Test model saving and loading
def test_model_saving_and_loading(sample_data):
    X = sample_data['X']
    y = sample_data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    with open('test_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open('test_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    y_pred = loaded_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    assert accuracy > 0
    assert f1 > 0