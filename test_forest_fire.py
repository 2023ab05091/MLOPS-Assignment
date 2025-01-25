import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import mlflow
from datetime import datetime
from mlflow.models.signature import infer_signature


class TestForestFireModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data
        cls.data = pd.read_csv("Forest_fire.csv")
        cls.data = np.array(cls.data)

        cls.X = cls.data[1:, 1:-1].astype('int')
        cls.y = cls.data[1:, -1].astype('int')

        # Split the data into training and testing sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=0
        )

        # Define the parameter grid for hyperparameter tuning
        cls.param_grid = {
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200, 500, 1000],
            'multi_class': ['auto', 'ovr'],
            'random_state': [8888]
        }

        # Initialize the Logistic Regression model
        cls.lr = LogisticRegression()

        # Initialize GridSearchCV with 5-fold cross-validation
        cls.grid_search = GridSearchCV(
            estimator=cls.lr,
            param_grid=cls.param_grid, cv=5, scoring='accuracy'
        )

        # Perform GridSearchCV to find the best hyperparameters
        cls.grid_search.fit(cls.X_train, cls.y_train)

        # Get the best parameters and best estimator from GridSearchCV
        cls.best_params = cls.grid_search.best_params_
        cls.best_estimator = cls.grid_search.best_estimator_

        # Predict on the test set using the best estimator
        cls.y_pred = cls.best_estimator.predict(cls.X_test)

        # Calculate the accuracy of the model
        cls.accuracy = accuracy_score(cls.y_test, cls.y_pred)

    def test_best_params(self):
        expected_params = {
            'solver': 'lbfgs',
            'max_iter': 100,
            'multi_class': 'auto',
            'random_state': 8888
        }
        self.assertEqual(self.best_params, expected_params)

    def test_accuracy(self):
        self.assertGreaterEqual(self.accuracy, 0.7)

    def test_mlflow_logging(self):
        mlflow.set_experiment("Forest Fire MLFlow")
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=run_name) as mlflow_run:
            mlflow.log_params(self.best_params)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.set_tag(
                "Training Info", "Basic LR model for Forest Fire prediction"
            )
            signature = infer_signature(
                self.X_train, self.lr.predict(self.X_train)
            )
            mlflow_run_id = mlflow_run.info.run_id
            model_info = mlflow.sklearn.log_model(
                sk_model=self.lr,
                artifact_path="forest_fire_model",
                signature=signature,
                input_example=self.X_train,
                registered_model_name="tracking-quickstart",
            )
            self.assertIsNotNone(mlflow_run_id)
            self.assertIsNotNone(model_info)

    def test_pickle_model(self):
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(self.lr, model_file)
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        self.assertIsInstance(model, LogisticRegression)

if __name__ == '__main__':
    unittest.main()
