import warnings
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500, 1000],
    'multi_class': ['auto', 'ovr'],
    'random_state': [8888]
}

# Initialize the Logistic Regression model
lr = LogisticRegression()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator from GridSearchCV
best_params = grid_search.best_params_
# Print the best parameters
print(f"Best Parameters: {best_params}")

best_estimator = grid_search.best_estimator_

# Predict on the test set using the best estimator
y_pred = best_estimator.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Accuracy: {accuracy}")

# Create a new MLflow Experiment
mlflow.set_experiment("Forest Fire MLFlow")

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
# Start an MLflow run
with mlflow.start_run(run_name = run_name) as mlflow_run:
    # Log the best parameters and accuracy to MLflow
    mlflow.log_params(best_params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for Forest Fire prediction")

    # Infer the model signature
    signature = infer_signature(
        X_train, lr.predict(X_train)
    )

    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="forest_fire_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

with open('model.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
