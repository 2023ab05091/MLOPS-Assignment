import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Create a new MLflow Experiment
mlflow.set_experiment("Forest Fire MLFlow")

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
# Start an MLflow run
with mlflow.start_run(run_name = run_name) as mlflow_run:
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for Forest Fire prediction")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

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

pickle.dump(lr, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))