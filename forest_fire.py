import warnings
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import mlflow
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore")

data = pd.read_csv("forest_fire.csv")
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

# Define models and their parameter distributions for RandomizedSearchCV
models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': randint(100, 500),  # Max iterations
            'C': uniform(0.1, 10.0)        # Regularization strength
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': randint(10, 200),       # Number of trees
            'max_depth': [None, 10, 20, 30],       # Maximum tree depth
            'criterion': ['gini', 'entropy'],      # Splitting criterion
            'min_samples_split': randint(2, 10),   # MinSamplesToSplitANode
            'min_samples_leaf': randint(1, 10)     # Minimum samples in a leaf
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': uniform(0.1, 10.0),       # Regularization parameter
            'kernel': ['linear', 'rbf'],   # Kernel type
            'gamma': ['scale', 'auto']     # Kernel coefficient
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': randint(3, 20),     # Number of neighbors
            'weights': ['uniform', 'distance']  # Weight function
        }
    }
}

# List to store model comparison results
results = []

# Iterate over each model, perform RandomizedSearchCV, and evaluate
for model_name, config in models.items():
    print(f"Training and tuning {model_name}...")

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=50,                # Number of parameter settings sampled
        cv=5,                     # 5-fold cross-validation
        scoring='accuracy',       # Use accuracy as the scoring metric
        random_state=42,
        verbose=0                 # Suppress verbose output
    )

    random_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    y_pred = best_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Best Parameters for {model_name}: {best_params}")
    print(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

    # Store results for comparison
    results.append({
        'Model': model_name,
        'Best Model': best_model,
        'Best Parameters': best_params,
        'Accuracy': accuracy,
        'F1 Score': f1
    })

# Create a DataFrame to summarize results
results_df = pd.DataFrame(results)
print("\nComparison of Models:")
print(results_df)

# Select the best model based on accuracy
best_result = max(results, key=lambda x: x['Accuracy'])
best_model_name = best_result['Model']
best_model = best_result['Best Model']
best_params = best_result['Best Parameters']
best_accuracy = best_result['Accuracy']
best_f1 = best_result['F1 Score']

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}, F1 Score: {best_f1:.4f}")

# Create a new MLflow Experiment
mlflow.set_experiment("Forest Fire MLFlow")

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
# Start an MLflow run
with mlflow.start_run(run_name=run_name) as mlflow_run:
    # Log the best parameters and accuracy to MLflow
    mlflow.log_params(best_params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag(
        "Training Info", "Basic LR model for Forest Fire prediction"
    )

    # Infer the model signature
    signature = infer_signature(
        X_train, best_model.predict(X_train)
    )

    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="forest_fire_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
