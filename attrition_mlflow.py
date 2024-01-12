import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
# from sklearn.tree import DecisionTreeClassifier

# MLFLOW
# Set the tracking URI to the local tracking server

def fn_mlflow(model,X_train,X_test,y_train,y_test,model_list):
    
    cwd = os.getcwd()

    # Create a subdirectory for MLflow
    # mlflow_dir = os.path.join(cwd, "mlruns")
    mlflow_dir = "mlruns"

    # Check if the directory exists and is accessible
    if os.access(mlflow_dir, os.R_OK):
        print(f"The directory {mlflow_dir} exists and is accessible.")
    else:
        print(f"The directory {mlflow_dir} does not exist or is not accessible.")
        # Create the directory if it doesn't exist
        os.makedirs(mlflow_dir, exist_ok=True)
    
    # Set the tracking URI to the MLflow directory
    # mlflow.set_tracking_uri('file://' + mlflow_dir)

    mlflow.set_tracking_uri('http://localhost:5000') 
   
   # Define the experiment name
    experiment_name = "exp_attrition"

    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # If the experiment does not exist, create it
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    else:
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        
    # Set the experiment
        
    # mlflow.set_experiment(experiment_name)

    for model in model_list:
    # Start a new MLflow run
        with mlflow.start_run(run_name=f"{model}"):
            # Define and train the model
            clf = model
            clf.fit(X_train, y_train)

            # Make predictions
            predictions = clf.predict(X_test)

            # Calculate metrics
            auc = roc_auc_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')

            # Log model
            mlflow.sklearn.log_model(clf, "model")

            # Log metrics
            mlflow.log_metric("auc", round(auc,3))
            mlflow.log_metric("accuracy", round(accuracy,3))
            mlflow.log_metric("f1", round(f1,3))

            print(f"{model} AUC: {auc}")
            print(f"{model} accuracy: {accuracy}")
            print(f"{model} F1 score: {f1}")

    # Register the model
    # model_details = mlflow.register_model(model_uri=artifact_uri,name="play_attrition")
