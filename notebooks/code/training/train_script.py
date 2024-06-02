# | filename: train_script.py
# | code-line-numbers: true

import argparse
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import tarfile

def save_metrics(metrics, model_directory, model_name):
    metrics_filepath = Path(model_directory) / f"{model_name}_metrics.json"
    with open(metrics_filepath, 'w') as f:
        json.dump(metrics, f)

def train(
    model_directory,
    train_path,
    test_path,
    #pipeline_path,
    learning_rate=0.1,
    max_depth=3
):
    X_train = pd.read_csv(Path(train_path) / "train_clf.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    X_test = pd.read_csv(Path(test_path) / "test_clf.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    y_train = y_train.astype(int)


    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1
    }

    save_metrics(metrics, model_directory, "discount_on_release-xgboost")

    model_filepath = (Path(model_directory) / "discount_on_release-xgboost")

    model.save_model(model_filepath)
    #pkl.dump(model, open(model_filepath, 'wb'))

def train_reg(
    model_directory,
    train_reg_path,
    test_reg_path,
    learning_rate=0.05,
    min_child_weight=5,
    n_estimators=50,
    reg_alpha=0.01,
    reg_lambda=1.5
):
    X_train = pd.read_csv(Path(train_reg_path) / "train_reg.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    X_test = pd.read_csv(Path(test_reg_path) / "test_reg.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    params = {
        'learning_rate': learning_rate,
        'min_child_weight': min_child_weight,
        'n_estimators': n_estimators,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'objective': 'reg:squarederror',
    }

    model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2_score': r2
    }

    save_metrics(metrics, model_directory, "time_until_discount-xgboost")

    model_filepath = (Path(model_directory) / "time_until_discount-xgboost")

    model.save_model(model_filepath)
    #pkl.dump(model, open(model_filepath, 'wb'))

if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to
    # the entry point as script arguments.
    # default values came from a local grid search run
    print(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR')}")
    print(f"SM_CHANNEL_TRAIN: {os.environ.get('SM_CHANNEL_TRAIN_CLF')}")
    print(f"SM_CHANNEL_TEST: {os.environ.get('SM_CHANNEL_TEST_CLF')}")
    #print(f"SM_CHANNEL_PIPELINE: {os.environ.get('SM_CHANNEL_PIPELINE')}")
    
    parser = argparse.ArgumentParser()
    # for the classifier model
    parser.add_argument("--learning_rate_clf", type=float, default=0.1)
    parser.add_argument("--max_depth_clf", type=int, default=3)
    #for the regression model
    parser.add_argument("--learning_rate_reg", type=float, default=0.05)
    parser.add_argument("--min_child_weight_reg", type=int, default=5)
    parser.add_argument("--n_estimators_reg", type=int, default=50)
    parser.add_argument("--reg_alpha_reg", type=float, default=0.01)
    parser.add_argument("--reg_lambda_reg", type=float, default=1.5)
    args, _ = parser.parse_known_args()

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", {}))
    job_name = training_env.get("job_name", None) if training_env else None

    train(
        # This is the location where we need to save our model.
        # SageMaker will create a model.tar.gz file with anything
        # inside this directory when the training script finishes.
        model_directory=os.environ["SM_MODEL_DIR"],
        # SageMaker creates one channel for each one of the inputs
        # to the Training Step.
        train_path=os.environ['SM_CHANNEL_TRAIN_CLF'],
        test_path=os.environ['SM_CHANNEL_TEST_CLF'],
        learning_rate=args.learning_rate_clf,
        max_depth=args.max_depth_clf,
    )

    train_reg(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_reg_path=os.environ["SM_CHANNEL_TRAIN_REG"],
        test_reg_path=os.environ["SM_CHANNEL_TEST_REG"],
        learning_rate=args.learning_rate_reg,
        min_child_weight=args.min_child_weight_reg,
        n_estimators=args.n_estimators_reg,
        reg_alpha=args.reg_alpha_reg,
        reg_lambda=args.reg_lambda_reg
    )

    model_dir = Path(os.environ["SM_MODEL_DIR"])
    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir / "discount_on_release-xgboost", arcname="discount_on_release-xgboost")
        tar.add(model_dir / "time_until_discount-xgboost", arcname="time_until_discount-xgboost")

