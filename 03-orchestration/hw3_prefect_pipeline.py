import mlflow
import pandas as pd
import yaml

from pathlib import Path
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"


@task
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_parquet(url)
    print(f"Q3 records loaded: {len(df)}")
    return df


@task
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df["duration"].dt.total_seconds() / 60

    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    print(f"Q4 prepared size: {len(df)}")
    return df


@task
def train_model(df: pd.DataFrame):
    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]

    train_dicts = df[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer()
    X = dv.fit_transform(train_dicts)

    y = df["duration"].values
    model = LinearRegression()
    model.fit(X, y)

    print(f"Q5 intercept: {model.intercept_:.2f}")
    return dv, model


@task
def log_model(dv: DictVectorizer, model: LinearRegression):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hw3-prefect")

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    mlmodel_local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/MLmodel"
    )

    with open(mlmodel_local, "r", encoding="utf-8") as f:
        mlmodel_yaml = yaml.safe_load(f)

    size_bytes = mlmodel_yaml.get("model_size_bytes")
    print(f"Q6 model_size_bytes: {size_bytes}")
    print(f"Run ID: {run_id}")


@flow(name="hw3-prefect-flow")
def main():
    raw_df = load_data(DATA_URL)
    prepared_df = prepare_data(raw_df)
    dv, model = train_model(prepared_df)
    log_model(dv, model)


if __name__ == "__main__":
    main()