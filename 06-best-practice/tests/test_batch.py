import pandas as pd

from datetime import datetime

from batch import prepare_data


def dt(hour, minute, second):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [dt(0, 0, 0), dt(0, 0, 0), dt(0, 0, 0)],
            "tpep_dropoff_datetime": [dt(0, 1, 0), dt(1, 0, 0), dt(2, 0, 0)],
            "PULocationID": [1.0, None, 3.0],
            "DOLocationID": [None, 2.0, 3.0],
        }
    )

    categorical = ["PULocationID", "DOLocationID"]

    df_prepared = prepare_data(df, categorical)
    print(df_prepared)

    assert len(df_prepared) == 2
    assert df_prepared["duration"].iloc[0] == 1
    assert df_prepared["duration"].iloc[1] == 60
    assert df_prepared["PULocationID"].iloc[0] == "1"
    assert df_prepared["PULocationID"].iloc[1] == "-1"
    assert df_prepared["DOLocationID"].iloc[0] == "-1"
    assert df_prepared["DOLocationID"].iloc[1] == "2"

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def dt(hour, minute, second):
    return datetime(2023, 1, 1, hour, minute, second)