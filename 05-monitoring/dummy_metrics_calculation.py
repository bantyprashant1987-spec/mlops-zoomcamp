import datetime
import time
import random
import logging
import uuid
import py4j
import pytz
import pandas as pd
import io
import psycopg
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement  = """
drop table if exists dummy_metrics;
create table dummy_metrics (    
    #id uuid primary key,
    timestamp timestamptz,
    #metric_name text,
    #metric_value float
    value1 integer,
    value2 varchar,
    value3 float
);
"""

def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example ", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
    with psycopg.connect("host=localhost port=5432 user=postgres password=example dbname=test", autocommit=True) as conn:
        conn.execute(create_table_statement)

def calculate_dummy_metrics_postgresql(curr):
    value1 = rand.randint(0, 100)
    value2 = str(uuid.uuid4())
    value3 = rand.random() * 100
    curr.execute("INSERT INTO dummy_metrics(timestamp, value1, value2, value3) VALUES (%s, %s, %s, %s)", (datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3))



def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 user=postgres password=example dbname=test", autocommit=True) as conn:
        for i in range(0,100):
            with conn.cursor() as curr:
            
                calculate_dummy_metrics_postgresql(curr)
                new_send = datetime.datetime.now()
                seconds_elapsed = (new_send - last_send).total_seconds()
                if seconds_elapsed <  SEND_TIMEOUT:
                    time.sleep(SEND_TIMEOUT - seconds_elapsed)
                while last_send < new_send:
                    last_send = last_send + datetime.timedelta(seconds=10)
                    logging.info(f"Sent dummy metrics to PostgreSQL, value1={curr.execute('SELECT value1 FROM dummy_metrics ORDER BY timestamp DESC LIMIT 1').fetchone()[0]}, value2={curr.execute('SELECT value2 FROM dummy_metrics ORDER BY timestamp DESC LIMIT 1').fetchone()[0]}, value3={curr.execute('SELECT value3 FROM dummy_metrics ORDER BY timestamp DESC LIMIT 1').fetchone()[0]}")
                    
if __name__ == "__main__":
    main()