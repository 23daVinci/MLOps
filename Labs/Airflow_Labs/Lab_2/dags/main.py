# File: main.py  (DAG 1 of 3: data preparation)
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
)

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Lab2 DAG 1/3: load and preprocess data, then trigger model training",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Naman Lazarus": "https://github.com/23daVinci"},
    max_active_runs=1,
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Naman Lazarus",
    dag=dag,
)

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="separate_data_outputs_task",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

# Hand the preprocessed-data path to the training DAG through conf.
# Default trigger rule (all_success): if data prep fails, do NOT train on stale data.
trigger_train_task = TriggerDagRunOperator(
    task_id="trigger_train_dag",
    trigger_dag_id="Airflow_Lab2_Train",
    conf={"preprocessed_path": "{{ ti.xcom_pull(task_ids='separate_data_outputs_task') }}"},
    reset_dag_run=False,
    wait_for_completion=False,  # fire and forget
    dag=dag,
)

# ---------- Dependencies ----------
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> trigger_train_task
