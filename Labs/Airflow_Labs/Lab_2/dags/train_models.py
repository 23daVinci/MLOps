# File: train_models.py  (DAG 2 of 3: train several models, keep the best)
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule

from src.model_development import (
    CANDIDATES,
    resolve_input_path,
    evaluate_candidate,
    select_and_save_best,
    load_model,
)

default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# schedule=None: this DAG only runs when Airflow_Lab2 (or a person) triggers it.
dag = DAG(
    dag_id="Airflow_Lab2_Train",
    default_args=default_args,
    description="Lab2 DAG 2/3: cross-validate several models, save the best one",
    schedule=None,
    catchup=False,
    tags=["example"],
    max_active_runs=1,
    is_paused_upon_creation=False,
)

resolve_input_task = PythonOperator(
    task_id="resolve_input_path",
    python_callable=resolve_input_path,
    dag=dag,
)

# Dynamic task mapping: one evaluate task instance per candidate model, run in parallel.
evaluate_task = PythonOperator.partial(
    task_id="evaluate_candidate",
    python_callable=evaluate_candidate,
    dag=dag,
).expand(
    op_args=[[resolve_input_task.output, name] for name in CANDIDATES],
)

select_best_task = PythonOperator(
    task_id="select_best_model_task",
    python_callable=select_and_save_best,
    op_args=[resolve_input_task.output, evaluate_task.output, "model.sav"],
    dag=dag,
)

verify_task = PythonOperator(
    task_id="load_model_task",
    python_callable=load_model,
    op_args=[resolve_input_task.output, "model.sav"],
    dag=dag,
)

send_email = EmailOperator(
    task_id="send_email",
    to="you@example.com",
    subject="Best model: {{ ti.xcom_pull(task_ids='select_best_model_task')['model'] }}",
    html_content=(
        "<p>Training finished.</p>"
        "<p>Best model: <b>{{ ti.xcom_pull(task_ids='select_best_model_task')['model'] }}</b><br>"
        "Cross-validation accuracy: {{ '%.4f' | format(ti.xcom_pull(task_ids='select_best_model_task')['cv_accuracy']) }}<br>"
        "Test accuracy: {{ '%.4f' | format(ti.xcom_pull(task_ids='select_best_model_task')['test_accuracy']) }}</p>"
    ),
    dag=dag,
)

# Fire-and-forget; still launch the status page if something upstream failed.
trigger_flask_task = TriggerDagRunOperator(
    task_id="my_trigger_task",
    trigger_dag_id="Airflow_Lab2_Flask",
    conf={"message": "Data from training DAG"},
    reset_dag_run=False,
    wait_for_completion=False,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

resolve_input_task >> evaluate_task >> select_best_task >> verify_task >> [trigger_flask_task, send_email]
