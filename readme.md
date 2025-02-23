# Sentiment Analysis for Cell Phone Reviews: FastAPI Endpoints and MLflow Experiment Tracking

## About this project

This project focuses on developing a sentiment analysis model for cell phone and accessory reviews using Random Forest. It leverages MLflow for experiment tracking, including multiple experiments with varying TF-IDF and model parameters, with one model deployed to production. Performance is evaluated using metrics like recall, and F1-score. The project also includes FastAPI endpoints to interact with the review dataset, enabling exploration and analysis via Transformer models.

## Requirements:

* Python version: 3.10.16

* Environment requirements: see [requirements.txt]()

## Files description:

* [sentiment_analysis_of_reviews.ipynb](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/sentiment_analysis_of_reviews.ipynb): jupyter notebook used to develop the sentiment analysis model, track experiments with MLflow, and interact with the review dataset via FastAPI.

* [main.py](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/main.py): Python script that runs sentiment analysis experiments using MLflow for tracking, processing review data, and splitting datasets for training, validation, and testing.

* [mlflow_logging.py](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/mlflow_logging.py): Python script that processes a review dataset, trains a Random Forest model with TF-IDF features, and tracks experiments using MLflow.

* [mlruns](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/tree/main/mlruns): folder that stores the experiment logs, model artifacts, and metrics tracked by MLflow during the execution of the main script.

* [api_main.py](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/api_main.py): Python file with FastAPI script that provides multiple endpoints to retrieve, process, summarize, and analyze customer reviews using NLP techniques and machine learning models.

* [docker_image](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/tree/main/docker_image): The Docker image folder contains all necessary files (requirements, Dockerfile, main script, and utilities) to deploy the FastAPI app on GCP with MLflow logging and cloud-based JSON data handling.

* [Cell_Phones_and_Accessories_5.json](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/Cell_Phones_and_Accessories_5.json): raw dataframe used for the project. Source: [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/)

* [cleaned_reviews.csv](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/cleaned_reviews.csv): Dataset exported after a preliminary preprocessing previous to data exploration.

* [exploration_utils.py](https://github.com/luherod/sentiment_analysis_pipeline_-_mlflow_fastapi_gcp/blob/main/exploration_utils.py): Python utility file containing functions for exploring the dataframe.


## Author

Lucía Herrero Rodero.
