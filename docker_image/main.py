
from fastapi import FastAPI, HTTPException

from google.cloud import storage

import json

import pandas as pd

import mlflow_logging as func

import spacy

from transformers import pipeline



app = FastAPI()

storage_client = storage.Client()

bucket_name = 'review_service_bucket'
file_name = 'Cell_Phones_and_Accessories_5.json'


# IMPORT JSON FROM CLOUD STORAGE -----------------------------------------------------------------------------------------------------------

def load_data_from_gcs(bucket_name, file_name):
    
    """
    Loads data from a Google Cloud Storage bucket and returns it as a pandas DataFrame.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        file_name (str): The name of the file in the bucket to load.

    Returns:
        pandas.DataFrame: The data from the file as a DataFrame.

    Raises:
        google.cloud.exceptions.NotFound: If the specified file or bucket does not exist.
        ValueError: If the file content cannot be parsed as JSON.
    """

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file_name)

    data = blob.download_as_text()

    return pd.read_json(data, lines=True)


# GET ORIGINAL REVIEW AND ITS RATING -------------------------------------------------------------------------------------------------------

@app.get('/review/original')

def root(position:int):

    """
    Retrieves the original review and its rating based on the provided index.

    Args:
        position (int): The index of the review to retrieve.

    Returns:
        dict: A dictionary containing the review text and its corresponding rating.

    Raises:
        HTTPException: If the provided index is out of range (not between 0 and the number of reviews).

    """

    reviews_df = load_data_from_gcs(bucket_name, file_name)

    if position >= len(reviews_df):
        raise HTTPException(status_code=404, detail=f"The index {position} is not valid, it must be between 0 and {len(reviews_df) - 1}.")
    
    else:
        result={
            str(position):{
                'reviewText': reviews_df.iloc[position]['reviewText'],
                'overall': int(reviews_df.iloc[position]['overall'])
            }
        }
        return result


# GET PRECESSED REVIEW AND ITS SENTIMENT ---------------------------------------------------------------------------------------------------

@app.get('/review/processed')

def root(position:int):

    """
    Retrieves the processed review text and its sentiment label based on the provided index.

    Args:
        position (int): The index of the review to retrieve and process.

    Returns:
        dict: A dictionary containing the processed review text and its sentiment label.

    Raises:
        HTTPException: If the provided index is out of range (not between 0 and the number of reviews).

    """

    reviews_df = load_data_from_gcs(bucket_name, file_name)
    
    if position >= len(reviews_df):
        raise HTTPException(status_code=404, detail=f"The index {position} is not valid, it must be between 0 and {len(reviews_df) - 1}.")
    
    else:
        
        language_model=spacy.load("en_core_web_sm")

        processed_text = func.text_preprocesser(
            reviews_df.iloc[position]['reviewText'], 

            language_model=language_model,

            additional_stopwords=['skinomi', 'zagg', 'screen', 'product', 
                                  'device', 'cell', 'phone', 'iphone', 'galaxy', 
                                  'charger', 'headset', 'battery', 'usb', 
                                  'samsung'], 
            token_min_length=2
        )

        sentiment=func.sentiment_labeler(reviews_df.iloc[position]['overall'])

        result={
            str(position):{
                'review_processed': processed_text,
                'sentiment_label': sentiment
            }
        }
        return result


# GET THE SUMMARY OF THE REVIEW ------------------------------------------------------------------------------------------------------------

@app.get('/review/summary')

def root(position:int):

    """
    Retrieves a summarized version of the review text based on the provided index.

    Args:
        position (int): The index of the review to summarize.

    Returns:
        dict: A dictionary containing the summarized review text.

    Raises:
        HTTPException: If the provided index is out of range (not between 0 and the number of reviews).

    """

    reviews_df = load_data_from_gcs(bucket_name, file_name)
    
    if position >= len(reviews_df):
        raise HTTPException(status_code=404, detail=f"The index {position} is not valid, it must be between 0 and {len(reviews_df) - 1}.")
    
    else:
        summarizer = pipeline("summarization")
        review_summary = summarizer(
            reviews_df.iloc[position]['reviewText'], 
            min_length=2, 
            max_length=20
        )
        result={'review_summary': review_summary,}
        
        return result


# SOLVE DOUBTS ABOUT THE REVIEW ------------------------------------------------------------------------------------------------------------

@app.get('/review/review_doubts_solver')

def root(position:int, question:str):

    """
    Answers a specific question related to the review text based on the provided index.

    Args:
        position (int): The index of the review to analyze.
        question (str): The question to be answered using the review text.

    Returns:
        dict: A dictionary containing the answer to the provided question based on the review.

    Raises:
        HTTPException: If the provided index is out of range (not between 0 and the number of reviews).
    
    """

    reviews_df = load_data_from_gcs(bucket_name, file_name)
    
    if position >= len(reviews_df):
        raise HTTPException(status_code=404, detail=f"The index {position} is not valid, it must be between 0 and {len(reviews_df) - 1}.")
    
    else:
        oracle = pipeline(model="deepset/roberta-base-squad2")
        answer = oracle(
            question=question, 
            context=reviews_df.iloc[position]['reviewText']
        )

        result={'review_doubt_answer': answer}
        
        return result


# CUSTOMER SERVICE ANSWER TO THE REVIEW ----------------------------------------------------------------------------------------------------

@app.get('/review/customer_service_answer')

def root(position:int):

    """
    Provides a customer service response based on the sentiment of the review rating.

    Args:
        position (int): The index of the review to analyze.

    Returns:
        dict: A dictionary containing the customer service response based on the review sentiment.

    Raises:
        HTTPException: If the provided index is out of range (not between 0 and the number of reviews).

    """
    
    reviews_df = load_data_from_gcs(bucket_name, file_name)
    
    if position >= len(reviews_df):
        raise HTTPException(status_code=404, detail=f"The index {position} is not valid, it must be between 0 and {len(reviews_df) - 1}.")
    
    else:
        sentiment=func.sentiment_labeler(reviews_df.iloc[position]['overall'])

        if sentiment==0:
            answer = "We're sorry the product didn't meet your expectations"
        else:
            answer = "We're delighted that you liked the product! Thank you for your support!"

        result={'customer_service_answer': answer}
        return result

