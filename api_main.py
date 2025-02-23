
from fastapi import FastAPI, HTTPException

import pandas as pd

import mlflow_logging as func

import spacy

from transformers import pipeline



app = FastAPI()

# GET ORIGINAL REVIEW AND ITS RATING -------------------------------------------------------------------------------------------------------

@app.get('/review/original')

def root(position:int):

    """
    Retrieves the original review text and rating from the dataset based on the specified index.

    Args:

        position (int): The index of the review to retrieve.

    Returns:

        dict: A dictionary containing the review text and rating for the specified index.

    Raises:

        HTTPException: If the provided index is out of bounds.
    
    Notes:

        - Loads the dataset from a JSON file.

        - Returns a dictionary with the review text and its corresponding rating.

        - Raises a 404 error if the index is not within the valid range.
    """

    reviews_df=pd.read_json('./Cell_Phones_and_Accessories_5.json', lines =  True)
    
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
    Retrieves the processed review text and its sentiment label from the dataset based on the specified index.

    Args:

        position (int): The index of the review to retrieve.

    Returns:

        dict: A dictionary containing the processed review text and its sentiment label.

    Raises:

        HTTPException: If the provided index is out of bounds.

    Notes:

        - Loads the dataset from a JSON file.

        - Preprocesses the review text using text cleaning, lemmatization, and stopword removal.

        - Determines the sentiment label based on the review rating.

        - Raises a 404 error if the index is not within the valid range.
    """

    reviews_df=pd.read_json('./Cell_Phones_and_Accessories_5.json', lines =  True)
    
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
    Summarizes the review text from the dataset based on the specified index.

    Args:

        position (int): The index of the review to summarize.

    Returns:

        dict: A dictionary containing the summarized review text.

    Raises:

        HTTPException: If the provided index is out of bounds.

    Notes:

        - Loads the dataset from a JSON file.

        - Uses a summarization model to generate a summary of the review text.

        - Raises a 404 error if the index is not within the valid range.

        - The summary length is constrained between 2 and 20 tokens.
    """

    reviews_df=pd.read_json('./Cell_Phones_and_Accessories_5.json', lines =  True)
    
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
    Answers a question about a review based on the specified review text using a QA model.

    Args:

        position (int): The index of the review to analyze.

        question (str): The question to be answered based on the review text.

    Returns:
    
        dict: A dictionary containing the answer to the question based on the review.

    Raises:

        HTTPException: If the provided index is out of bounds.

    Notes:

        - Loads the dataset from a JSON file.

        - Uses a pre-trained question answering model (deepset/roberta-base-squad2) to extract an answer from the review text.

        - The model requires the review text and the question to generate an answer.

        - Raises a 404 error if the index is not within the valid range.
    """

    reviews_df=pd.read_json('./Cell_Phones_and_Accessories_5.json', lines =  True)
    
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

        position (int): The index of the review to evaluate.

    Returns:

        dict: A dictionary containing a customer service response based on the review sentiment.

    Raises:

        HTTPException: If the provided index is out of bounds.

    Notes:

        - Loads the dataset from a JSON file.

        - Determines the sentiment of the review based on the rating.

        - Returns a predefined customer service response depending on whether the sentiment is positive or negative.
        
        - Raises a 404 error if the index is not within the valid range.
    """
    
    reviews_df=pd.read_json('./Cell_Phones_and_Accessories_5.json', lines =  True)
    
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

