
import argparse

import contractions

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

import numpy as np

import pandas as pd

import re

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import spacy
import en_core_web_sm


# TUPLE PARSER ------------------------------------------------------------------------------------------------------------------------------------------------

def parse_tuple(value):
    
    """
    Parses a comma-separated string and converts it into a tuple of integers.

    Args:
        value (str): A comma-separated string representing the values to be converted into a tuple.

    Returns:
        tuple: A tuple containing the integers from the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not in the correct format or contains non-integer values.
    """

    try:
        return tuple(map(int, value.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {value}. Use format: num1,num2")


# COMAND-LINE ARGUMENTS PARSER --------------------------------------------------------------------------------------------------------------------------------

def arguments():

    """
    Parses the command-line arguments for the application and returns the parsed arguments.

    Args:
        None

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Arguments:
        --seed (int, optional): Seed value for reproducibility. Default is 42.
        --dataset_path (str, optional): Path to the dataset file. Default is './Cell_Phones_and_Accessories_5.json'.
        --unnecessary_columns_list (list of str, optional): List of columns to exclude from the dataset. Default includes 'reviewerID', 'asin', 'reviewerName', etc.
        --review_text_column_name (str, optional): Name of the column containing the review text. Default is 'reviewText'.
        --review_rating_column_name (str, optional): Name of the column containing the review rating. Default is 'overall'.
        --additional_stopwords (list of str, optional): Additional stopwords to include. Default contains common product-related terms.
        --token_min_length (int, optional): Minimum word length for tokens. Default is 2.
        --processed_review_column_name (str, optional): Name of the column containing the processed review text. Default is 'reviews'.
        --sentiment_column_name (str, optional): Name of the column with sentiment labels. Default is 'sentiment_label'.
        --test_size_from_total (float, optional): Percentage of data to be used for testing. Default is 0.2.
        --val_size_from_train (float, optional): Percentage of training data to be used for validation. Default is 0.2.
        --experiment_name (str, required): Name of the experiment set.
        --ngram_range_list (list of tuple, required): List of tuples specifying N-gram sizes.
        --max_features_list (list of int, required): List of maximum feature counts.
        --n_estimators_list (list of int, required): List of number of trees for the Random Forest model.

    Raises:
        ArgumentParserError: If required arguments are missing or invalid.
    """

    parser = argparse.ArgumentParser(description='The __main__ of the application with input arguments.')
    parser.add_argument('--seed', 
        type=int, 
        help='Seed to ensure reproducibility.', 
        required=False, 
        default=42
    )
    parser.add_argument('--dataset_path', 
        type=str, 
        help='Path to the dataset from the current location.', 
        required=False, 
        default='./Cell_Phones_and_Accessories_5.json'
    )
    parser.add_argument('--unnecessary_columns_list', 
        nargs='+',
        type=str, 
        help='Columns whose content is different from the text used to evaluate sentiment and the reviews ratings.', 
        required=False, 
        default=['reviewerID','asin','reviewerName','helpful','summary',
                     'unixReviewTime','reviewTime']
    )
    parser.add_argument('--review_text_column_name', 
        type=str, 
        help='Name of the column with the text from which the sentiment is to be inferred.', 
        required=False, 
        default='reviewText'
    )
    parser.add_argument('--review_rating_column_name', 
        type=str, 
        help='Name of the column with the review rating.', 
        required=False, 
        default='overall'
    )
    parser.add_argument('--additional_stopwords', 
        nargs='+',
        type=str, 
        help='Additional stopwords to the common ones in the vocabulary.', 
        required=False, 
        default=['skinomi', 'zagg', 'screen', 'product', 'device', 'cell', 'phone', 
              'iphone', 'galaxy', 'charger', 'headset', 'battery', 'usb', 'samsung']
    )
    parser.add_argument('--token_min_length', 
        type=int, 
        help='Minimum word length to not be discarded.', 
        required=False, 
        default=2
    )
    parser.add_argument('--processed_review_column_name', 
        type=str, 
        help='Name of the column with the processed text.', 
        required=False, 
        default='reviews'
    )
    parser.add_argument('--sentiment_column_name', 
        type=str, 
        help='Name of the column with the sentiment labels.', 
        required=False, 
        default='sentiment_label'
    )
    parser.add_argument('--test_size_from_total', 
        type=float, 
        help='Percentage of all examples that will be used for testing.', 
        required=False, 
        default=0.2
    )
    parser.add_argument('--val_size_from_train', 
        type=float, 
        help='Percentage of test examples that will be used for validation.', 
        required=False, 
        default=0.2
    )
    parser.add_argument('--experiment_name', 
        type=str, 
        help='Name of the set of experiments to be conducted.', 
        required=True
    )
    parser.add_argument('--ngram_range_list', 
        nargs='+',
        type=parse_tuple, 
        help='List of tuples with the possible sizes of N-grams (possible N values).', 
        required=True
    )
    parser.add_argument('--max_features_list', 
        nargs='+',
        type=int, 
        help='Maximum number of represented features.', 
        required=True
    )
    parser.add_argument('--n_estimators_list', 
        nargs='+',
        type=int, 
        help='Number of trees to generate by the Random Forest model.', 
        required=True
    )

    return parser.parse_args() 


# SEED SETTER -------------------------------------------------------------------------------------------------------------------------------------------------

def set_seed(seed):

    """
    Sets the random seed for reproducibility in numpy operations.

    Args:
        seed (int): The seed value to initialize the random number generator.

    Returns:
        int: The seed value that was set.
    """

    np.random.seed(seed)
    return seed


# JSON DATASET LOADER -----------------------------------------------------------------------------------------------------------------------------------------

def json_dataset_loader(dataset_path):

    """
    Loads a JSON dataset from a specified file path.

    Args:
        dataset_path (str): The path to the JSON file to be loaded.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded JSON data.
    """
    
    return pd.read_json(dataset_path, lines =  True, nrows=50)


# TEXT PROCESSER ----------------------------------------------------------------------------------------------------------------------------------------------

def text_preprocesser(text, language_model, additional_stopwords, token_min_length):

    """
    Processes a given text by normalizing, removing stopwords, and lemmatizing tokens.

    Args:
        text (str): The input text to be processed.
        language_model (spacy.Language): A spaCy language model for tokenization and lemmatization.
        additional_stopwords (list of str): A list of additional stopwords to exclude from the text.
        token_min_length (int): The minimum length of tokens to retain.

    Returns:
        str: The processed text after normalization, lemmatization, and stopword removal.

    Processing Steps:
        - Converts text to lowercase.
        - Expands contractions.
        - Removes punctuation and non-alphabetic characters.
        - Encodes text to ASCII and removes non-ASCII characters.
        - Tokenizes and lemmatizes using the provided language model.
        - Removes stopwords except for "no" and "not".
        - Filters out short tokens and additional stopwords.
    """


    if not isinstance(text, str):
        return ""
    
    else:
        text = text.lower()

        text = contractions.fix(text)

        text = re.sub(r'[^\w\s]', ' ', text)

        text = re.sub("[^A-Za-z']+", ' ', text)

        text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')

        doc = language_model(text)

        tokens = [token.lemma_ for token in doc if not token.is_stop or token.text == "no" or token.text == "not"]
        
        processed_text=" ".join(
            token for token in tokens if len(token) >= token_min_length and token not in additional_stopwords
        )
        
        return processed_text


# SENTIMENT LABELER -------------------------------------------------------------------------------------------------------------------------------------------

def sentiment_labeler(review_rating):

    """
    Labels a review based on its rating as positive (1) or negative (0).

    Args:
        review_rating (int or str): The rating of the review, which is compared to determine sentiment.

    Returns:
        int: 1 if the review rating is greater than 3 (positive sentiment), 0 otherwise (negative sentiment).
    """

    if int(review_rating) > 3:
        return 1
    else:
        return 0


# DATAFRAME PROCESSER -----------------------------------------------------------------------------------------------------------------------------------------

def dataset_processer(reviews_dataframe, review_text_column_name,
                      review_rating_column_name,unnecessary_columns_list,
                      additional_stopwords, token_min_length, 
                      processed_review_column_name, sentiment_column_name,
                      text_preprocesser=text_preprocesser,
                      label_sentiment=sentiment_labeler):

    """
    Processes a dataset by applying text preprocessing and sentiment labeling, and removes unnecessary columns.

    Args:
        reviews_dataframe (pandas.DataFrame): The DataFrame containing the dataset to be processed.
        review_text_column_name (str): The name of the column containing the review text.
        review_rating_column_name (str): The name of the column containing the review ratings.
        unnecessary_columns_list (list of str): List of columns to be removed from the dataset after processing.
        additional_stopwords (list of str): List of additional stopwords to be removed during text preprocessing.
        token_min_length (int): The minimum length for tokens to be retained during preprocessing.
        processed_review_column_name (str): The name of the column where processed review text will be stored.
        sentiment_column_name (str): The name of the column where sentiment labels will be stored.
        text_preprocesser (function, optional): The function to apply text preprocessing. Defaults to `text_preprocesser`.
        label_sentiment (function, optional): The function to apply sentiment labeling. Defaults to `sentiment_labeler`.

    Returns:
        pandas.DataFrame: A DataFrame with processed reviews, sentiment labels, and unnecessary columns removed.
    """
    print('Starting text processing...')

    language_model=spacy.load("en_core_web_sm")

    reviews_dataframe[processed_review_column_name] = (
        reviews_dataframe[review_text_column_name]
            .apply(lambda x: text_preprocesser(
                text=x,
                language_model=language_model,
                additional_stopwords=additional_stopwords, 
                token_min_length=token_min_length
            ))
    )

    print('Starting sentiment labeling...')

    reviews_dataframe[sentiment_column_name] = (
        reviews_dataframe[review_rating_column_name]
            .apply(lambda x: label_sentiment(x))
    )

    print('Deleting unnecesary colums...')

    if review_rating_column_name !=sentiment_column_name and review_text_column_name != processed_review_column_name:

        unnecessary_columns_list.extend([review_text_column_name, review_rating_column_name])

    reviews_dataframe=reviews_dataframe.drop(unnecessary_columns_list, axis=1)

    return reviews_dataframe


# TRAIN VALIDATION AND TEST SPLITTER --------------------------------------------------------------------------------------------------------------------------

def train_val_test_splitter(reviews_dataframe, test_size_from_total,
                            val_size_from_train, sentiment_column_name, 
                            seed):
    
    """
    Splits the dataset into training, validation, and test sets based on specified proportions.

    Args:
        reviews_dataframe (pandas.DataFrame): The DataFrame containing the dataset to be split.
        test_size_from_total (float): The proportion of the dataset to be used for testing.
        val_size_from_train (float): The proportion of the training data to be used for validation.
        sentiment_column_name (str): The name of the column containing sentiment labels to stratify by.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing three DataFrames - train_df, val_df, and test_df.
    """
    print('Starting train - validation - test splitting')

    train_df, test_df = train_test_split(
        reviews_dataframe,
        test_size = test_size_from_total, 
        shuffle = True, 
        random_state = seed, 
        stratify=reviews_dataframe[sentiment_column_name].values
    )
    
    print('Test split has been created')

    train_df, val_df = train_test_split(
        train_df,
        test_size = val_size_from_train, 
        shuffle = True, 
        random_state = seed, 
        stratify=train_df[sentiment_column_name].values
    )

    print('Train and validation splits have been created.')

    return train_df, val_df, test_df


# MLFLOW TRACKER ----------------------------------------------------------------------------------------------------------------------------------------------

def mlflow_tracking(experiment_name, train_df, val_df, test_df,
                    processed_review_column_name, sentiment_column_name, seed, 
                    ngram_range_list, max_features_list, n_estimators_list):
    
    """
    Performs machine learning experiments by training and evaluating multiple models using different hyperparameters.

    Args:
        experiment_name (str): The name of the experiment to be tracked in MLflow.
        train_df (pandas.DataFrame): The training dataset.
        val_df (pandas.DataFrame): The validation dataset.
        test_df (pandas.DataFrame): The test dataset.
        processed_review_column_name (str): The name of the column containing processed review text.
        seed (int): The random seed for reproducibility.
        sentiment_column_name (str): The name of the column containing sentiment labels.
        ngram_range_list (list of tuple): List of tuples representing the n-gram ranges for the TF-IDF vectorizer.
        max_features_list (list of int): List of maximum feature counts for the TF-IDF vectorizer.
        n_estimators_list (list of int): List of the number of trees for the Random Forest classifier.

    Returns:
        None: Logs the model training results and metrics to MLflow.
    
    Notes:
        - For each combination of hyperparameters, trains a Random Forest classifier with TF-IDF features.
        - Logs classification metrics (f1-score, recall) for training, validation, and test sets.
        - Tracks hyperparameters and the trained model in MLflow.
    """

    mlflow.set_experiment(experiment_name)

    experiment_no=1
    for ngram_range in ngram_range_list:
        for max_features in max_features_list:
            for n_estimators in n_estimators_list:
                
                print(f'Running experiment {experiment_no}...')

                tf_idf_vectorizer = TfidfVectorizer(
                    ngram_range = ngram_range, 
                    max_df=0.95, 
                    max_features=max_features
                )

                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    random_state=seed
                )

                tfidf_rf_pipeline = Pipeline([
                    ('feature_nlp', tf_idf_vectorizer),
                    ('model', rf_model)
                ])

                tfidf_rf_pipeline.fit(
                    train_df[processed_review_column_name], 
                    train_df[sentiment_column_name]
                )
                
                y_train_pred = (
                    tfidf_rf_pipeline
                        .predict(train_df[processed_review_column_name])
                )
                y_val_pred = (
                    tfidf_rf_pipeline
                        .predict(val_df[processed_review_column_name])
                )
                y_test_pred = (
                    tfidf_rf_pipeline
                        .predict(test_df[processed_review_column_name])
                )

                classif_report_train = classification_report(
                    train_df[sentiment_column_name], 
                    y_train_pred, 
                    output_dict=True
                )
                classif_report_val = classification_report(
                    val_df[sentiment_column_name], 
                    y_val_pred, 
                    output_dict=True
                )
                classif_report_test = classification_report(
                    test_df[sentiment_column_name], 
                    y_test_pred, 
                    output_dict=True
                )

                
                with mlflow.start_run(run_name='Experiment_'+str(experiment_no)):
                    
                    mlflow.log_metric('weighted_avg_f1_score_train', classif_report_train['weighted avg']['f1-score'])
                    mlflow.log_metric('weighted_avg_f1_score_val', classif_report_val['weighted avg']['f1-score'])
                    mlflow.log_metric('weighted_avg_f1_score_test', classif_report_test['weighted avg']['f1-score'])

                    mlflow.log_metric('0_class_f1_score_train', classif_report_train['0']['f1-score'])
                    mlflow.log_metric('0_class_f1_score_val', classif_report_val['0']['f1-score'])
                    mlflow.log_metric('0_class_f1_score_test', classif_report_test['0']['f1-score'])

                    mlflow.log_metric('1_class_f1_score_train', classif_report_train['1']['f1-score'])
                    mlflow.log_metric('1_class_f1_score_val', classif_report_val['1']['f1-score'])
                    mlflow.log_metric('1_class_f1_score_test', classif_report_test['1']['f1-score'])

                    mlflow.log_metric('wighted_avg_recall_train', classif_report_train['weighted avg']['recall'])
                    mlflow.log_metric('wighted_avg_recall_val', classif_report_val['weighted avg']['recall'])
                    mlflow.log_metric('wighted_avg_recall_test', classif_report_test['weighted avg']['recall'])

                    mlflow.log_metric('0_class_recall_train', classif_report_train['0']['recall'])
                    mlflow.log_metric('0_class_recall_val', classif_report_val['0']['recall'])
                    mlflow.log_metric('0_class_recall_test', classif_report_test['0']['recall'])

                    mlflow.log_metric('1_class_recall_train', classif_report_train['1']['recall'])
                    mlflow.log_metric('1_class_recall_val', classif_report_val['1']['recall'])
                    mlflow.log_metric('1_class_recall_test', classif_report_test['1']['recall'])

                    mlflow.log_param('tfidf_ngram_range', ngram_range)
                    mlflow.log_param('tfidf_max_df', 0.95)
                    mlflow.log_param('tfidf_max_features', max_features)
                    mlflow.log_param('rf_n_estimators', n_estimators)
                    mlflow.log_param('rf_random_state', seed)

                    mlflow.sklearn.log_model(tfidf_rf_pipeline, 'tfidf_rf_pipeline')
                
                print(f'Experiment {experiment_no} finished')
                experiment_no += 1
