
import mlflow_logging as func

def main():
    print('Running main...')

    args_values = func.arguments()

    seed=func.set_seed(seed=args_values.seed)

    print(f'Seed {seed} has been set.')

    reviews_df=func.json_dataset_loader(dataset_path=args_values.dataset_path)
    
    print(f'Dataset has been loaded from {args_values.dataset_path}')
    
    reviews_df = func.dataset_processer(
        reviews_dataframe = reviews_df, 
        review_text_column_name = args_values.review_text_column_name,
        review_rating_column_name = args_values.review_rating_column_name,
        unnecessary_columns_list = args_values.unnecessary_columns_list,
        additional_stopwords = args_values.additional_stopwords, 
        token_min_length = args_values.token_min_length,
        processed_review_column_name = args_values.processed_review_column_name,
        sentiment_column_name = args_values.sentiment_column_name,
        text_preprocesser = func.text_preprocesser,
        label_sentiment = func.sentiment_labeler
    )

    print('Dataset has been processed')

    train_df, val_df, test_df = func.train_val_test_splitter(
            reviews_dataframe=reviews_df, 
            test_size_from_total=args_values.test_size_from_total,
            val_size_from_train=args_values.val_size_from_train, 
            sentiment_column_name=args_values.sentiment_column_name, 
            seed=seed
        )
    
    print('Running experiments...')

    func.mlflow_tracking(
        experiment_name = args_values.experiment_name, 
        train_df = train_df, 
        val_df = val_df, 
        test_df = test_df,
        processed_review_column_name = args_values.processed_review_column_name, 
        sentiment_column_name = args_values.sentiment_column_name,
        seed=seed, 
        ngram_range_list = args_values.ngram_range_list, 
        max_features_list = args_values.max_features_list, 
        n_estimators_list = args_values.n_estimators_list
    )
    print('All experiments have run successfully.')

if __name__ == '__main__':
    main()
