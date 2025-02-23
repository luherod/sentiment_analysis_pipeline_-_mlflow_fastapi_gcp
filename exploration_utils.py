import contractions

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

import re

from wordcloud import WordCloud


# PRELIMINARY PREPROCESSER -----------------------------------------------------------------------------------------------------------

def basic_preprocessing(text, language_model):

    """
    Preprocesses the given text by applying several cleaning steps and lemmatization.

    The following steps are performed:
        * Converts text to lowercase.
        * Expands contractions.
        * Removes punctuation.
        * Retains only alphabetic characters.
        * Converts text to ASCII.
        * Removes stopwords except 'no' and 'not', and lemmatizes the words.

    Args:
        text (str): The input text to be processed.
        language_model (spacy.language.Language): A spaCy language model used for lemmatization and token analysis.

    Returns:
        list: A list of processed tokens, which are lemmatized words without stopwords (except "no" and "not").
    """
    
    text = text.lower()

    text = contractions.fix(text)

    text = re.sub(r'[^\w\s]', ' ', text)

    text = re.sub("[^A-Za-z']+", ' ', text)

    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')

    doc = language_model(text)

    processed_tokens = [token.lemma_ for token in doc if not token.is_stop or token.text == "no" or token.text == "not"]
    
    return processed_tokens


# N-GRAM PLOTTER ----------------------------------------------------------------------------------------------------------------------

def plot_ngram_frequencies(ngrams_freq_most_common, ngram_type):
    
    """
    Plots a horizontal bar chart of the frequencies of the most common n-grams.

    Args:
        ngrams_freq_most_common (list of tuples): A list of tuples where each tuple contains an n-gram and its frequency.
        ngram_type (str): A string representing the type of n-gram (e.g., "unigrams", "bigrams", etc.).

    Returns:
        None: This function displays a plot and does not return any value.
    """
    
    ngrams_list_ = [str(ngram[0]) for ngram in ngrams_freq_most_common]
    ngrams_frequencies_ = [ngram[1] for ngram in ngrams_freq_most_common]
    
    
    ngrams_frequencies_, ngrams_list_ = zip(*sorted(zip(ngrams_frequencies_, ngrams_list_)))

    
    plt.figure(figsize=(10, 6))
    plt.barh(ngrams_list_, ngrams_frequencies_)
    plt.title(f'{ngram_type} Frequencies')
    plt.xlabel('Frequency')
    plt.ylabel(f'{ngram_type}')
    plt.show()


# WORDCLOUD PLOTTER -------------------------------------------------------------------------------------------------------------------

def plot_word_cloud(text):

    """
    Generates and displays a word cloud from the given text.

    Args:
        text (str): The input text used to generate the word cloud.

    Returns:
        None: This function displays a word cloud plot and does not return any value.
    """

    wordcloud = WordCloud(
		    max_font_size=50, 
		    max_words=50, 
		    background_color="white"
		).generate(text)
    
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# TSNE PLOTTER ------------------------------------------------------------------------------------------------------------------------

def tsne_plot_similar_words(labels, embedding_clusters_2d, word_clusters, a=0.7):

    """
    Plots a t-SNE visualization of word clusters in 2D space.

    Args:
        labels (list): A list of labels representing different word clusters.
        embedding_clusters_2d (numpy.ndarray): A 2D array where each row represents a word's 2D embedding.
        word_clusters (list of lists): A list of word clusters, each containing words corresponding to a cluster.
        a (float, optional): The alpha value for scatter plot transparency. Defaults to 0.7.

    Returns:
        None: This function displays a t-SNE plot and does not return any value.
    """

    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    for label, embeddings, words, color in zip(labels, embedding_clusters_2d, word_clusters, colors):
        x = embeddings[:,0]
        y = embeddings[:,1]
        
        plt.scatter(x, y, c=[color], alpha=a, label=label)
        
        for i, word in enumerate(words):
            plt.annotate(
		            word, 
		            alpha=0.5, 
		            xy=(x[i], y[i]), 
		            xytext=(5, 2),
		            textcoords='offset points', 
		            ha='right', 
		            va='bottom', 
		            size=8
		        )
    
    plt.legend(loc=4)
    plt.grid(True)
    plt.title('Representación en 2D de los embeddings de algunos clusters de palabras')   
    plt.show()

    
# ADDITIONAL PREPROCESSER -----------------------------------------------------------------------------------------------------------

def additional_preprocessing(tokens, stopwords, token_min_length):
    
    """
    Further processes the list of tokens by filtering out stopwords and tokens shorter than the specified length.

    Args:
        tokens (list): A list of tokens to be processed.
        stopwords (list): A list of stopwords to be removed from the tokens.
        token_min_length (int): The minimum length for a token to be retained.

    Returns:
        str: The processed text as a string with tokens longer than the specified length and without the stopwords.
    """
    
    processed_text=" ".join(
        token for token in tokens if len(token) >= token_min_length and token not in stopwords
    )
    
    return processed_text