# Start with loading all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from google.cloud import bigquery

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import stem

DF_NAME = 'fulldf.csv'


def getStopWords():
    return stopwords.words('english')

def reload():
    # Construct a BigQuery client object.
    client = bigquery.Client(project='sotis-lyric-analysis')

    table_id = "sotis-lyric-analysis.kendrick_gold.all_songs"

    query_string = f"""
    SELECT
    *
    FROM `{table_id}`

    """

    dataframe = (
        client.query(query_string)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )

    dataframe.to_csv(DF_NAME)

def run(reload=False):

    if reload:
        reload()

    dataframe = pd.read_csv(DF_NAME)
    text = " ".join(word.lower().strip() for word in dataframe.word)
    # wordcloud = WordCloud(collocations=False).generate(text)
    text_tokens = word_tokenize(text)


    # stem = stem.PorterStemmer()
    ps = stem.lancaster.LancasterStemmer()

    stop_words = getStopWords()
    # tokens_without_sw = [ps.stem(word) for word in text_tokens if not word in stop_words]
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]

    filtered_sentence = (" ").join(tokens_without_sw)

    wordcloud = WordCloud(collocations=False).generate(filtered_sentence)



    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    run(reload=False)
