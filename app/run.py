import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    '''
    tokenizes text for further analysis

            Parameters:
                    text (str): text which should get tokenized
            Returns:
                    clean_tokenz (obj): list with cleaned tokens
    '''

    # remove all special characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # define Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # leammatize all tokens and append them to clean_tokens list
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def message_length_category(length):
    if length <= 50:
        return '<= 50'
    elif length <= 100:
        return '<= 100'
    elif length <= 150:
        return '<= 150'
    else:
        return '> 150'

# index webpage displays cool visuals and receives user input text for model


@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    sum_of_categories_df = df.iloc[:, 5:].astype(int).sum()
    categories = sum_of_categories_df.keys()
    values = sum_of_categories_df.values

    temp_df = df
    temp_df["message_len"] = temp_df["message"].str.len()
    message_length_categorie_list = temp_df.message_len.apply(
        message_length_category)
    temp_df["message_length_category"] = message_length_categorie_list

    categories_graph2 = temp_df["message_length_category"].unique()
    values_graph_2 = temp_df.groupby(
        "message_length_category").count().iloc[:, :1]["index"]
    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=categories,
                    y=values
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Number of Categories"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=categories_graph2,
                    y=values_graph_2
                )
            ],

            'layout': {
                'title': 'Distribution of message length (number of characters)',
                'yaxis': {
                    'title': "Number of messages with certain length"
                },
                'xaxis': {
                    'title': "Categories of message length"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
