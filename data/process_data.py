from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sqlalchemy import create_engine
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])



def load_data(database_filepath):
    '''
    Load data from databse
            Parameters:
                    messages_filepath (str): filepath to database
            Returns:
                    X (obj): messages column from the data
                    y (obj): dataframe including all categories
                    categorie_names (obj): list of all cateorie names
    '''

    # reads date from database
    table_name = "disaster_response"
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con=engine)

    # creating return variables
    X = df["message"].values

    y = df.iloc[:, 5:]
    y= y.astype('int')
    category_names = y.columns

    return X, y, category_names


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


def build_model():
    '''
    build the GridSearchCV Model via usage of a pipeline and defines the parameters for the moden 
            Parameters:
                    None
            Returns:
                    cv(obj): GridSearchCV model 
    '''

    # define Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # define parameters
    parameters = {
        'clf__estimator__min_samples_leaf': [50, 100, 200],
        'clf__estimator__min_samples_leaf': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the Accuracy of the model and prints the scikit classification report
            Parameters:
                    model(obj): the trained model
                    X_test(obj): test messasges
                    Y_test(obj): test categories
                    category_names: list of all categories
            Returns:
                    None
    '''
    # predict with model to be able to evaluate
    y_pred = model.predict(X_test)

    # print evaluation
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Saves model for later usage
            Parameters:
                    model(obj): the model which should be saved
                    model_filepath(str): the path where the model should be saved
            Returns:
                    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()