# Disaster Response Pipeline Project

## Overview
Repository for the Date Engineer Project for the Udacity Data Science Nanodegree.

This projects goal is to build a NLP model to categorize messages from disaster events.
The dataset to achieve this goal contains labeled messages real disaster events.

This project is distributed in 3 parts:
- Data preprocessing (ETL Pipeline for preparing the dataset)
- Modeling (Implementing the model for this task)
- App (Hosting a Flask App to use the trained model on new messages in real time)


## Installation

This code was implemented using python 3.8.x

Used packages are:

- Pandas
- Seaborn
- Scikit 
- Flask
- SQL Alchemy
- NLTK 
- Pickle
- Plotly
- Numpy

### Instructions for excecuting the program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database from the data directory
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves from the model directory
        `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Relevant Files

### /Data Folder
This is where the two data sources (csv files) and the python script for preprocessing the data (process_data.py) are located.

### /Model Folder
Here is the train_classifier.py script, which trains and saves the used model.

### /app Folder
run.py is used to launch the flask web application.

