# Disaster-Response-Pipelines
# Project Motivation
This project (Disaster Response Pipelines) is part of Udacity Data Scientists Nanodegree Program.
In this project, we will build a model to classify messages that are sent during disasters. There are 36 pre-defined categories, and examples of these categories include (related	request,	offer,	aid_related	,medical_help	,medical_products etc...).
This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task and finally web app to represent results. 

# Libraries
1-pandas

2-numpy

3-sqlalchemy

4-re

5-nltk

6-sklearn

# Project Files
# 1- /Data
contain preparation processing for data and storet atDB table.

command line : python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

# 2- /Models
contain the machine learning pipeline , trainning model , scoring and save result as pickle file

command line : python train_classifier.py ../data/DisasterResponse.db classifier.pkl

# 3- App

contain internet Web app for result and 3 visualizations for result .

command line : python run.py

env|grep WORK

Web link: https://view6914b2f4-3001.udacity-student-workspaces.com/

# Row Code 
    # ETL Pipeline Preparation
    https://github.com/Nahla-shaban-salem/Disaster-Response-Pipelines/blob/main/ETL%20Pipeline%20Preparation%20.ipynb
    # ML Pipeline Preparation
    https://github.com/Nahla-shaban-salem/Disaster-Response-Pipelines/blob/main/ML%20Pipeline%20Preparation%20.ipynb
    
    
