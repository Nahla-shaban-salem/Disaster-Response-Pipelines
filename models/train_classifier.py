# import libraries
import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download(['wordnet', 'punkt', 'stopwords'])


def load_data(database_filepath, table_name='InsertTableName'):
    """
    INPUT      : Takes a datafile name and relative path as parameter
        OUTPUT     : Pandas dataframe
       
        PROCESSING : 1. create a Database connection object    
                     2. Read SQL table into a pandas dataframe
                     3. retrun the dataframe
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name,con=engine)
    X = df['message']   # Target
    Y =df.iloc[:,4:] #features
    
    return X,Y,Y.columns


stop_words = stopwords.words("english")# Remove stop words# Lemmatization
lemmatizer = WordNetLemmatizer()
def tokenize(text):
    """
    INPUT   : Takes a sentences  
    OUTPUT  : list of tokens from the sentence 
       
    PROCESSING : parse the sentence with nltk word tokeniser
                 parse again the wordnet lemmatize
                 remove stop words
                 make the tokens into lower case and remove spaces
                 return the cleaned tokes as a list
    """
    # normalize case and remove punctuation
    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    # tokenize text
    
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    define pipeline useing :1-CountVectorizer
                            2-TfidfTransformer
                            3-MultiOutputClassifier-----AdaBoostClassifier
    define parameters
    define cv useing :GridSearchCV
    """
     
    pipeline = pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(AdaBoostClassifier()))
            ])
    parameters = {'clf__estimator__n_estimators':[10,80],
             'vect__max_df':[ 1.0,0.7]}
    cv = GridSearchCV(pipeline,param_grid=parameters, n_jobs=5, verbose=2)   
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
     INPUT : model object, X_test , Y_test and  category names
     Output: reult of classification_report
     PROCESSING:
         1 - predict using model for test data 
         2 - use and print result from classification_report
    """
    y_pred = model.predict(X_test)
    for i , category_names in enumerate(Y_test):
        print('Column name: ',category_names)
        print(classification_report(Y_test[category_names],y_pred[:,i]))


def save_model(model, model_filepath):
    """
    INPUT:model varible and path for save
    Output:----
    PROCESSING: save model as pickle file
    """
    pickle.dump( model, open( model_filepath, "wb" ) )
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()