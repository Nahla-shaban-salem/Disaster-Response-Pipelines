import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """"
    INPUT 
    - messages csv file name and file path
    -categories csv file name and file path
    
    OUTPUT - pandas dataframe 
       1. read the message file into a pandas dataframe
       2. read the categories file into a pandas dataframe
       3. merge the messages dataframe and catergories dataframe
       4. return the merged dataframe
    
    """
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    df = messages.merge(categories,how='inner',on='id')
    return df


def clean_data(df):
    '''
        INPUT - pandas  dataframe
        OUTPUT - pandas dataframe with cleaned data
        1. create categories dataframe by spliting the categories column by ';'
        2. rename the new columns created by category values.
        3. Convert category values to numeric values 0 or 1.
        4. concate the input dataframe and the concate categories split columnn
        5. remove any duplicate messages
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[[0]]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [colna.split('-')[0] for colna in row.values[0]]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """" INPUT - data frame , DB file name
         OUTPUT - None
         1- open DB connection
         2- create table by DF
         3- colse connection
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False,if_exists = 'replace')
    engine.dispose()
    return None  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()