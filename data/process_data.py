import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load and merges the 2 given files and returns them as 1 pandas dataframe object
            Parameters:
                    messages_filepath (str): filepath to messsages csv
                    categories_filepath (str): filepath to categories csv
            Returns:
                    df_merged (object): loaded files merged into one dataframe
    '''
    #read csvs
    df_messages = pd.read_csv("disaster_messages.csv")
    df_categories = pd.read_csv("disaster_categories.csv")
    
    #merge csvs
    df_merged = pd.merge(df_messages, df_categories, on="id")
    
    return df_merged


def clean_data(df):
    '''
    Cleans the merged dataframe from load_data()
            Parameters:
                    df (object): uncleanded data as dataframe 
                    
            Returns:
                    df_cleaned (object): cleaned data as dataframe
    '''
    
   #create a new dataframe with a column for each categorie
    df_categories = pd.DataFrame(df["categories"].str.split(';', expand=True).values,
                 columns=[df["categories"].str.split(';')[0]])

    #reset multilevel index by changing columns list so single level index
    df_categories.columns = df["categories"].str.split(';')[0]

    #only keeping 0 or 1 from values and 
    for column in df_categories.columns:
        df_categories[column] = df_categories[column].str[-1:]

    #correct column names
    temp_col_name_list = []
    for col_name in df_categories.columns:
        temp_col_name_list.append(col_name[:-2])

    df_categories.columns = temp_col_name_list
   

    #concat both dataframes to get a single one with full informatio
    df = pd.concat([df, df_categories], axis=1)
    
    #drop empty and pointless data
    df = df[df.message.str.len() >  10]
    
    #drop all rows where related is 2
    df.drop(df[df["related"] == "2"].index, inplace=True)

    #drop categories column
    df_cleaned = df.drop(columns=["categories"])
    
    #drop duplicate cols
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned



def save_data(df, database_filename):
    '''
    Takes data as dataframe and stores it into a database at a given location
            Parameters:
                    df (object): data as dataframe 
                    database_filename (str): name of the database, where the df will be saved in
                    
            Returns:
                    None
    '''
    #create sqllite engine
    engine = create_engine('sqlite:///' + database_filename, echo=True)
    sqlite_connection = engine.connect()
    
    #save dataframe to sql database
    df.to_sql("disaster_response", sqlite_connection, if_exists='replace')
    return  


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