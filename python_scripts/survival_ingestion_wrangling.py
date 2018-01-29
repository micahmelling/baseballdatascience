# Citation: https://www.cscu.cornell.edu/news/statnews/stnews67.pdf

# Library imports
import pymysql.cursors
import pandas as pd
from datetime import datetime


# Database connection
connection = pymysql.connect(host='localhost',
                             user='xxxxx',
                             password='xxxxx',
                             db='lahman2016',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
  
  
# Data ingestion
def ingest_data():
    master_query = '''
    SELECT * from master;'''

    pitching_query = '''
    SELECT sum(GS), playerID from pitching group by playerID;''' 

    salaries_query = '''
    SELECT yearID, salary, playerID from salaries;'''

    master = pd.read_sql(master_query, connection)     
    pitching = pd.read_sql(pitching_query, connection)    
    salaries = pd.read_sql(salaries_query, connection)  
    return master, pitching, salaries

  
def data_cleaning(master, pitching, salaries):
    # Put salaries in 2016 dollars
    inflation = pd.read_csv('inflation_conversion.csv')
    salaries = pd.merge(salaries, inflation, how = 'left', on = 'yearID')
    salaries.fillna(value = 1, inplace = True)

    salaries['adjusted_salary'] = salaries['salary'] / salaries['CF']

    # Get each players average salary
    salaries = salaries.groupby(['playerID'])['adjusted_salary'].mean()
    salaries = pd.DataFrame(salaries)
    salaries.reset_index(inplace = True)

    current = pd.read_csv('http://seamheads.com/baseballgauge/downloads/events.csv')
    current.rename(columns={' BAT_ID':'retroID'}, inplace=True)
    current = current[['GAME_ID', 'retroID']]
    current.drop_duplicates(subset = 'retroID', inplace = True)

    # Merge the data
    df = pd.merge(salaries, master, how = 'left', on = 'playerID')
    df = pd.merge(df, pitching, how = 'left', on = 'playerID')
    df = pd.merge(df, current, how = 'left', on = 'retroID')

    # Keep only position players
    df = df.loc[df['sum(GS)'] == 0]
    return df


# Mark current players as being censored
def prep_data_for_survival_analysis(df):
    df.rename(columns={'GAME_ID': 'Censored'}, inplace=True)
    df['Censored'].fillna(value = 2, inplace = True)

    def label_censor (row):
       if row['Censored'] == 2:
          return 2
       else:
        return 1

    df['Censored'] = df.apply (lambda row: label_censor(row), axis=1)

    # Age at time of debut
    # Time being in the MLB
    df['birthYear'] = df['birthYear'].astype('str')
    df['birthYear'] = df['birthYear'].str[:4]

    df['birthMonth'] = df['birthMonth'].astype('str')

    numbers_map = {'1.0': '01', '2.0': '02', '3.0': '03', '4.0': '04', '5.0': '05',
                   '6.0': '06', '7.0': '07', '8.0': '08', '9.0': '09', '10.0': '10',
                   '11.0': '11', '12.0': '12'}

    df['birthMonth'] = df['birthMonth'].map(numbers_map)

    df['birthDay'] = df['birthDay'].astype('str')
    df['birthDay'] = df['birthDay'].str[:-2]

    df['birthday'] = df['birthYear'] + '-' + df['birthMonth'] + '-' + df['birthDay']

    df['birthday'] = pd.to_datetime(df['birthday'])
    df['debut'] = pd.to_datetime(df['debut'])
    df['finalGame'] = pd.to_datetime(df['finalGame'])

    df['age_at_debut'] = df['debut'] - df['birthday']
    df['time_in_mlb'] = df['finalGame'] - df['debut']

    # Select columns for analysis
    df = df[['adjusted_salary', 'birthCountry', 'weight', 'height', 'bats',
                  'throws', 'debut', 'finalGame', 'age_at_debut',
                  'time_in_mlb', 'Censored']]

    df.columns = ['average_salary', 'birth_country', 'weight', 'height', 'hits',
                  'throws', 'debut', 'final_game', 'age_at_debut',
                  'time_in_mlb', 'status']
    return df

  
if __name__ == "__main__":
    master, pitching, salaries = ingest_data()
    df = data_cleaning(master, pitching, salaries)
    df = prep_data_for_survival_analysis(df)
    df.to_csv('player_data_for_survival_analysis.csv', index = False)
                            
