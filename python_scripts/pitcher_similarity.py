# Citation: Programming Collective Intelligence by Toby Segaran
# Import Libraries
import os

import pandas as pd
import pymysql.cursors
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


## Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='xxxxx',
                             db='xxxxx',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def ingest_data():
    pitchers_query = '''
    select
    concat(master.nameFirst, ' ', master.nameLast) as "Name",
    pitching.yearID as "Year",
    pitching.W as "Wins",
    pitching.L as "Losses",
    pitching.G as "Appearances",
    pitching.GS as "Games_Started",
    pitching.CG as "Complete_Games",
    pitching.SHO as "Shutouts",
    pitching.SV as "Saves",
    pitching.IPouts as "Outs_Recorded",
    pitching.H as "Hits_Surrendered",
    pitching.ER as "Earned_Runs",
    pitching.HR as "Home_Runs_Surrendered",
    pitching.BB as "Walks_Surrendered",
    pitching.SO as "Strikeouts",
    pitching.BAOpp as "Opponent_Batting_Average",
    pitching.ERA as "ERA",
    pitching.R as "Runs_Surrendered"

    from pitching
    inner join master on pitching.playerID = master.playerID

    where pitching.YearID >= 1900;'''

    pitchers = pd.read_sql(pitchers_query, connection)
    return pitchers

def clean_data(pitchers):
    pitchers = pitchers.dropna()    
    
    pitchers['Year'] = pitchers['Year'].astype('str')
    pitchers['Player_and_Year'] = pitchers['Name'] + ' ' + pitchers['Year']
    
    pitchers['Decisions'] = pitchers['Wins'] + pitchers['Losses']
    pitchers['Wins_Over_Decisions'] = pitchers['Wins'] / pitchers['Decisions']
    pitchers['Wins_Over_Starts'] = pitchers['Wins'] / pitchers['Games_Started']
    
    pitchers['Relief_Appearances'] = pitchers['Appearances']\
    - pitchers['Games_Started'] 
    
    pitchers['Shutout_Percentage'] = pitchers['Shutouts']\
    / pitchers['Games_Started']
    
    pitchers['Outs_Recorded_Per_Appearance'] = pitchers['Outs_Recorded']\
    /pitchers['Appearances']
    
    pitchers['Hits_Allowed_Per_Appearance'] = pitchers['Hits_Surrendered']\
    /pitchers['Appearances']
    
    pitchers['Earned_Runs_Per_Appearance'] = pitchers['Earned_Runs']\
    /pitchers['Appearances']
    
    pitchers['Runs_Per_Appearance'] = pitchers['Runs_Surrendered']\
    /pitchers['Appearances']
    
    pitchers['Home_Runs_Per_Appearance'] = pitchers['Home_Runs_Surrendered']\
    /pitchers['Appearances']
    
    pitchers['Walks_Per_Appearance'] = pitchers['Walks_Surrendered']\
    /pitchers['Appearances']
    
    pitchers['Strikeouts_Per_Appearance'] = pitchers['Strikeouts']\
    /pitchers['Appearances']
    
    pitchers = pitchers[['Player_and_Year', 'Decisions', 'Wins_Over_Decisions',
                         'Wins_Over_Starts', 'Relief_Appearances',
                         'Shutout_Percentage', 'Outs_Recorded_Per_Appearance',
                         'Hits_Allowed_Per_Appearance', 'Earned_Runs_Per_Appearance',
                         'Runs_Per_Appearance', 'Home_Runs_Per_Appearance',
                         'Walks_Per_Appearance', 'Strikeouts_Per_Appearance',
                         'ERA']]
                         
    pitchers = pitchers.fillna(value=0)
    pitchers['Wins_Over_Starts'] = pitchers['Wins_Over_Starts'].astype('str')
    pitchers['Wins_Over_Starts'] = pitchers['Wins_Over_Starts'].str.replace('inf', '0')
    pitchers['Wins_Over_Starts'] = pitchers['Wins_Over_Starts'].astype('float')
    pitchers['Decisions'] = pitchers['Decisions'].astype('int')
    pitchers['Relief_Appearances'] = pitchers['Relief_Appearances'].astype('int')
                         
    return pitchers
    
    
# Scale data
def scale_data(pitchers):
    num_data = pitchers[['Decisions', 'Wins_Over_Decisions',
                         'Wins_Over_Starts', 'Relief_Appearances',
                         'Shutout_Percentage', 'Outs_Recorded_Per_Appearance',
                         'Hits_Allowed_Per_Appearance', 'Earned_Runs_Per_Appearance',
                         'Runs_Per_Appearance', 'Home_Runs_Per_Appearance',
                         'Walks_Per_Appearance', 'Strikeouts_Per_Appearance',
                         'ERA']]
                         
    scaler = MinMaxScaler()
    scaler.fit(num_data)
    num_data = scaler.transform(num_data)
    num_data = pd.DataFrame(num_data)

    num_data.columns = ['Decisions', 'Wins_Over_Decisions',
                         'Wins_Over_Starts', 'Relief_Appearances',
                         'Shutout_Percentage', 'Outs_Recorded_Per_Appearance',
                         'Hits_Allowed_Per_Appearance', 'Earned_Runs_Per_Appearance',
                         'Runs_Per_Appearance', 'Home_Runs_Per_Appearance',
                         'Walks_Per_Appearance', 'Strikeouts_Per_Appearance',
                         'ERA']
                         
    pitchers = pitchers[['Player_and_Year']]
    
    pitchers = pd.merge(pitchers, num_data, how='inner', left_index=True,
                        right_index=True)
    
    return pitchers
    

# Create dictionary of pitchers
def create_dictionary(pitchers):
    pitchers_melted = pd.melt(pitchers, id_vars=['Player_and_Year'], 
                   value_vars=['Decisions', 'Wins_Over_Decisions',
                             'Wins_Over_Starts', 'Relief_Appearances',
                             'Shutout_Percentage', 'Outs_Recorded_Per_Appearance',
                             'Hits_Allowed_Per_Appearance', 'Earned_Runs_Per_Appearance',
                             'Runs_Per_Appearance', 'Home_Runs_Per_Appearance',
                             'Walks_Per_Appearance', 'Strikeouts_Per_Appearance',
                             'ERA'])


    player_dictionary = pitchers_melted.groupby('Player_and_Year').apply(lambda x: x.set_index\
    ('variable')['value'].to_dict()).to_dict()
    
    return player_dictionary


# Euclidean Distance Function
def sim_distance(atts, p1, p2):
    si = {}
    for item in atts[p1]:
        if item in atts[p2]:
            si[item] = 1
            
    if len(si) == 0:
        return 0

    sum_of_squares = sum([pow(atts[p1][item] - atts[p2][item], 2) for item in
                         atts[p1] if item in atts[p2]])
                         
    return 1 / (1 + sqrt(sum_of_squares))

  
# Get top matches
def top_matches(atts, person, n=15, similarity=sim_distance):

    scores = [(similarity(atts, person, other), other) for other in atts
              if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

  
# Run the similarity analysis
def get_top_matches(player_and_year):
    df = top_matches(player_dictionary, player_and_year)
    df = pd.DataFrame(df)
    df.columns = ['Similarity', 'Pitcher_and_Year']
    return df
  
if __name__ == "__main__:
    pitchers = ingest_data()
    pitchers = clean_data(pitchers)
    pitchers = scale_data(pitchers)
    player_dictionary = create_dictionary(pitchers)
  
    pedro2000 = get_top_matches('Pedro Martinez 2000')
    clemens1997 = get_top_matches('Roger Clemens 1997')
    johnson2002 = get_top_matches('Randy Johnson 2002')
    greinke2009 = get_top_matches('Zack Greinke 2009')
    maddux1992 = get_top_matches('Greg Maddux 1992')
    schilling2001 = get_top_matches('Curt Schilling 2001')
    rivera2004 = get_top_matches('Mariano Rivera 2004')
    gagne2003 = get_top_matches('Eric Gagne 2003')

    pedro2000.to_csv('pedro2000.csv', index=False)
    clemens1997.to_csv('clemens1997.csv', index=False)
    johnson2002.to_csv('johnson2002.csv', index=False)
    greinke2009.to_csv('greinke2009.csv', index=False)
    maddux1992.to_csv('maddux1992.csv', index=False)
    schilling2001.to_csv('schilling2001.csv', index=False)
    rivera2004.to_csv('rivera2004.csv', index=False)
    gagne2003.to_csv('gagne2003.csv', index=False)

