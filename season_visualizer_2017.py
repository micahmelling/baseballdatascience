## Libraries and set working directory
import pandas as pd
import os
import pymysql.cursors

## Read in the data
df = pd.read_csv('http://seamheads.com/baseballgauge/downloads/events.csv')

## Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='xxxxx',
                             db='xxxxx',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

rosters_query = '''
select 
rosters.LAST_NAME_TX as "Last_Name",
rosters.FIRST_NAME_TX as "First_Name",
rosters.PLAYER_ID as 'player_id',
rosters.TEAM_ID,
teams.LG_ID as 'League'

from rosters
inner join teams on rosters.TEAM_ID = teams.TEAM_ID;'''

rosters = pd.read_sql(rosters_query, connection)
lg_id = {'A': 'AL', 'N': 'NL'}
rosters['League'] = rosters['League'].map(lg_id)

## Left vs. right batter splits; 
# On-base percentage
# OBP = (Hits + Walks + Hit by Pitch) / 
#(At Bats + Walks + Hit by Pitch + Sacrifice Fly)
# Interactive heatmap - Done!
def get_obp(df):
    df[' EVENT_CD'] = df[' EVENT_CD'].astype('str')    
    
    for i in range(20,24):
        i = str(i)
        df.loc[df[' EVENT_CD'].str.startswith(i),
                 'obp_numerator'] = 1
     
    df.loc[df[' EVENT_CD'].str.startswith('14'),
                 'obp_numerator'] = 1
     
    df.loc[df[' EVENT_CD'].str.startswith('16'),
                 'obp_numerator'] = 1            
                       
    df['obp_numerator'] = df['obp_numerator'].fillna(0)
   
    denom_objects = ['14', '16', '20', '21', '22', '23', '2', '3', '18']
    
    df.loc[df[' EVENT_CD'].isin(denom_objects),
                 'obp_denominator'] = 1
   
    df.loc[df[' SF_FL'].str.startswith('T'),
                 'obp_denominator'] = 1
                 
    df['obp_denominator'] = df['obp_denominator'].fillna(0)
    
    df_at_bats = df[df['obp_denominator'] == 1]   
    
    team_overall_abs = df_at_bats.groupby(' BAT_TEAM_ID')[' BAT_TEAM_ID'].count()
    team_overall_abs = pd.DataFrame(team_overall_abs)
    team_overall_abs.columns = ['at_bats_overall']
    
    team_left_abs = df_at_bats[df_at_bats[' PIT_HAND_CD'] == 'L'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_left_abs = pd.DataFrame(team_left_abs)
    team_left_abs.columns = ['at_bats_against_lefties']
    
    team_right_abs = df_at_bats[df_at_bats[' PIT_HAND_CD'] == 'R'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_right_abs = pd.DataFrame(team_right_abs)
    team_right_abs.columns = ['at_bats_against_righties']
    
    df_on_base = df[df['obp_numerator'] == 1]
    
    team_overall_ob = df_on_base.groupby(' BAT_TEAM_ID')[' BAT_TEAM_ID'].count()
    team_overall_ob = pd.DataFrame(team_overall_ob)
    team_overall_ob.columns = ['on_base_overall']
    
    team_left_ob = df_on_base[df_on_base[' PIT_HAND_CD'] == 'L'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_left_ob = pd.DataFrame(team_left_ob)
    team_left_ob.columns = ['on_base_against_lefties']
    
    team_right_ob = df_on_base[df_on_base[' PIT_HAND_CD'] == 'R'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_right_ob = pd.DataFrame(team_right_ob)
    team_right_ob.columns = ['on_base_against_righties']
       
    obp_split = pd.merge(team_left_abs, team_right_abs, left_index=True, 
                         right_index=True, how='left') 
                         
    obp_split = pd.merge(obp_split, team_left_ob, left_index=True, 
                         right_index=True, how='left')
                         
    obp_split = pd.merge(obp_split, team_right_ob, left_index=True, 
                         right_index=True, how='left')
                         
    obp_split = pd.merge(obp_split, team_overall_ob, left_index=True, 
                         right_index=True, how='left')
                         
    obp_split = pd.merge(obp_split, team_overall_abs, left_index=True, 
                         right_index=True, how='left')                     
                                                  
    obp_split['obp_against_lefties'] = obp_split['on_base_against_lefties'] / obp_split['at_bats_against_lefties'] 
    obp_split['obp_against_righties'] = obp_split['on_base_against_righties'] / obp_split['at_bats_against_righties'] 
    obp_split['obp_overall'] = obp_split['on_base_overall'] / obp_split['at_bats_overall']    
    obp_split['team'] = obp_split.index
    
    team_names = {'ANA': 'ANA', 'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL',
                  'BOS': 'BOS', 'CHA': 'CHW', 'CHN': 'CHC', 'CIN': 'CIN',
                  'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU',
                  'KCA': 'KCR', 'LAN': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
                  'MIN': 'MIN', 'NYA': 'NYY', 'NYN': 'NYM', 'OAK': 'OAK',
                  'PHI': 'PHI', 'PIT': 'PIT', 'SDN': 'SDP', 'SEA': 'SEA',
                  'SFN': 'SFG', 'SLN': 'STL', 'TBA': 'TBR', 'TEX': 'TEX',
                  'TOR': 'TOR', 'WAS': 'WAS'}
                  
    obp_split['team'] = obp_split['team'].map(team_names)
    
    obp_split = obp_split.drop('on_base_against_lefties', 1)
    obp_split = obp_split.drop('at_bats_against_lefties', 1)
    obp_split = obp_split.drop('on_base_against_righties', 1)
    obp_split = obp_split.drop('at_bats_against_righties', 1)
    obp_split = obp_split.drop('on_base_overall', 1)
    obp_split = obp_split.drop('at_bats_overall', 1)    
    
    obp_split = obp_split[['team', 'obp_overall', 'obp_against_righties',
                           'obp_against_lefties']]    
    
    return obp_split
       
obp_data = get_obp(df)
os.chdir('C:\\Users\\Micah\\Desktop\Baseball Data Science\\on_base_percentage\\data')
obp_data.to_csv('obp_by_team.csv', index = False)

## Best hitters with two strikes
# Add a column for slugging percentage and make a scatter plot - Done!
def top_two_strike_hitters(df):
    df[' EVENT_CD'] = df[' EVENT_CD'].astype('str')
    df = df.loc[df[' STRIKES_CT'] == 2]
    
    df[' EVENT_CD'] = df[' EVENT_CD'].astype('str')    
    
    for i in range(20,24):
        i = str(i)
        df.loc[df[' EVENT_CD'].str.startswith(i),
                 'obp_numerator'] = 1
     
    df.loc[df[' EVENT_CD'].str.startswith('14'),
                 'obp_numerator'] = 1
     
    df.loc[df[' EVENT_CD'].str.startswith('16'),
                 'obp_numerator'] = 1            
                       
    df['obp_numerator'] = df['obp_numerator'].fillna(0)
   
    denom_objects = ['14', '16', '20', '21', '22', '23', '2', '3', '18']
    
    df.loc[df[' EVENT_CD'].isin(denom_objects),
                 'obp_denominator'] = 1
   
    df.loc[df[' SF_FL'].str.startswith('T'),
                 'obp_denominator'] = 1
                 
    df['obp_denominator'] = df['obp_denominator'].fillna(0)
    
    df_at_bats = df[df['obp_denominator'] == 1] 
        
    df_on_base = df[df['obp_numerator'] == 1]
    
    df.loc[df[' EVENT_CD'].str.startswith('20'),
                 'slg_numerator'] = 1
                 
    df.loc[df[' EVENT_CD'].str.startswith('21'),
                 'slg_numerator'] = 2
                 
    df.loc[df[' EVENT_CD'].str.startswith('23'),
                 'slg_numerator'] = 3
                 
    df.loc[df[' EVENT_CD'].str.startswith('24'),
                 'slg_numerator'] = 4
                 
    df['slg_numerator'] = df['slg_numerator'].fillna('0')
    df['slg_numerator'] = df['slg_numerator'].astype('int')
    
    slg = df.groupby(' BAT_ID')['slg_numerator'].sum()
    slg = pd.DataFrame(slg)
    slg.columns = ['slg_count']    
    
    at_bats = df_at_bats.groupby(' BAT_ID')[' BAT_ID'].count()
    at_bats = pd.DataFrame(at_bats)
    at_bats.columns = ['at_bats_with_two_strikes']
        
    ob = df_on_base.groupby(' BAT_ID')[' BAT_ID'].count()
    ob = pd.DataFrame(ob)
    ob.columns = ['on_base_with_two_strikes']
            
    obp_by_player = pd.merge(at_bats, ob, left_index=True, 
                         right_index=True, how='left')
                         
    obp_by_player = pd.merge(obp_by_player, slg, left_index=True, 
                         right_index=True, how='left')
                         
    obp_by_player['player_id'] = obp_by_player.index
    
    obp_by_player = pd.merge(obp_by_player, rosters, how='left', 
                             on='player_id')
                             
    obp_by_player['Player'] = obp_by_player['First_Name'] + ' ' +\
    obp_by_player['Last_Name']
    
    obp_by_player = obp_by_player.drop_duplicates(subset='Player')
    obp_by_player = obp_by_player.dropna()
                         
    obp_by_player['on_base_percentage'] = obp_by_player['on_base_with_two_strikes']\
    / obp_by_player['at_bats_with_two_strikes']
    
    obp_by_player['slg_percentage'] = obp_by_player['slg_count']\
    / obp_by_player['at_bats_with_two_strikes']
    
    obp_by_player = obp_by_player[obp_by_player.at_bats_with_two_strikes\
    > obp_by_player.at_bats_with_two_strikes.quantile(.50)]  
    
    obp_by_player = obp_by_player.drop('at_bats_with_two_strikes', 1)
    obp_by_player = obp_by_player.drop('on_base_with_two_strikes', 1)
    obp_by_player = obp_by_player.drop('slg_count', 1)
    obp_by_player = obp_by_player.drop('player_id', 1)
    obp_by_player = obp_by_player.drop('Last_Name', 1)
    obp_by_player = obp_by_player.drop('First_Name', 1)
    
    return obp_by_player
    
obp_by_player = top_two_strike_hitters(df)
os.chdir('C:\\Users\\Micah\\Desktop\\Baseball Data Science\\obp_slg_two_strikes\\data')
obp_by_player.to_csv('obp_slg_two_strikes.csv', index=False)
    
## Best hitters for extra base hits in clutch situations
# Clutch situation is: three run game, 7th inning or later
# Segment by AL and NL and show strip plot with each point being a player    
def top_clutch_hitters(df):
    df[' EVENT_CD'] = df[' EVENT_CD'].astype('str') 
    
    df['score_difference'] = df[' AWAY_SCORE_CT'] - df[' HOME_SCORE_CT']  
    df['score_difference'] = df['score_difference'].abs()
    
    df[' INN_CT'] = df[' INN_CT'].astype('str')
    innings = ['7', '8', '9']
    
    df = df.loc[df[' INN_CT'].isin(innings)]
    df = df.loc[df['score_difference'] <= 3]
    
    for i in range(21, 24):
        i = str(i)
        df.loc[df[' EVENT_CD'].str.startswith(i),
                 'extra_base_hits'] = 1
                 
    df['extra_base_hits'] = df['extra_base_hits'].fillna(0)    
    
    denom_objects = ['14', '16', '20', '21', '22', '23', '2', '3', '18']
    
    df.loc[df[' EVENT_CD'].isin(denom_objects),
                 'obp_denominator'] = 1
   
    df.loc[df[' SF_FL'].str.startswith('T'),
                 'obp_denominator'] = 1
                 
    df['obp_denominator'] = df['obp_denominator'].fillna(0)
    
    df_at_bats = df[df['obp_denominator'] == 1] 
    
    at_bats = df_at_bats.groupby(' BAT_ID')[' BAT_ID'].count()
    at_bats = pd.DataFrame(at_bats)
    at_bats.columns = ['at_bats_in_clutch_situations']
    
    extra_base_hits = df[df['extra_base_hits'] == 1]
    
    xbh = extra_base_hits.groupby(' BAT_ID')[' BAT_ID'].count()
    xbh = pd.DataFrame(xbh)
    xbh.columns = ['extra_base_hits']
    
    xbh_by_player = pd.merge(at_bats, xbh, left_index=True, 
                         right_index=True, how='left')
                         
    xbh_by_player['player_id'] = xbh_by_player.index
    
    xbh_by_player = pd.merge(xbh_by_player, rosters, how='left', 
                             on='player_id')
                             
    xbh_by_player['Player'] = xbh_by_player['First_Name'] + ' ' +\
    xbh_by_player['Last_Name']
    
    xbh_by_player = xbh_by_player.drop_duplicates(subset='Player', keep = 'last')
                             
    xbh_by_player['xbh_percentage'] = xbh_by_player['extra_base_hits']\
    / xbh_by_player['at_bats_in_clutch_situations']
    
    xbh_by_player = xbh_by_player.dropna()
    
    xbh_by_player = xbh_by_player[xbh_by_player.at_bats_in_clutch_situations\
    > xbh_by_player.at_bats_in_clutch_situations.quantile(.50)] 
    
    xbh_by_player = xbh_by_player.drop('at_bats_in_clutch_situations', 1)
    xbh_by_player = xbh_by_player.drop('extra_base_hits', 1)
    xbh_by_player = xbh_by_player.drop('player_id', 1)
    xbh_by_player = xbh_by_player.drop('Last_Name', 1)
    xbh_by_player = xbh_by_player.drop('First_Name', 1)
    xbh_by_player = xbh_by_player.drop('TEAM_ID', 1)
    
    return xbh_by_player
    
xbh_clutch_situations = top_clutch_hitters(df)
os.chdir('C:\\Users\Micah\\Desktop\\Baseball Data Science\\xbh_clutch_hitters\\data')
xbh_clutch_situations.to_csv('xbh_clutch_situations.csv', index=False)
    
# OBP in different situations
# late innings, close game, down in count, runners on
def obp_scenarios(df):       
    df[' EVENT_CD'] = df[' EVENT_CD'].astype('str')     
    
    numer_objects = ['20', '21', '22', '23', '14', '16']
    
    df.loc[df[' EVENT_CD'].isin(numer_objects),
                 'obp_numerator'] = 1            
                       
    df['obp_numerator'] = df['obp_numerator'].fillna(0)
   
    denom_objects = ['14', '16', '20', '21', '22', '23', '2', '3', '18']
    
    df.loc[df[' EVENT_CD'].isin(denom_objects),
                 'obp_denominator'] = 1
   
    df.loc[df[' SF_FL'].str.startswith('T'),
                 'obp_denominator'] = 1
                 
    df['obp_denominator'] = df['obp_denominator'].fillna(0)
    
    df[' INN_CT'] = df[' INN_CT'].astype('str')
    innings = ['8', '9']
    df_late_innings = df.loc[df[' INN_CT'].isin(innings)]
    
    df_at_bats_late_innings = df_late_innings[df_late_innings['obp_denominator'] == 1] 
    
    at_bats_late_innings = df_at_bats_late_innings.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    at_bats_late_innings = pd.DataFrame(at_bats_late_innings)
    at_bats_late_innings.columns = ['at_bats_late_innings']
    
    df_on_base_late_innings = df_late_innings[df_late_innings['obp_numerator'] == 1]
    
    on_base_late_innings = df_on_base_late_innings.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    on_base_late_innings = pd.DataFrame(on_base_late_innings)
    on_base_late_innings.columns = ['on_base_late_innings']
    
    df['score_difference'] = df[' AWAY_SCORE_CT'] - df[' HOME_SCORE_CT']  
    df['score_difference'] = df['score_difference'].abs() 
    df_close_game = df.loc[df['score_difference'] <= 3]
    
    df_at_bats_close_game = df_close_game[df_close_game['obp_denominator'] == 1] 
    
    df_at_bats_close_game = df_at_bats_close_game.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    df_at_bats_close_game = pd.DataFrame(df_at_bats_close_game)
    df_at_bats_close_game.columns = ['at_bats_close_game']
    
    df_on_base_close_game = df_close_game[df_close_game['obp_numerator'] == 1]
    
    df_on_base_close_game = df_on_base_close_game.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    df_on_base_close_game = pd.DataFrame(df_on_base_close_game)
    df_on_base_close_game.columns = ['on_base_close_game']
    
    df_runners_on = df[(df[' BASE1_RUN_ID'].notnull()) |\
    (df[' BASE2_RUN_ID'].notnull()) | (df[' BASE3_RUN_ID'].notnull())]
    
    df_at_bats_runners_on = df_runners_on[df_runners_on['obp_denominator'] == 1] 
    
    df_at_bats_runners_on = df_at_bats_runners_on.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    df_at_bats_runners_on = pd.DataFrame(df_at_bats_runners_on)
    df_at_bats_runners_on.columns = ['at_bats_runners_on']
    
    df_on_base_runners_on = df_runners_on[df_runners_on['obp_numerator'] == 1]
    
    df_on_base_runners_on = df_on_base_runners_on.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    df_on_base_runners_on = pd.DataFrame(df_on_base_runners_on)
    df_on_base_runners_on.columns = ['on_base_runners_on']
    
    df['count'] = df[' BALLS_CT'].astype(str) + '-' + df[' STRIKES_CT'].astype(str)
           
    counts = ['0-1', '0-2', '1-2', '2-2', '3-2']
    df_down_in_count = df.loc[df['count'].isin(counts)]    
    
    df_at_bats_down_in_count = df_down_in_count[df_down_in_count['obp_denominator'] == 1] 
    
    df_at_bats_down_in_count = df_at_bats_down_in_count.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    df_at_bats_down_in_count = pd.DataFrame(df_at_bats_down_in_count)
    df_at_bats_down_in_count.columns = ['at_bats_down_in_count']
    
    df_on_base_down_in_count = df_down_in_count[df_down_in_count['obp_numerator'] == 1]
    
    df_on_base_down_in_count = df_on_base_down_in_count.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    df_on_base_down_in_count = pd.DataFrame(df_on_base_down_in_count)
    df_on_base_down_in_count.columns = ['on_base_down_in_count']
    
    obp_by_player = pd.merge(at_bats_late_innings, on_base_late_innings,
                             left_index=True, right_index=True, how='left')
  
    obp_by_player = pd.merge(obp_by_player, df_at_bats_close_game,
                             left_index=True, right_index=True, how='left')
                             
    obp_by_player = pd.merge(obp_by_player, df_on_base_close_game,
                             left_index=True, right_index=True, how='left') 
                             
    obp_by_player = pd.merge(obp_by_player, df_at_bats_runners_on,
                             left_index=True, right_index=True, how='left')
                             
    obp_by_player = pd.merge(obp_by_player, df_on_base_runners_on,
                             left_index=True, right_index=True, how='left')
                             
    obp_by_player = pd.merge(obp_by_player, df_at_bats_down_in_count,
                             left_index=True, right_index=True, how='left')
                             
    obp_by_player = pd.merge(obp_by_player, df_on_base_down_in_count,
                             left_index=True, right_index=True, how='left')
                      
    obp_by_player['player_id'] = obp_by_player.index
    
    obp_by_player = pd.merge(obp_by_player, rosters, how='left', 
                             on='player_id')
                             
    obp_by_player['Player'] = obp_by_player['First_Name'] + ' ' +\
    obp_by_player['Last_Name']
    
    obp_by_player = obp_by_player.drop_duplicates(subset='Player', keep = 'Last')
    obp_by_player = obp_by_player.dropna()
                         
    obp_by_player['obp_late_innings'] = obp_by_player['on_base_late_innings']\
    / obp_by_player['at_bats_late_innings']
    
    obp_by_player['obp_close_game'] = obp_by_player['on_base_close_game']\
    / obp_by_player['at_bats_close_game']
    
    obp_by_player['obp_runners_on'] = obp_by_player['on_base_runners_on']\
    / obp_by_player['at_bats_runners_on']
    
    obp_by_player['obp_down_in_count'] = obp_by_player['on_base_down_in_count']\
    / obp_by_player['at_bats_down_in_count']
    
    obp_by_player = obp_by_player.dropna()    
    
    obp_by_player = obp_by_player[obp_by_player.at_bats_late_innings\
    > obp_by_player.at_bats_late_innings.quantile(.50)]
    
    team_names = {'ANA': 'ANA', 'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL',
                  'BOS': 'BOS', 'CHA': 'CHW', 'CHN': 'CHC', 'CIN': 'CIN',
                  'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU',
                  'KCA': 'KCR', 'LAN': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
                  'MIN': 'MIN', 'NYA': 'NYY', 'NYN': 'NYM', 'OAK': 'OAK',
                  'PHI': 'PHI', 'PIT': 'PIT', 'SDN': 'SDP', 'SEA': 'SEA',
                  'SFN': 'SFG', 'SLN': 'STL', 'TBA': 'TBR', 'TEX': 'TEX',
                  'TOR': 'TOR', 'WAS': 'WAS'}
                  
    obp_by_player['TEAM_ID'] = obp_by_player['TEAM_ID'].map(team_names)
    
    obp_by_player = pd.melt(obp_by_player, id_vars=['Player'],
                            value_vars=['obp_late_innings',
                   'obp_close_game', 'obp_runners_on',
                   'obp_down_in_count'])
                   
    obp_by_player.columns = ['Player', 'Situation', 'OBP']
    
    obp_by_player.sort('OBP', ascending=0, inplace=True)
    
    return obp_by_player
    
obp_scenarios = obp_scenarios(df)
os.chdir('C:\\Users\Micah\\Desktop\\Baseball Data Science\\obp_scenarios\data')
obp_scenarios.to_csv('obp_scenarios.csv', index=False)
    
## Types of hits breakdown
# Show distribution by team, overlaying the types of hits    
def types_of_hits_by_team(df):
    fly = df[(df[' BATTEDBALL_CD'] == 'F')].groupby(' BAT_TEAM_ID')\
    [' BATTEDBALL_CD'].count() 
    fly = pd.DataFrame(fly)
    fly.columns = ['fly balls']    
    
    pop_up = df[(df[' BATTEDBALL_CD'] == 'P')].groupby(' BAT_TEAM_ID')\
    [' BATTEDBALL_CD'].count()
    pop_up = pd.DataFrame(pop_up)
    pop_up.columns = ['pop ups']     
    
    line_drive = df[(df[' BATTEDBALL_CD'] == 'L')].groupby(' BAT_TEAM_ID')\
    [' BATTEDBALL_CD'].count()
    line_drive = pd.DataFrame(line_drive)
    line_drive.columns = ['line drives']     
    
    ground_ball = df[(df[' BATTEDBALL_CD'] == 'G')].groupby(' BAT_TEAM_ID')\
    [' BATTEDBALL_CD'].count()
    ground_ball = pd.DataFrame(ground_ball)
    ground_ball.columns = ['ground balls'] 
      
    hits = pd.merge(fly, pop_up, left_index=True, 
                         right_index=True, how='left')
                         
    hits = pd.merge(hits, line_drive, left_index=True, 
                         right_index=True, how='left')
                         
    hits = pd.merge(hits, ground_ball, left_index=True, 
                         right_index=True, how='left')
                         
    hits['sum_of_hits'] = hits['fly balls'] + hits['pop ups'] \
                          + hits['line drives'] + hits['ground balls']
                          
    hits['fly_ball_percentage'] = hits['fly balls'] / hits['sum_of_hits']
    hits['pop_ups_percentage'] = hits['pop ups'] / hits['sum_of_hits']
    hits['line_drive_percentage'] = hits['line drives'] / hits['sum_of_hits']
    hits['ground_balls_percentage'] = hits['ground balls'] / hits['sum_of_hits']
    
    hits['Team'] = hits.index
    
    team_names = {'ANA': 'ANA', 'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL',
                  'BOS': 'BOS', 'CHA': 'CHW', 'CHN': 'CHC', 'CIN': 'CIN',
                  'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU',
                  'KCA': 'KCR', 'LAN': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
                  'MIN': 'MIN', 'NYA': 'NYY', 'NYN': 'NYM', 'OAK': 'OAK',
                  'PHI': 'PHI', 'PIT': 'PIT', 'SDN': 'SDP', 'SEA': 'SEA',
                  'SFN': 'SFG', 'SLN': 'STL', 'TBA': 'TBR', 'TEX': 'TEX',
                  'TOR': 'TOR', 'WAS': 'WAS'}
                  
    hits['Team'] = hits['Team'].map(team_names)
    
    hits = hits.drop('fly balls', 1)
    hits = hits.drop('pop ups', 1)
    hits = hits.drop('line drives', 1)
    hits = hits.drop('ground balls', 1)
    hits = hits.drop('sum_of_hits', 1)
    
    hits = pd.melt(hits, id_vars=['Team'], value_vars=['fly_ball_percentage',
                   'pop_ups_percentage', 'line_drive_percentage',
                   'ground_balls_percentage'])
                   
    hits.columns = ['Team', 'Hit_Type', 'Percentage_of_Hits']
    
    hit_types = {'fly_ball_percentage': 'fly_balls',
                 'pop_ups_percentage': 'pop_ups',
                 'line_drive_percentage': 'line_drives',
                 'ground_balls_percentage': 'ground_balls'}
                 
    hits['Hit_Type'] = hits['Hit_Type'].map(hit_types)
                            
    return hits

hits = types_of_hits_by_team(df)
os.chdir('C:\\Users\Micah\\Desktop\\Baseball Data Science\\hit_types\\data')
hits.to_csv('hit_types_by_team.csv', index=False)

# Top players at bat outcomes timeline
def plate_appearance_timelines(df):
    df['timestamp'] = df['GAME_ID'].str[7:]
    df[' INN_CT'] = df[' INN_CT'].astype('str')
    df['timestamp'] = df['timestamp'] + df[' INN_CT']
    
    df[' EVENT_CD'] = df[' EVENT_CD'].astype('str')  
    
    denom_objects = ['14', '16', '20', '21', '22', '23', '2', '3', '18']
    
    df.loc[df[' EVENT_CD'].isin(denom_objects),
                 'obp_denominator'] = 1
   
    df.loc[df[' SF_FL'].str.startswith('T'),
                 'obp_denominator'] = 1
                 
    df['obp_denominator'] = df['obp_denominator'].fillna(0)
    
    df = df[df['obp_denominator'] == 1] 
    
    df = df[[' BAT_ID', ' EVENT_CD', 'timestamp']]
    
    at_bats = df.groupby(' BAT_ID')\
    [' BAT_ID'].count()
    at_bats = pd.DataFrame(at_bats)
    at_bats.columns = ['at_bats']
    
    play_outcomes = {'2': 'Out', '3': 'Strikeout', '4': 'Other', 
                     '5': 'Other', '6': 'Other', '7': 'Other', '8': 'Other',
                     '9': 'Other', '10': 'Other', '11': 'Other', '12': 'Other',
                     '13': 'Other', '14': 'Walk', '15': 'Walk', '16': 'Walk',
                     '17': 'Other', '18': 'Other', '19': 'Other', '20': 'Hit',
                     '21': 'Hit', '22': 'Hit', '23': 'Hit', '24': 'Other', 
                     '0': 'Other', '1': 'Other'}
                     
    event_outcomes = {'0': 'Unknown', '1': 'None', '2': 'Generic Out',
                      '3': 'Strikeout', '4': 'Stolen Base',
                      '5': 'Defensive Indifference', '6': 'Caught Stealing',
                      '7': 'Pickoff error', '8': 'Pickoff', '9': 'Wild Pitch',
                      '10': 'Passed Ball', '11': 'Balk', '12': 'Other Advance',
                      '13': 'Foul Error', '14': 'Walk', '15': 'Intentional Walk',
                      '16': 'Hit By Pitch', '17': 'Interference',
                      '18': 'Error', '19': 'Fielders Choice', '20': 'Single',
                      '21': 'Double', '22': 'Triple', '23': 'Home Run',
                      '24': 'Missing'}
    
    df['Hits_Class'] = df[' EVENT_CD'].map(play_outcomes)  
    df[' EVENT_CD'] = df[' EVENT_CD'].map(event_outcomes)
    
    df.sort([' BAT_ID', 'timestamp'], ascending=[1,1], inplace=True)

    df['counter'] = 1
    df['Previous_Hitter'] = df[' BAT_ID'].shift(1)
    df = df.fillna(value='abrej003')    
    df['Plate_Appearance']= df.groupby([' BAT_ID'])['counter'].cumsum()
    
    df.columns = ['player_id', 'Outcome', 'timestamp', 'Play_Class',
                  'counter', 'Previous_Hitter', 'Plate_Appearance']
    
    df = pd.merge(df, rosters, how='inner',  on='player_id')
    df['uid'] = df['player_id'] + df['Outcome'] + df['timestamp']
    df = df.drop_duplicates(subset='uid')
    df['Player'] = df['First_Name'] + ' ' + df['Last_Name']
    df = df[['Player', 'Outcome', 'Play_Class', 'Plate_Appearance']]
    
    df = df.loc[(df.Player == 'Bryce Harper') | (df.Player == 'Buster Posey')\
    | (df.Player == 'Zack Cozart') | (df.Player == 'Jean Segura')\
    | (df.Player == 'Charlie Blackmon') | (df.Player == 'Jose Altuve')\
    | (df.Player == 'Mike Trout') | (df.Player == 'Paul Goldschmidt')]
        
    return df

timelines = plate_appearance_timelines(df)           
os.chdir('C:\\Users\Micah\\Desktop\\Baseball Data Science\\player_timelines\\data')
timelines.to_csv('player_timelines.csv', index=False)    
