import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

import os
os.chdir('C:\Users\Micah\Desktop\Baseball Data Science')

df = pd.read_csv('http://seamheads.com/baseballgauge/downloads/events.csv')
stats = df.copy()
stats[' EVENT_CD'] = stats[' EVENT_CD'].astype('str')

def define_counts(stats):
    stats['final_count'] = stats[' BALLS_CT'].astype(str) + '-' + stats[' STRIKES_CT'].astype(str)
    counts = stats.groupby([' BAT_TEAM_ID', 'final_count']).size()
    counts = pd.DataFrame(counts)
    counts['id'] = counts.index
    
    counts['id'] = counts['id'].astype(str)
    
    counts['team'] = counts['id'].str[2:5]
    counts['count'] = counts['id'].str[9:12]
    
    counts = counts.drop('id', 1)
    counts.columns = ['occurrences', 'team', 'count']
    
    totals = counts.groupby(['team']).agg({'occurrences': sum}).reset_index()
    counts = pd.merge(counts, totals, on='team', how='left')
    counts.columns = ['occurrences', 'team', 'count', 'total_at_bats']
    counts['percentage'] = counts['occurrences'] / counts['total_at_bats']
    counts['percentage'] = counts['percentage'] * 100
    return counts
    
counts = define_counts(stats)

zero_zero = counts.loc[counts['count'] == '0-0']
zero_two = counts.loc[counts['count'] == '0-2']
one_two = counts.loc[counts['count'] == '1-2']
two_zero = counts.loc[counts['count'] == '2-0']
three_one = counts.loc[counts['count'] == '3-1']

def make_count_plot(count_data, title):
    sns.set(font_scale=1.25)
    plt.figure(figsize=(16, 10))
    plt1 = sns.barplot(x="team", y="percentage", data=count_data)
    ax = plt.axes()
    ax.set_title(title)
    return plt1

make_count_plot(zero_zero, 'Percentage of Plays Ending with 0-0 Count')
make_count_plot(zero_two, 'Percentage of Plays Ending with 0-2 Count')
make_count_plot(one_two, 'Percentage of Plays Ending with 1-2 Count')
make_count_plot(two_zero, 'Percentage of Plays Ending with 2-0 Count')
make_count_plot(three_one, 'Percentage of Plays Ending with 3-1 Count')

# Lead off runner analysis
def lead_off_runners(stats):
    lead_off = stats.loc[stats[' LEADOFF_FL'] == 'T']

    lead_off.loc[lead_off[' EVENT_CD'].str.startswith('20'),
                 'On Base'] = 'yes'

    for i in range(14,24):
        i = str(i)
        lead_off.loc[lead_off[' EVENT_CD'].str.startswith(i),
                 'On Base'] = 'yes'

    lead_off['On Base'] = lead_off['On Base'].fillna('no')

    lead_off_team_counts = lead_off.groupby(' BAT_TEAM_ID')['On Base'].count()
    lead_off_team_counts = pd.DataFrame(lead_off_team_counts)
    lead_off_team_counts.columns = ['lead_off_at_bats']

    lead_off_teams_success = lead_off[lead_off['On Base'] == 'yes'].groupby(' BAT_TEAM_ID')['On Base'].count()
    lead_off_teams_success = pd.DataFrame(lead_off_teams_success)
    lead_off_teams_success.columns = ['lead_off_successes']

    lead_off_stats = pd.merge(lead_off_teams_success, lead_off_team_counts,
                              left_index=True, right_index=True, how='inner')

    lead_off_stats['success_rate'] = lead_off_stats['lead_off_successes'] / lead_off_stats['lead_off_at_bats']

    lead_off_stats['team'] = lead_off_stats.index
    lead_off_stats = lead_off_stats.sort('success_rate', ascending=0)
    return lead_off_stats

lead_off_stats = lead_off_runners(stats)

ax = sns.barplot(data=lead_off_stats, x='success_rate', y='team')
ax.set(xlabel='Success Rate')
sns.set(font_scale=1.25)
sns.plt.title('Success Rate of Lead-off Batter Getting on Base')

# Strikeouts - walks
def strikeouts_and_walks(stats):
    strikeouts = stats[stats[' EVENT_CD'] == '3'].groupby(' BAT_TEAM_ID')[' EVENT_CD'].count()
    strikeouts = pd.DataFrame(strikeouts)
    strikeouts.columns = ['strikeouts']

    walks = stats[stats[' EVENT_CD'] == '14'].groupby(' BAT_TEAM_ID')[' EVENT_CD'].count()
    walks = pd.DataFrame(walks)
    walks.columns = ['walks']

    k_bb_difference = pd.merge(strikeouts, walks,
                              left_index=True, right_index=True, how='inner')

    k_bb_difference['difference'] = k_bb_difference['strikeouts'] - k_bb_difference['walks']

    team_home_runs = stats[stats[' EVENT_CD'] == '23'].groupby(' BAT_TEAM_ID')[' EVENT_CD'].count()
    team_home_runs = pd.DataFrame(team_home_runs)
    team_home_runs.columns = ['home_runs']

    k_bb_difference = pd.merge(k_bb_difference, team_home_runs,
                              left_index=True, right_index=True, how='inner')

    k_bb_difference['team'] = k_bb_difference.index

    k_bb_difference = k_bb_difference[['difference', 'home_runs', 'team']]

    k_bb_difference = k_bb_difference.sort('difference', ascending=0)
    return k_bb_difference

k_bb_difference = strikeouts_and_walks(stats)

sns.stripplot(x="difference", y="team", data=k_bb_difference)
sns.plt.title('Strikeouts minus Walks')

k_bb_difference = k_bb_difference.sort('home_runs', ascending=0)
sns.stripplot(x="home_runs", y="team", data=k_bb_difference)
sns.plt.title('Home Runs')

# Left vs. right hand batter obp heatmap
def right_left_obp(stats):
    for i in range(14,24):
        i = str(i)
        stats.loc[stats[' EVENT_CD'].str.startswith(i),
                 'On Base'] = 'yes'

    stats.loc[stats[' EVENT_CD'].str.startswith('19'),
                 'On Base'] = 'no'

    stats['On Base'] = stats['On Base'].fillna('no')

    team_left_abs = stats[stats[' PIT_HAND_CD'] == 'L'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_left_abs = pd.DataFrame(team_left_abs)
    team_left_abs.columns = ['at_bats_against_lefties']

    team_right_abs = stats[stats[' PIT_HAND_CD'] == 'R'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_right_abs = pd.DataFrame(team_right_abs)
    team_right_abs.columns = ['at_bats_against_righties']

    on_base_subset = stats[stats['On Base'] == 'yes']

    team_left_on_base = on_base_subset[on_base_subset[' PIT_HAND_CD'] == 'L'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_left_on_base = pd.DataFrame(team_left_on_base)
    team_left_on_base.columns = ['on_base_against_lefties']

    team_right_on_base = on_base_subset[on_base_subset[' PIT_HAND_CD'] == 'R'].groupby(' BAT_TEAM_ID')[' PIT_HAND_CD'].count()
    team_right_on_base = pd.DataFrame(team_right_on_base)
    team_right_on_base.columns = ['on_base_against_righties']

    obp_split = pd.merge(team_left_abs, team_right_abs, left_index=True,
                             right_index=True, how='left')

    obp_split = pd.merge(obp_split, team_left_on_base, left_index=True,
                             right_index=True, how='left')

    obp_split = pd.merge(obp_split, team_right_on_base, left_index=True,
                             right_index=True, how='left')

    obp_split['obp_against_lefties'] = obp_split['on_base_against_lefties'] / obp_split['at_bats_against_lefties']
    obp_split['obp_against_righties'] = obp_split['on_base_against_righties'] / obp_split['at_bats_against_righties']
    obp_split['team'] = obp_split.index
    obp_split = obp_split.sort(['obp_against_lefties', 'obp_against_righties'], ascending=[1, 0])

    melt = pd.melt(obp_split, id_vars=['team'],
                   value_vars=['obp_against_lefties', 'obp_against_righties'])

    melt.columns = ['team', 'pitcher_hand', 'obp']
    heat = melt.pivot("team", "pitcher_hand", "obp")
    return heat

heat = right_left_obp()

sns.set(font_scale=1.25)
plt.figure(figsize=(20, 14))
sns.heatmap(heat, annot=True,  linewidths=.5)
sns.plt.title('OBP Splits')

# Batted Ball CD
team_set = stats[stats[' BAT_TEAM_ID'].isin(['KCA', 'HOU', 'WAS', 'COL', 'BAL'])]

def batted_balls(team):
    fly = team_set[(team_set[' BAT_TEAM_ID'] == team) & (team_set[' BATTEDBALL_CD'] == 'F')].groupby('GAME_ID')[' BATTEDBALL_CD'].count() 
    fly = pd.DataFrame(fly)
    fly.columns = ['fly balls']    
    
    pop_up = team_set[(team_set[' BAT_TEAM_ID'] == team) & (team_set[ ' BATTEDBALL_CD'] == 'P')].groupby('GAME_ID')[' BATTEDBALL_CD'].count()
    pop_up = pd.DataFrame(pop_up)
    pop_up.columns = ['pop ups']     
    
    line_drive = team_set[(team_set[' BAT_TEAM_ID'] == team) & (team_set[ ' BATTEDBALL_CD'] == 'L')].groupby('GAME_ID')[' BATTEDBALL_CD'].count()
    line_drive = pd.DataFrame(line_drive)
    line_drive.columns = ['line drives']     
    
    ground_ball = team_set[(team_set[' BAT_TEAM_ID'] == team) & (team_set[ ' BATTEDBALL_CD'] == 'G')].groupby('GAME_ID')[' BATTEDBALL_CD'].count()
    ground_ball = pd.DataFrame(ground_ball)
    ground_ball.columns = ['ground balls'] 
      
    df = pd.merge(fly, pop_up, left_index=True, 
                         right_index=True, how='left')
                         
    df = pd.merge(df, line_drive, left_index=True, 
                         right_index=True, how='left')
                         
    df = pd.merge(df, ground_ball, left_index=True, 
                         right_index=True, how='left')
                         
    df['team'] = team
    
    return df

df = batted_balls(stats)
    
kca_hits = get_batted_balls('KCA')
hou_hits = get_batted_balls('HOU')
was_hits = get_batted_balls('WAS')
col_hits = get_batted_balls('COL')
bal_hits = get_batted_balls('BAL')  

def get_averages(dataframe):
    print(dataframe['fly balls'].mean())
    print(dataframe['pop ups'].mean())
    print(dataframe['line drives'].mean())
    print(dataframe['ground balls'].mean())
    
get_averages(kca_hits)
get_averages(hou_hits)
get_averages(was_hits)
get_averages(col_hits)
get_averages(bal_hits)

def clean_hits_data(kca_hits, hou_hits, was_hits, col_hits, bal_hits):
    frames = [kca_hits, hou_hits, was_hits, col_hits, bal_hits]
    hits = pd.concat(frames)

    hits = hits.apply(lambda x: x.fillna(0))

    melt_hits = pd.melt(hits, id_vars=['team'],
                   value_vars=['fly balls', 'pop ups', 'ground balls',
                   'line drives'])

    melt_hits.columns = ['team', 'hit_type', 'count']
    return melt_hits

melt_hits = clean_hits_data(kca_hits, hou_hits, was_hits, col_hits, bal_hits)
      
sns.swarmplot(x="hit_type", y="count", hue="team", color="green", data=melt_hits)
sns.plt.title('Types of Hits by Game')

#Calculate rolling batting average for Royals
#Isolate royals
def rolling_batting_average(stats):
    kcr = stats.loc[stats[' BAT_TEAM_ID'] == 'KCA']
    kcr['date'] = kcr['GAME_ID'].str[7:]
    kcr['date'] = kcr['date'].astype('int')

    plate_appearances = kcr.groupby(['GAME_ID']).size()
    plate_appearances = pd.DataFrame(plate_appearances)
    plate_appearances.columns = ['plate_appearances']

    singles = kcr[kcr[' EVENT_CD'] == '20'].groupby('GAME_ID')[' EVENT_CD'].count()
    singles = pd.DataFrame(singles)
    singles.columns = ['singles']

    doubles = kcr[kcr[' EVENT_CD'] == '21'].groupby('GAME_ID')[' EVENT_CD'].count()
    doubles = pd.DataFrame(doubles)
    doubles.columns = ['doubles']

    triples = kcr[kcr[' EVENT_CD'] == '22'].groupby('GAME_ID')[' EVENT_CD'].count()
    triples = pd.DataFrame(triples)
    triples.columns = ['triples']

    homeruns = kcr[kcr[' EVENT_CD'] == '23'].groupby('GAME_ID')[' EVENT_CD'].count()
    homeruns = pd.DataFrame(homeruns)
    homeruns.columns = ['homeruns']

    walks = kcr[kcr[' EVENT_CD'] == '14'].groupby('GAME_ID')[' EVENT_CD'].count()
    walks = pd.DataFrame(walks)
    walks.columns = ['walks']

    intentional_walks = kcr[kcr[' EVENT_CD'] == '15'].groupby('GAME_ID')[' EVENT_CD'].count()
    intentional_walks = pd.DataFrame(intentional_walks)
    intentional_walks.columns = ['intentional_walks']

    hit_by_pitch = kcr[kcr[' EVENT_CD'] == '16'].groupby('GAME_ID')[' EVENT_CD'].count()
    hit_by_pitch = pd.DataFrame(hit_by_pitch)
    hit_by_pitch.columns = ['hit_by_pitch']

    interference = kcr[kcr[' EVENT_CD'] == '17'].groupby('GAME_ID')[' EVENT_CD'].count()
    interference = pd.DataFrame(interference)
    interference.columns = ['interference']

    hitting_stats = pd.merge(plate_appearances, singles, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, doubles, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, triples, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, homeruns, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, walks, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, intentional_walks, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, hit_by_pitch, left_index=True,
                             right_index=True, how='left')

    hitting_stats = pd.merge(hitting_stats, interference, left_index=True,
                             right_index=True, how='left')

    hitting_stats = hitting_stats.apply(lambda x: x.fillna(0))

    hitting_stats['hits'] = hitting_stats['singles'] + hitting_stats['doubles'] + hitting_stats['triples'] + hitting_stats['homeruns']

    hitting_stats['at_bats'] = hitting_stats['plate_appearances'] - (
        hitting_stats['walks'] + hitting_stats['intentional_walks'] +
        hitting_stats['hit_by_pitch'] + hitting_stats['interference'])

    hitting_stats['batting_average_per_game'] = hitting_stats['hits'] / hitting_stats['at_bats']
    hitting_stats['cum_hits'] = hitting_stats.hits.cumsum()
    hitting_stats['cum_at_bats'] = hitting_stats.at_bats.cumsum()
    hitting_stats['rolling_batting_average'] = hitting_stats['cum_hits'] / hitting_stats['cum_at_bats']
    return hitting_stats

hitting_stats = rolling_batting_average(stats)

plt.figure()
ba = hitting_stats[['rolling_batting_average']]
ba.plot()
