#Import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

#Use requests library to get the URL that we'll use to construct game IDs
r  = requests.get("http://www.brooksbaseball.net/tabs.php?player=477132&p_hand=-1&ppos=-1&cn=200&compType=none&gFilt=&time=month&minmax=ci&var=gl&s_type=2&startDate=03/30/2007&endDate=10/22/2016&balls=-1&strikes=-1&b_hand=-1")

data = r.text

#Create beautiful soup object from the data
soup = BeautifulSoup(data)

#Create empty list to store results of for loop
results=[]

#Put all the links in a list
for gid in soup.find_all('a'):
    results.append(gid.get('href'))

#Delete unnecessary elements of the list
del results[0:21]

#Create empty list to store results of a for loop
results1=[]

#Extract the Game ID for each link
results1 = [i.split('&prevGame=', 1)[1] for i in results]

#Concantenate strings to create URLs
results2 = list(map('http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=477132&game={0}'.format, results1))

#Two games are missing data, so we need to delete them
#missing game: http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=477132&game=gid_2009_09_04_sdnmlb_lanmlb_1/
del results2[53]

#http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=477132&game=gid_2009_10_03_colmlb_lanmlb_1/&s_type=3&h_size=700&v_size=500
del results2[55]

#Create empty list to store results of a for loop
results3 = []

#Scrape each link
for i in results2:
    results3.append(requests.get(i))

#Create empty list to store results of a for loop
results4 = []

#Grab the text for each link
for i in results3:
    results4.append(i.text)

#Create empty list to store results of a for loop
results5 = []

#Make each a beautiful soup object
for i in results4:
    results5.append(BeautifulSoup(i))

#Define a test data frame
test = results5[1]

#Extract column headers
column_headers = [th.getText() for th in 
                  test.findAll('tr', limit=2)[0].findAll('th')]

#Define data rows
data_rows = test.findAll('tr') 

#Get player data from table
player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]

#Convert to data frame
df = pd.DataFrame(player_data, columns = column_headers)
print(df)

#Now do this for all the data, also adding empty lists to store the data
results6 = []

for i in results5:
    results6.append(i.findAll('tr'))

results7 = []

for j in results6:
    results7.append([[td.getText() for td in j[i].findAll('td')]
            for i in range(len(j))])

results8 = []

for i in results7:
    results8.append(pd.DataFrame(i))

#I ended up not needing the following empty list
#results9 = []

for i in results8:
    i.columns = column_headers
    

#Create empty list to store results of a for loop
results10 = []

#Select only the numeric columns
for i in results8:
    results10.append(i[["sz_top", "sz_bot", "pitch_con", "spin", "norm_ht", "tstart", "vystart", "ftime", "pfx_x", "pfx_z", 
           "uncorrected_pfx_x", "uncorrected_pfx_z", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
          "start_speed", "px", "pz", "pxold", "pzold", "sb"]])
          
#Create another empty list to store results of a for loop
results11 = []

#Tell Pandas to treat columns as numeric
for i in results10:
    results11.append(i.convert_objects(convert_numeric=True))
    
#Create another empty list to store results of a for loop
results12 = []

#Calculate the mean for each variable for each game
for i in results11:
    results12.append(i[["sz_top", "sz_bot", "pitch_con", "spin", "norm_ht", "tstart", "vystart", "ftime", "pfx_x", "pfx_z", 
           "uncorrected_pfx_x", "uncorrected_pfx_z", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
          "start_speed", "px", "pz", "pxold", "pzold", "sb"]].mean()) 


#Convert each list to a data frame
pitchfx_df = pd.DataFrame(results12)
print(pitchfx_df)

#Scrape bbref data
#Create list of years
results12 = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']

#Marry above list with URL structure so we can scrape all years of data
results14 = map('http://www.baseball-reference.com/players/gl.cgi?id=kershcl01&t=p&year={0}'.format, results12)

#Create another empty list to store results of a for loop
results15 = []

#Scrape each link created above
for i in results14:
    results15.append(requests.get(i))

#Create another empty list
results16 = []

#Grab the text for each link
for i in results15:
    results16.append(i.text)

#Create yet another empty list
results17 = []

#Turn each into a Beautiful Soup object
for i in results16:
    results17.append(BeautifulSoup(i))

#Define a test data frame
test = results17[0]

#Define data rows
data_rows = test.findAll('tr') 

player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]

#Convert to data frame
df = pd.DataFrame(player_data)
print(df)

#Now do this for all the data
results18 = []

for i in results17:
    results18.append(i.findAll('tr'))

results19 = []

for j in results18:
    results19.append([[td.getText() for td in j[i].findAll('td')]
            for i in range(len(j))])

results20 = []

for i in results19:
    results20.append(pd.DataFrame(i))

#Separate each year; we have to do this since some years have different columns
#Name columns for each year
year08 = results20[0]
year09 = results20[1]
year10 = results20[2]
year11 = results20[3]
year12 = results20[4]
year13 = results20[5]
year14 = results20[6]
year15 = results20[7]
year16 = results20[8]

year08.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year09.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year10.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year11.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year12.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year13.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year14.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year15.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

year16.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)

#Subset columns for each year
year08 = year08[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year09 = year09[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year10 = year10[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year11 = year11[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year12 = year12[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year13 = year13[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year14 = year14[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year15 = year15[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

year16 = year16[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]

#Creaste a single dataframe of all years
frames = [year08, year09, year10, year11, year12, year13, year14, year15, year16]
bbref_df = pd.concat(frames)
bbref_df = bbref_df[bbref_df.Pit.str.contains("None") == False]
print(bbref_df)

#Write dataframes to CSVs
pitchfx_df.to_csv('kershaw_pitchfx.csv')
bbref_df.to_csv('kershaw_bbref.csv')

