
# coding: utf-8

# In[15]:

#Import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# In[16]:

#Use requests library to get the URL that we'll use to construct game IDs
r  = requests.get("http://www.brooksbaseball.net/tabs.php?player=477132&p_hand=-1&ppos=-1&cn=200&compType=none&gFilt=&time=month&minmax=ci&var=gl&s_type=2&startDate=03/30/2007&endDate=10/22/2016&balls=-1&strikes=-1&b_hand=-1")


# In[17]:

data = r.text


# In[18]:

#Create beautiful soup object from the data
soup = BeautifulSoup(data)


# In[19]:

#Create empty list to store results of for look
results=[]


# In[20]:

#Put all the links in a list
for gid in soup.find_all('a'):
    results.append(gid.get('href'))


# In[21]:

#Delete unnecessary elements of the list
del results[0:21]


# In[22]:

#Create empty list to store results of a for loop
results1=[]


# In[23]:

#Extract the Game ID for each link
results1 = [i.split('&prevGame=', 1)[1] for i in results]


# In[24]:

#Concantenate strings to create URLs
results2 = map('http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=477132&game={0}'.format, results1)


# In[25]:

#One game is missing data, so we need to delete it
#missing game: http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=477132&game=gid_2009_09_04_sdnmlb_lanmlb_1/
del results2[53]


# In[26]:

#We also need to delete one more game, as it is also missing
del results2[55]


# In[27]:

#Create empty list to store results of a for loop
results3 = []


# In[28]:

#Scrape each link
for i in results2:
    results3.append(requests.get(i))


# In[29]:

#Create empty list to store results of a for loop
results4 = []


# In[30]:

#Grab the text for each link
for i in results3:
    results4.append(i.text)


# In[31]:

#Create empty list to store results of a for loop
results5 = []


# In[32]:

#Make each a beautiful soup object
for i in results4:
    results5.append(BeautifulSoup(i))


# In[33]:

#Define a test data frame
test = results5[1]


# In[48]:

#Extract column headers
column_headers = [th.getText() for th in 
                  test.findAll('tr', limit=2)[0].findAll('th')]


# In[35]:

#Define data rows
data_rows = test.findAll('tr') 


# In[36]:

#Get player data from table
player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]


# In[37]:

#Convert to data frame
df = pd.DataFrame(player_data, columns = column_headers)


# In[38]:

#Now do this for all the data, also adding empty lists to store the data
results6 = []


# In[39]:

for i in results5:
    results6.append(i.findAll('tr'))


# In[40]:

results7 = []


# In[41]:

for j in results6:
    results7.append([[td.getText() for td in j[i].findAll('td')]
            for i in range(len(j))])


# In[42]:

results8 = []


# In[43]:

for i in results7:
    results8.append(pd.DataFrame(i))


# In[48]:

#I ended up not needing this empty list
#results9 = []


# In[111]:

for i in results8:
    i.columns = column_headers


# In[173]:

#Create empty list
results10 = []


# In[144]:

columns1 = ["sz_top", "sz_bot", "pitch_con", "spin", "norm_ht", "tstart", "vystart", "ftime", "pfx_x", "pfx_z", 
           "uncorrected_pfx_x", "uncorrected_pfx_z", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
          "start_speed", "px", "pz", "pxold", "pzold", "sb"]


# In[201]:

for i in results8:
    results10.append(i[["sz_top", "sz_bot", "pitch_con", "spin", "norm_ht", "tstart", "vystart", "ftime", "pfx_x", "pfx_z", 
           "uncorrected_pfx_x", "uncorrected_pfx_z", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
          "start_speed", "px", "pz", "pxold", "pzold", "sb"]])
    


# In[211]:

#Create another empty list
results11 = []


# In[ ]:

for i in results10:
    results11.append(i.convert_objects(convert_numeric=True))
    
results12 = []


# In[ ]:

for i in results11:
    results12.append(i[["sz_top", "sz_bot", "pitch_con", "spin", "norm_ht", "tstart", "vystart", "ftime", "pfx_x", "pfx_z", 
           "uncorrected_pfx_x", "uncorrected_pfx_z", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
          "start_speed", "px", "pz", "pxold", "pzold", "sb"]].mean()) 


#Convert each list to a data frame


# In[217]:

pitchfx_df = pd.DataFrame(results12)


# In[218]:

print(pitchfx_df)

# In[ ]:

#Scrape bbref data


# In[135]:

results12 = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']


# In[136]:

results14 = map('http://www.baseball-reference.com/players/gl.cgi?id=kershcl01&t=p&year={0}'.format, results12)


# In[138]:

results15 = []


# In[139]:

#Scrape each link
for i in results14:
    results15.append(requests.get(i))


# In[140]:

results16 = []


# In[141]:

for i in results15:
    results16.append(i.text)


# In[142]:

results17 = []


# In[144]:

for i in results16:
    results17.append(BeautifulSoup(i))


# In[145]:

#Define a test data frame
test = results17[0]


# In[148]:

#Define data rows
data_rows = test.findAll('tr') 


# In[153]:

player_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]


# In[156]:

#Convert to data frame
df = pd.DataFrame(player_data)


# In[159]:

#Now do this for all the data
results18 = []


# In[160]:

for i in results17:
    results18.append(i.findAll('tr'))


# In[161]:

results19 = []


# In[263]:

for j in results18:
    results19.append([[td.getText() for td in j[i].findAll('td')]
            for i in range(len(j))])


# In[264]:

results20 = []


# In[265]:

for i in results19:
    results20.append(pd.DataFrame(i))


# In[283]:

year08 = results20[0]
year09 = results20[1]
year10 = results20[2]
year11 = results20[3]
year12 = results20[4]
year13 = results20[5]
year14 = results20[6]
year15 = results20[7]
year16 = results20[8]


# In[284]:

year08.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[286]:

year09.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[287]:

year10.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[288]:

year11.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[289]:

year12.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[290]:

year13.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[291]:

year14.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[292]:

year15.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[293]:

year16.rename(columns={11: 'IP', 12: 'H', 13: 'R', 14: 'ER', 15: 'BB', 16: 'SO', 17: 'HR', 
                      18: 'HBP', 20: 'BF', 21: 'Pit', 22: 'Str', 23: 'StL', 24: 'StS', 
                      25: 'GB', 26: 'FB', 27: 'LD', 28: 'PU', 29: 'Unk', 30: 'GSc', 33: 'SB',
                      34: 'CS', 35: 'PO', 36: 'AB', 37: '2B', 38: '3B', 39: 'IBB', 40: 'GDP', 
                      41: 'SF', 42: 'ROE', 43: 'aLI', 44: 'WPA', 45: 'RE24'}, inplace=True)


# In[295]:

year08 = year08[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[297]:

year09 = year09[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[298]:

year10 = year10[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[299]:

year11 = year11[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[300]:

year12 = year12[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[301]:

year13 = year13[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[302]:

year14 = year14[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[303]:

year15 = year15[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[304]:

year16 = year16[['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD', 'PU',
               'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24']]


# In[307]:

frames = [year08, year09, year10, year11, year12, year13, year14, year15, year16]


# In[334]:

bbref_df = pd.concat(frames)


# In[338]:

bbref_df = bbref_df[bbref_df.Pit > 0]


# In[336]:

#Delete excess rows from pitchfx data
#Munge
#Cluster


# In[ ]:




# In[ ]:

