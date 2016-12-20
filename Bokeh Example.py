
# coding: utf-8

# In[40]:

#Import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# In[41]:

#Use requests library to get the URL
r  = requests.get("http://www.baseball-reference.com/teams/tgl.cgi?team=KCR&t=b&year=2016")
data = r.text


# In[ ]:

#Create beautiful soup object from the data
soup = BeautifulSoup(data)


# In[43]:

#Extract column headers
column_headers = [th.getText() for th in 
                  soup.findAll('tr', limit=2)[0].findAll('th')]


# In[44]:

#Define data rows
data_rows = soup.findAll('tr') 


# In[45]:

#Get data from table
team_data = [[td.getText() for td in data_rows[i].findAll('td')]
            for i in range(len(data_rows))]


# In[46]:

#Convert to data frame
df = pd.DataFrame(team_data, columns = column_headers)


# In[23]:

#Bokeh imports
from bokeh.io import output_notebook, show
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import output_notebook
output_notebook()


# In[36]:

#Develop hover plot for hits and runs
source = ColumnDataSource(
        data=dict(
            x=df['R'],
            y=df['H'],
            desc=df['Opp'],
        )
    )

hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    )

p = figure(plot_width=900, plot_height=900, tools=[hover], title="Mouse over the dots")

p.circle('x', 'y', size=20, source=source)

show(p)

