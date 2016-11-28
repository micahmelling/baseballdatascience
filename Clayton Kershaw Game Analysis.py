
# coding: utf-8

# In[3]:

#Import libraries
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[381]:

#Read in cleaned datasets
pitchfx = pd.read_csv('kershaw_pitchfx_working.csv')
bbref = pd.read_csv('kershaw_bbref_working.csv')
pitches = pd.read_csv('kershaw_pitches_working.csv')


# In[382]:

#Change display settings
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 100)


# In[383]:

#Concatenate dataframes
kershaw_df_original = pd.concat([pitchfx, bbref, pitches], axis=1)


# In[384]:

#Get summary of the dataframe
kershaw_df_original.describe()


# In[385]:

#Standardize data in dataframes
from sklearn import preprocessing
kershaw_df = preprocessing.scale(kershaw_df_original)
kershaw_df = pd.DataFrame(kershaw_df)


# In[386]:

#Conduct k-means clustering
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
k_means = KMeans(init='k-means++', n_clusters=3, n_init=25)
k_means.fit(kershaw_df)
kmeans_labels = k_means.labels_
centroids = k_means.cluster_centers_

#Get lables for k-means cluster and add to the dataframe
kmeans_labels = pd.DataFrame(kmeans_labels)
kershaw_df1 = pd.concat([kershaw_df_original, kmeans_labels], axis=1)
kershaw_df1.rename(columns={0: 'K_Means'}, inplace=True)


# In[387]:

#Conduct hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
ward = AgglomerativeClustering(n_clusters=5, linkage='ward').fit(kershaw_df)
hierarchical_labels = ward.labels_

#Get lables for hierarchical cluster and add to the dataframe
hierarchical_labels = pd.DataFrame(hierarchical_labels)
kershaw_df1 = pd.concat([kershaw_df1, hierarchical_labels], axis=1)
kershaw_df1.rename(columns={0: 'Hierarchical'}, inplace=True)


# In[388]:

#Add quartile columns based on  strikes looking (StL), strikes swinging (StS), ground balls (GB), and fly balls (FB)
StL_Quartile = pd.qcut(kershaw_df1['StL'], 4)
StS_Quartile = pd.qcut(kershaw_df1['StS'], 4)
GB_Quartile = pd.qcut(kershaw_df1['GB'], 4)
FB_Quartile = pd.qcut(kershaw_df1['FB'], 4)


# In[389]:

#Add quartile designations to the main dataframe
#StL
StL_Quartile = pd.DataFrame(StL_Quartile)
kershaw_df1 = pd.concat([kershaw_df1, StL_Quartile], axis=1)


# In[390]:

#StS
StS_Quartile = pd.DataFrame(StS_Quartile)
kershaw_df1 = pd.concat([kershaw_df1, StS_Quartile], axis=1)


# In[391]:

#GB
GB_Quartile = pd.DataFrame(GB_Quartile)
kershaw_df1 = pd.concat([kershaw_df1, GB_Quartile], axis=1)


# In[392]:

#FB
FB_Quartile = pd.DataFrame(FB_Quartile)
kershaw_df1 = pd.concat([kershaw_df1, FB_Quartile], axis=1)


# In[393]:

#Split ER into a binary classification 
ER_Classification = pd.cut(kershaw_df1['ER'], [0, 2, 9], labels=['low', 'high'])


# In[394]:

#Recode NaN values, which corresponed to zero ER
ER_Classification = ER_Classification.fillna('low')


# In[395]:

#Add ER classification to the dataframe
kershaw_df1 = pd.concat([kershaw_df1, ER_Classification], axis=1)


# In[398]:

#Write to CSV
kershaw_df1.to_csv('kershaw_df1.csv')


# In[14]:

#Summarize data by k-means cluster assignment
by_kmeanscluster = kershaw_df1.groupby('K_Means')


# In[15]:

#Create empty list
results1 = []


# In[16]:

#Run for loop to group descriptive statistics by each k-means cluster
for i in kershaw_df1:
    results1.append(by_kmeanscluster[i].describe())


# In[17]:

#View results
print(results1)


# In[18]:

#Create empty list
results2 = []


# In[19]:

#Summarize data by hierarchical cluster assignment
by_hiercluster = kershaw_df1.groupby('Hierarchical')


# In[20]:

#Run for loop to group descriptive statistics by each hierarchical cluster
for i in kershaw_df1:
    results2.append(by_hiercluster[i].describe())


# In[21]:

#View results
print(results2)


# In[30]:

#Analyze data by strike looking
g = sns.pointplot(x="Fourseam", y="StL_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Fastballs by Strike Looking Quartile', fontsize = 14)


# In[29]:

#Analyze data by strike looking
g = sns.pointplot(x="Curve", y="StL_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Curves by Strike Looking Quartile', fontsize = 14)


# In[32]:

#Analyze data by strike swinging
g = sns.pointplot(x="Fourseam", y="StS_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Fastballs by Strike Swinging Quartile', fontsize = 14)


# In[33]:

#Analyze data by strike swinging
g = sns.pointplot(x="Curve", y="StS_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Curves by Strike Swinging Quartile', fontsize = 14)


# In[34]:

#Analyze data by ground ball
g =sns.pointplot(x="Fourseam", y="GB_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Fastballs by Ground Ball Quartile', fontsize = 14)


# In[36]:

#Analyze data by ground ball
g = sns.pointplot(x="Curve", y="GB_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Curves by Ground Ball Quartile', fontsize = 14)


# In[37]:

#Analyze data by fly ball
g = sns.pointplot(x="Fourseam", y="FB_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Fastballs by Fly Ball Quartile', fontsize = 14)


# In[38]:

#Analyze data by fly ball
g = sns.pointplot(x="Curve", y="FB_Quartile", data=kershaw_df1, join=False)
g.axes.set_title('Number of Curves by Fly Ball Quartile', fontsize = 14)


# In[49]:

#Time-Series Plots
p = kershaw_df1['Fourseam']
p.plot()


# In[50]:

p = kershaw_df1['Change']
p.plot()


# In[51]:

p = kershaw_df1['Slider']
p.plot()


# In[52]:

p = kershaw_df1['Curve']
p.plot()


# In[54]:

p = kershaw_df1['pfx_x']
p.plot()


# In[55]:

p = kershaw_df1['pfx_z']
p.plot()


# In[59]:

p = kershaw_df1['StS']
p.plot()


# In[61]:

p = kershaw_df1['StL']
p.plot()


# In[74]:

p = kershaw_df1['px']
p.plot()


# In[75]:

p = kershaw_df1['pz']
p.plot()


# In[85]:

#Other selected visualizations
df1 = kershaw_df1[['SO', 'BB']]
df1.plot.hist(alpha=0.5)
plt.title('Walk and Strikeout Histogram')


# In[7]:

df2 = kershaw_df1[['GB', 'FB', 'LD', 'SO']]
df2.plot.area()
plt.title('Share of Outcomes Over Time')


# In[108]:

kershaw_df1.plot.hexbin(x='px', y='pz', gridsize=25)
plt.title('Pitch Location')


# In[4]:

from pandas.tools.plotting import andrews_curves
df3 = kershaw_df1[['ER_Class', 'sz_top', 'sz_bot', 'pitch_con', 'spin', 'norm_ht', 'tstart', 'vystart', 'ftime', 'pfx_x',
                  'pfx_z', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'start_speed', 'px', 'pz', 'sb']]
andrews_curves(df3, 'ER_Class')
plt.title('Andrews Curve for Earned Run Classification on Pitchfx')


# In[5]:

df8 = kershaw_df1[['ER_Class', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD',
                  'PU', 'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24', 
                   'Fourseam', 'Sinker', 'Change', 'Slider', 'Curve', 'Slow_Curve']]
andrews_curves(df8, 'ER_Class')
plt.title('Andrews Curve for Earned Run Classification on Bbref')


# In[114]:

kershaw_df1['K_Means'] = kershaw_df1['K_Means'].astype('category')


# In[141]:

fig = plt.figure(figsize=(30, 30), dpi=100)


# In[142]:

from pandas.tools.plotting import parallel_coordinates
df4 = kershaw_df1[['K_Means', 'sz_top', 'sz_bot', 'pitch_con', 'spin', 'norm_ht', 'tstart', 'vystart', 'ftime', 'pfx_x',
                  'pfx_z', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'start_speed', 'px', 'pz', 'sb']]
parallel_coordinates(df4, 'K_Means')
plt.title('Parallel Coordinates for Pitchfx Data')


# In[143]:

df5 = kershaw_df1[['K_Means', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD',
                  'PU', 'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24', 
                   'Fourseam', 'Sinker', 'Change', 'Slider', 'Curve', 'Slow_Curve']]
parallel_coordinates(df5, 'K_Means')
plt.title('Parallel Coordinates for Bbref Data')


# In[144]:

kershaw_df1['Hierarchical'] = kershaw_df1['Hierarchical'].astype('category')


# In[145]:

df6 = kershaw_df1[['Hierarchical', 'sz_top', 'sz_bot', 'pitch_con', 'spin', 'norm_ht', 'tstart', 'vystart', 'ftime', 'pfx_x',
                  'pfx_z', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'start_speed', 'px', 'pz', 'sb']]
parallel_coordinates(df6, 'Hierarchical')
plt.title('Parallel Coordinates for Pitchfx Data')


# In[146]:

df7 = kershaw_df1[['Hierarchical', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD',
                  'PU', 'Unk', 'GSc', 'SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 'aLI', 'WPA', 'RE24', 
                   'Fourseam', 'Sinker', 'Change', 'Slider', 'Curve', 'Slow_Curve']]
parallel_coordinates(df7, 'Hierarchical')
plt.title('Parallel Coordinates for Bbref Data')


# In[ ]:

#Run random forest model


# In[35]:

X = kershaw_df1[['IP', 'H', 'BB', 'SO', 'HR', 'HBP', 'BF', 'Pit', 'Str', 'StL', 'StS', 'GB', 'FB', 'LD',
                  'PU', 'Unk','SB', 'CS', 'PO', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'ROE', 
                   'Fourseam', 'Sinker', 'Change', 'Slider', 'Curve', 'Slow_Curve', 'sz_top', 'sz_bot', 'pitch_con', 'spin', 
      'norm_ht', 'tstart', 'vystart', 'ftime', 'pfx_x','pfx_z', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'start_speed', 
      'px', 'pz', 'sb']]
#Exclude runs and earned runs


# In[36]:

Y = kershaw_df1[['ER_Class']]


# In[37]:

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# In[38]:

model = RandomForestClassifier(n_estimators=100,random_state=0)


# In[39]:

model.fit(X_train, Y_train)


# In[40]:

class_predict = model.predict(X_test)


# In[41]:

from sklearn.metrics import accuracy_score
print (accuracy_score(Y_test, class_predict))


# In[42]:

importances = model.feature_importances_


# In[43]:

std = np.std([model.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:




# In[ ]:



