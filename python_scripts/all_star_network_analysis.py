import pandas as pd
import numpy as np
import os
import networkx as nx
import community


# Inspired by Complex Network Analysis in Python (Dmitry Zinoviev)
def ingest_and_create_dataframe():
    # Read in files from Lahman database
    all_star_df = pd.read_csv('AllStarFull.csv')
    people_df = pd.read_csv('People.csv')

    all_star_df = all_star_df.loc[all_star_df['yearID'] >= 1970]
    all_star_df['yearID'] = all_star_df['yearID'].astype('str')
    all_star_df['yearID'] = all_star_df['yearID'].str[2:4]
    all_star_df = all_star_df[['playerID', 'yearID']]

    people_df['name'] = people_df['nameFirst'] + ' ' + people_df['nameLast']
    people_df = people_df[['playerID', 'name']]

    all_star_df = pd.merge(all_star_df, people_df, how='inner', on='playerID')
    all_star_df.drop('playerID', 1, inplace=True)
    all_star_df.columns = ['yearID', 'name']

    all_game_ids = list(set(all_star_df['yearID'].tolist()))
    base_df = pd.DataFrame({'yearID': all_game_ids})
    all_names = list(set(all_star_df['name'].tolist()))

    for name in all_names:
        print(name)
        temp_df = all_star_df.loc[all_star_df['name'] == name]
        temp_df['name'] = 1
        temp_df.rename(columns={'name': name}, inplace=True)
        base_df = pd.merge(base_df, temp_df, how='left', on='yearID')

    base_df.fillna(value=0, inplace=True)
    base_df.to_csv('all_star_df.csv', index=False)
    return base_df


def prepare_network(df):
    df.set_index('yearID', inplace=True)

    # Create co-occurrence matrix
    cooc = df.dot(df.T) * (1 - np.eye(df.shape[0]))
    cooc.to_csv('cooc.csv')

    slicing = 3
    weights = cooc[cooc >= slicing]
    weights = weights.stack()
    weights = weights / weights.max()
    cd_network = weights.to_dict()
    cd_network = {key: float(value) for key, value in cd_network.items()}

    player_network = nx.Graph()
    player_network.add_edges_from(cd_network)
    nx.set_edge_attributes(player_network, 'weight', cd_network)

    partition = community.best_partition(player_network)
    nx.set_node_attributes(player_network, 'part', partition)

    if not os.path.isdir('results'):
        os.mkdir('results')

    with open('results/player_network.graphml', 'wb') as ofile:
        nx.write_graphml(player_network, ofile)
    return


if __name__ == "__main__":
    network_df = ingest_and_create_dataframe()
    prepare_network(network_df)
