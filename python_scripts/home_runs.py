import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
from time import sleep
from sklearn.preprocessing import MinMaxScaler


def create_aggregate_file(df):
    grouped = pd.DataFrame(df.groupby('yearID')['HR'].sum())
    grouped.reset_index(inplace=True)
    grouped.columns = ['year', 'home_runs']
    grouped[['year', 'home_runs']] = grouped[['year', 'home_runs']].astype('int')
    return grouped


def find_mean_and_variance(df, year, rolling_period, excluded_years):
    cutoff = year - rolling_period
    df_lag1 = df.loc[df['year'] == cutoff - 1]
    df_lag2 = df.loc[df['year'] == cutoff - 2]

    df = df.loc[(df['year'] >= cutoff) & (df['year'] < year)]
    len_1 = len(df)
    df = df.loc[~df['year'].isin(excluded_years)]
    len_2 = len(df)

    diff = len_1 - len_2
    if diff == 1:
        df = pd.concat([df_lag1, df])
    elif diff == 2:
        df = pd.concat([df_lag2, df])

    mean = df['home_runs'].mean()
    var = df['home_runs'].std()
    return mean, var


def scrape_2019_home_runs():
    master_df = pd.DataFrame()
    for page in range(1, 27):
        url = 'https://www.foxsports.com/mlb/stats?season=2019&category=BATTING&group=1&sort=7&time=0&pos=0&qual=1&' \
              'sortOrder=0&splitType=0&page={0}&statID=0'.format(page)
        page = requests.get(url)
        soup = BeautifulSoup(page.text)
        table = soup.findAll('tr')
        data = ([[td.getText() for td in table[i].findAll('td')] for i in range(len(table))])
        df = pd.DataFrame(data)
        master_df = master_df.append(df)
        sleep(5)
    master_df.to_csv('2019_hr.csv')
    return master_df


def fit_pdf_and_cdf():
    df_2019 = pd.read_csv('2019_hr.csv')
    hr_total_2019 = df_2019['8'].sum()
    batting_df = pd.read_csv('baseballdatabank-2019.2/core/Batting.csv')
    batting_df = create_aggregate_file(batting_df)
    df_2019 = pd.DataFrame({'home_runs': [hr_total_2019], 'year': [2019]})
    batting_df = pd.concat([batting_df, df_2019], axis=0)
    batting_df.to_csv('full_data_df.csv', index=False)

    pdf_df = pd.DataFrame()
    for year in range(1980, 2020):
        temp_items = find_mean_and_variance(batting_df, year, 20, excluded_years=[1981, 1994])
        temp_batting_df = (batting_df.loc[batting_df['year'] == year]).reset_index(drop=True)
        temp_hr_value = temp_batting_df['home_runs'][0]
        temp_pdf = norm(loc=temp_items[0], scale=temp_items[1]).pdf(temp_hr_value)
        temp_cdf = norm(loc=temp_items[0], scale=temp_items[1]).cdf(temp_hr_value)
        temp_pdf_df = pd.DataFrame({
            'year': [year],
            'home_runs': [temp_hr_value],
            'rolling_mean_hr': temp_items[0],
            'rolling_std': temp_items[1],
            'pdf': [temp_pdf * 100],
            'cdf': [temp_cdf]
        })
        pdf_df = pdf_df.append(temp_pdf_df)

    pdf_df['var_from_average'] = abs(pdf_df['home_runs'] - pdf_df['rolling_mean_hr'])
    pdf_df.to_csv('pdf_df.csv', index=False)
    return


def calculate_summary_stats():
    batting_df = pd.read_csv('baseballdatabank-2019.2/core/Batting.csv')
    batting_df = batting_df[['yearID', 'HR', 'AB']]
    batting_df.columns = ['year', 'home_runs', 'at_bats']

    df_2019 = pd.read_csv('2019_hr.csv')
    df_2019 = df_2019[['8', '3']]
    df_2019['year'] = '2019'
    df_2019.rename(columns={'8': 'home_runs', '3': 'at_bats'}, inplace=True)
    df_2019.dropna(inplace=True)

    master_df = pd.concat([batting_df, df_2019], axis=0)
    master_df.to_csv('all_home_runs.csv', index=False)

    master_df = master_df.loc[master_df['at_bats'] >= 150]
    grouped = master_df.groupby('year').agg({'home_runs': ['mean', 'median', 'std']})
    grouped.reset_index(inplace=True)
    grouped.columns = grouped.columns.droplevel()
    grouped.columns = ['year', 'mean', 'median', 'std']

    scalar = MinMaxScaler()
    grouped['mean_scaled'] = scalar.fit_transform(grouped['mean'].values.reshape(-1, 1))
    grouped['median_scaled'] = scalar.fit_transform(grouped['median'].values.reshape(-1, 1))
    grouped['std_scaled'] = scalar.fit_transform(grouped['std'].values.reshape(-1, 1))

    q = pd.DataFrame(master_df.groupby('year')['home_runs'].quantile(q=np.linspace(.10, .90, 9)))
    q.reset_index(inplace=True)
    q.columns = ['year', 'quantile', 'home_runs']
    q = q.pivot(index='year', columns='quantile', values='home_runs')
    q.reset_index(inplace=True)

    q.columns = ['year', 'quantile_0.1', 'quantile_0.2', 'quantile_0.3', 'quantile_0.4', 'quantile_0.5',
                 'quantile_0.6', 'quantile_0.7', 'quantile_0.8', 'quantile_0.9']

    grouped = pd.merge(grouped, q, how='left', on='year')
    grouped.to_csv('yearly_summary.csv', index=False)
    return


if __name__ == "__main__":
    scrape_2019_home_runs()
    calculate_summary_stats()
    fit_pdf_and_cdf()
