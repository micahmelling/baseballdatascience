# Import libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd


def pitch_fx_scraper(base_url):
    # Isolate the text data
    data = base_url.text

    # Create beautiful soup object from the data
    soup = BeautifulSoup(data)

    # Put all the links in a list
    results = []
    for gid in soup.find_all('a'):
        results.append(gid.get('href'))

    # Delete unnecessary elements of the list
    del results[0:21]

    # Extract the Game ID from each link
    results1 = [i.split('&prevGame=', 1)[1] for i in results]

    # Concatenate strings to create URLs
    results2 = list(map('http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=477132&game={0}'.format, results1))

    # Scrape each link
    results3 = []
    for i in results2:
        results3.append(requests.get(i))

    # Grab the text for each link
    results4 = []
    for i in results3:
        results4.append(i.text)

    # Make each a beautiful soup object
    results5 = []
    for i in results4:
        results5.append(BeautifulSoup(i))

    # Extract column headers
    sample = results5[1]
    column_headers = [th.getText() for th in
                      sample.findAll('tr', limit=2)[0].findAll('th')]

    # Define data rows
    results6 = []
    for i in results5:
        results6.append(i.findAll('tr'))

    # Get data from table
    results7 = []
    for j in results6:
        results7.append([[td.getText() for td in j[i].findAll('td')]
                for i in range(len(j))])

    # Convert to dataframe
    results8 = []
    for i in results7:
        results8.append(pd.DataFrame(i))

    for i in results8:
        i.columns = column_headers

    return


if __name__ == "__main__":
    # Get the URL that we'll use to construct game IDs
    r = requests.get("http://www.brooksbaseball.net/tabs.php?player=477132&p_hand=1&ppos=1&cn=200&compType=none&gFilt=&time=month&minmax=ci&var=gl&s_type=2&startDate=03/30/2007&endDate=10/22/2016&balls=1&strikes=1&b_hand=1")
    pitch_fx_scraper(r)