# Library imports
import pandas as pd
import random


# Define batting average class
class WARSimulation:
    def __init__(self, df, name):
        self.df = df
        self.name = name

    def monte_carlo(self):
        mean = self.df['war'].mean()
        std = self.df['war'].std()
        selections = random.normalvariate(mean, std)
        return selections

    def run_the_simulation(self):
        x = 0
        selection = []
        while x < 100:
            selection.append(self.monte_carlo())
            x += 1

        data = pd.DataFrame({'war': selection})
        data['war'] = data['war'].round(decimals=2)
        data.to_csv(self.name + '_results.csv', index=False)
        return


if __name__ == "__main__":
    # Define dataframes
    hosmer_df = pd.DataFrame({'war': [1.5, -0.4, 3.5, 0.8, 3.6, 1.0]})
    cain_df = pd.DataFrame({'war': [2.0, 3.2, 5.1, 7.2, 2.9]})
    perez_df = pd.DataFrame({'war': [2.9, 4.1, 3.4, 2.3, 2.7]})
    escobar_df = pd.DataFrame({'war': [0.5, 2.7, 3.4, 0.3, 2.5, 0.6, 0.3]})
    gordon_df = pd.DataFrame({'war': [2.0, 2.8, -0.5, 7.2, 6.3, 4.2, 6.6, 2.8, 0.8]})
    moustakes_df = pd.DataFrame({'war': [1.1, 3.1, -0.1, 0.4, 4.4]})

    # Hosmer - simulation
    hosmer_selections = WARSimulation(hosmer_df, 'hosmer')
    hosmer_selections.run_the_simulation()

    # Cain - simulation
    cain_selections = WARSimulation(cain_df, 'cain')
    cain_selections.run_the_simulation()

    # Perez - simulation
    perez_selections = WARSimulation(perez_df, 'perez')
    perez_selections.run_the_simulation()

    # Escobar - simulation
    escobar_selections = WARSimulation(escobar_df, 'escobar')
    escobar_selections.monte_carlo()
    escobar_selections.run_the_simulation()

    # Gordan - simulation
    gordon_selections = WARSimulation(gordon_df, 'gordon')
    gordon_selections.run_the_simulation()

    # Moustakes - simulation
    moustakes_selections = WARSimulation(moustakes_df, 'moustakes')
    moustakes_selections.run_the_simulation()
