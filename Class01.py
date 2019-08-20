import os
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

data_path = ['data']

def main():
    filepath = os.sep.join(data_path + ['Iris_Data.csv'])
    data = pd.read_csv(filepath)

    # Data head
    print(data.head())

    # Number of rows
    print(data.shape[0])

    # Column names
    print(data.columns.tolist())

    # Data types
    print(data.dtypes)

    # Remove front string from species data to just keep species name
    data['species'] = data.species.str.replace('Iris-', '')
    # alternatively
    # data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))
    print(data.head())

    # Count of each species
    print(data['species'].value_counts())

    # The mean, median, and quantiles and ranges (max-min) for each petal and
    # sepal measurement.
    range = data.max(numeric_only=True)-data.min(numeric_only=True)
    print(data.describe().append(pd.Series(range, name='range')))

    # The mean of each measurement
    print(data.groupby('species').mean())

    # The median of each of these measurements
    print(data.groupby('species').median())

    # Applying multiple functions at once - 2 methods
    print(data.groupby('species').agg(['mean', 'median']))      # passing a list of recognized strings
    print(data.groupby('species').agg([np.mean, np.median]))    # passing a list of explicit aggregation functions

    # If certain fields need to be aggregated differently, we can do:
    agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'species'}
    agg_dict['petal_length'] = 'max'
    pprint(agg_dict)
    pprint(data.groupby('species').agg(agg_dict))

    if False:
        # A simple scatter plot with Matplotlib
        ax = plt.axes()
        ax.scatter(data.sepal_length, data.sepal_width)
        # Label the axes
        ax.set(xlabel='Sepal Length (cm)',
               ylabel='Sepal Width (cm)',
               title='Sepal Length vs Width');

    if False:
        # Create a single plot with histograms for each feature
        ax = data.plot.hist(bins=25, alpha=0.5)
        ax.set_xlabel('Size (cm)');

    if False:
        # To create four separate plots, use Pandas `.hist` method
        axList = data.hist(bins=25)

        # Add some x- and y- labels to first column and last row
        for ax in axList.flatten():
            if ax.is_last_row():
                ax.set_xlabel('Size (cm)')

            if ax.is_first_col():
                ax.set_ylabel('Frequency')

    if False:
        color = dict(boxes='DarkGreen', whiskers='DarkOrange',
            medians='DarkBlue', caps='Gray')
        ax = data.plot.box(color=color, sym='r+')

    if False:
        sns.set(style="whitegrid")
        ax = sns.boxplot(x=data.sepal_width)

    if True:
        # First we have to reshape the data so there is
        # only a single measurement in each column
        plot_data = (data
             .set_index('species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

        # Now plot the dataframe from above using Seaborn
        sns.set_style('white')
        #sns.set_context('notebook')
        sns.set_palette('dark')

        f = plt.figure(figsize=(6,4))
        sns.boxplot(x='measurement', y='size',
                    hue='species', data=plot_data);

    plt.show()

if __name__== "__main__":
    main()
