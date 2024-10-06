import pandas as pd

import seaborn as sns
sns.set_theme(rc={'figure.figsize':(10,8)})
import matplotlib.pyplot as plt
from plotly import express as px

from src.components import data_ingestion as din
from src.logger import logging



class classification:

    def __init__(self) -> None:

        self.cluster_groups, self.grey_pixel_data = din.Orchestrate()
        self.data_with_cluster_and_count = pd.DataFrame()


    def _classify_pixels(self):

        self.grey_pixel_data['cluster_groups'] = self.cluster_groups
        clust_pixel_number = self.grey_pixel_data.cluster_groups.value_counts()

        cluster_pixel_number_df = pd.DataFrame({'cluster_groups':clust_pixel_number.index, 'number_pixel':clust_pixel_number.values})
        self.data_with_cluster_and_count = self.grey_pixel_data.merge(cluster_pixel_number_df, how='left', on=['cluster_groups'])

        # I decided that 360 is my cutoff.
        self.data_with_cluster_and_count = self.data_with_cluster_and_count.query('number_pixel >= 360 and number_pixel <= 1000') 
        return self.data_with_cluster_and_count

    def _get_average_coordinates(self):
        return self.data_with_cluster_and_count.groupby(by = 'cluster_groups',as_index=False)[['X','Y']].mean()


    def _make_sure_classify_has_run(funct):
        def wrapper(self):
            if self.data_with_cluster_and_count.empty:
                self._classify_pixels()
            
            output_value = funct(self)
            return output_value
        return wrapper


    @_make_sure_classify_has_run
    def test_pad_coordinates(self):

        coordinates = self._get_average_coordinates()
        return coordinates


    @_make_sure_classify_has_run
    def plot_pixels(self):
        
        fig = px.scatter(
            data_frame=self.data_with_cluster_and_count,
            x='X',
            y='Y',
            color='cluster_groups'
        )
        fig.show()


    @_make_sure_classify_has_run
    def plot_test_pad_coordinates(self):

        coordinates = self._get_average_coordinates()

        fig = px.scatter(
            data_frame = coordinates,
            x='X',
            y='Y',
            color='cluster_groups'
        )
        fig.show()

