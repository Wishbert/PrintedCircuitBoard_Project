import sys
import os 
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object

class Colour_Features(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.dataframe: pd.DataFrame = pd.DataFrame() 


    def compute_features(self, dataframe:pd.DataFrame):
        logging.info('Computing color std and mean')
        values_of_interest = dataframe[['R','G','B']].values
        return np.std(values_of_interest,axis=1, ddof = 1), np.mean(values_of_interest, axis=1)


    def fit(self, X:pd.DataFrame, y=None):
        self.dataframe = X
        return self

    def transform(self, X:pd.DataFrame):
        try:
            logging.info('Creating Color Features')
            self.dataframe = X
            self.dataframe[['R','G','B']] = self.dataframe[['R','G','B']].values*255
            self.dataframe['stdv'], self.dataframe['avg'] = self.compute_features(self.dataframe)

            return self.dataframe
        
        except Exception as e:
            raise CustomException(e,sys)



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            logging.info('Applying Custom Transformer')
            transformed_data = Pipeline(
                steps=[
                    ('features',Colour_Features())
                ]
            )

            return transformed_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, raw_data_path):

        try:

            logging.info('Read training Data')
            data = pd.read_csv(raw_data_path)

            logging.info('initiating feature engineering object')
            feature_engineering_obj = self.get_data_transformer_object()

            logging.info('Applying Feature Engineering Object')
            transfromed_data = feature_engineering_obj.fit_transform(data)

            logging.info('Saving feature engineering object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=feature_engineering_obj
            )

            return transfromed_data
        except Exception as e:
            raise CustomException(e,sys)