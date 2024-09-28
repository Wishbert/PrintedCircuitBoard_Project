import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.cluster import AgglomerativeClustering

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')



class Grey_Color_Model(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X,y=None):
        return self

    def transform(self,X,y=None):
        transformed_data = X

        # I determined that this would be a great way to determine grey pixels.
        stdv_vec = transformed_data.stdv
        avg_vec = transformed_data.avg
        transformed_data['IsGrey'] = np.where(stdv_vec<=14,1,0)*np.where(np.logical_and(avg_vec>= 85, avg_vec <= 190),1,0)

        data_to_cluster = transformed_data.query('IsGrey==1')

        return data_to_cluster[['X','Y']]


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,transformed_data):
        try:
            logging.info('Building Modeling PipeLine')

            modeling_pipeline = Pipeline(
                steps=[
                    ('Grey_Color_Model', Grey_Color_Model()),
                    ('AgglomerativeClustereing',AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=12,
                        linkage = "single"
                            )
                    )
                ]
            )

            logging.info('Training the Model')
            modeling_pipeline.fit_predict(transformed_data)
            

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = modeling_pipeline
            )

            logging.info('Saved Model')
            return modeling_pipeline

        except Exception as e:
            raise CustomException(e,sys)