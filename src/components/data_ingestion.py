import os
import sys 
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig, Grey_Color_Model


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion method or component')
        try:
            relative_path = os.path.join('../../','notebook/data/TestPad_PCB_XYRGB_V2.csv')
            df = pd.read_csv(relative_path) 
            logging.info('Read the Raw Dataset as a DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info('Data Ingestion Complete')

            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e, sys)


def Orchestrate():
    obj = DataIngestion()
    data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    transformed_data = data_transformation.initiate_data_transformation(data_path)

    get_grey_pixels = Grey_Color_Model()
    grey_pixels_data = get_grey_pixels.fit_transform(transformed_data)

    model_trainer_obj = ModelTrainer()
    model_output = model_trainer_obj.initiate_model_trainer(transformed_data=grey_pixels_data)

    return model_output, grey_pixels_data


if __name__ == "__main__":
    Orchestrate()
