import sys,os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact    
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def get_data_transformation_object(self) -> Pipeline:
        """
        It initializes the KNN imputer with the specified parameters and returns a pipeline object with the 
        knn imputer object as the first step.
        Args: cls: DataTransformation
        Returns: Pipeline object with KNN imputer
        """
        logging.info("Creating data transformation pipeline")
        try:
            imputer: KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Imputer object created")
            processor: Pipeline=Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Data Transformation started")
        try:
            # Load the training and testing data
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            # Separate features and target variable
            X_train = train_df.drop(columns=[TARGET_COLUMN, '_id'],axis=1)
            y_train = train_df[TARGET_COLUMN]
            y_train = y_train.replace(-1,0)

            X_test = test_df.drop(columns=[TARGET_COLUMN,'_id'],axis=1)
            y_test = test_df[TARGET_COLUMN]
            y_test = y_test.replace(-1,0)

            preprocessor= self.get_data_transformation_object()
            preprocessor_object=preprocessor.fit(X_train)
            transformed_X_train=preprocessor_object.transform(X_train)
            transformed_X_test=preprocessor_object.transform(X_test)
            logging.info("Data transformation completed")

            train_arr=np.c_[transformed_X_train, np.array(y_train)]
            test_arr=np.c_[transformed_X_test, np.array(y_test)]
            logging.info("Train and test data transformed")

            # Save the transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            #preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data transformation artifact created")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e