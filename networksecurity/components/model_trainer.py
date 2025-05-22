import os,sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils.utils import evaluate_model

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object, save_numpy_array_data, load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'='*20} Model Trainer {'='*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall", recall_score)
            mlflow.sklearn.log_model(best_model, "model")
        
    def train_model(self,x_train,y_train,x_test,y_test):
        models={
            'Logistic Regression': LogisticRegression(verbose=1),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(verbose=1),
            'Gradient Boosting': GradientBoostingClassifier(verbose=1),
            'AdaBoost': AdaBoostClassifier()   
        }
        params={
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "Random Forest": {
                'n_estimators': [8,16,32,64,128,256],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['auto','sqrt', 'log2', 'None'],
            },
            "Gradient Boosting": {
                'n_estimators': [8,16,32,64,128,256],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_features': ['sqrt', 'log2'],
                'subsample': [0.6,0.7,0.8, 0.9, 1.0],
                'criterion': ['friedman_mse', 'squared_error'],
                'loss': ['log_loss', 'exponential'],
            },
            "AdaBoost": {
                'n_estimators': [8,16,32,64,128,256],
                'learning_rate': [0.01, 0.1, 0.5,0.002],
            },
            "Logistic Regression": {}
        }
        model_report: dict=evaluate_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            models=models,
            params=params
        )

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(x_train)
        classification_train_metric=get_classification_score(y_true=y_train, y_pred=y_train_pred)

        #Track the MLflow
        self.track_mlflow(best_model,classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model,classification_test_metric)

        preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_model=NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_model)

        #Model trainer artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Best model found: {best_model_name} with score: {best_model_score}") 
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading the transformed training and testing data")
            train_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Splitting the data into features and target")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Loading the preprocessor object")
            return self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            
            
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
