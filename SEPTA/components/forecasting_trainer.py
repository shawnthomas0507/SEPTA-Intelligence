from prophet import Prophet
import pandas as pd 
import numpy as np
from pathlib import Path
import os 
import joblib
import mlflow



class Forecast_Trainer:
    
    def __init__(self,path_to_data:str,path_to_store:str):
        self.data_source=path_to_data
        self.model_store=path_to_store

    
    def data_loading(self)->pd.DataFrame:
        df=pd.read_csv(self.data_source)
        return df 
    
    def train_forecaster(self):

        df=self.data_loading()
        routes=df['Route'].unique()

        with mlflow.start_run(run_name="Forecast_Training_All_Routes"):

            for route in routes:
                with mlflow.start_run(run_name=f"Route_{route}", nested=True):

                    df_route = df[df["Route"] == route][["ds", "y"]]
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )

                    model.fit(df_route)
                    os.makedirs(self.model_store,exist_ok=True)
                    model_path = os.path.join(self.model_store, f"route_{route}.pkl")

                    joblib.dump(model,os.path.join(self.model_store,f"route_{route}.pkl"))
                    mlflow.log_param("route",route)
                    mlflow.log_artifact(model_path, artifact_path="models")


                       
    
