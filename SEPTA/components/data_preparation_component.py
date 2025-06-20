import os.path
import pandas as pd 
import numpy as np
from pathlib import Path
import os 
from SEPTA.utils.func import to_csv


class DataPreparationComponent:

    def __init__(self,path_to_data: str,path_to_store: str):
        self.path=path_to_data
        self.path_to_store=path_to_store
    
    def start_preparation(self) -> pd.DataFrame:
        try:
            df=pd.read_csv(self.path)
            df["datetime"] = pd.to_datetime(df["Calendar_Year"].astype(str) + "-" + df["Calendar_Month"].astype(str) + "-01")
            df=df.drop(['Calendar_Year','Calendar_Month','Source','ObjectId'],axis=1)
            temp=df
            temp['ds']=temp['datetime']
            temp['y']=temp['Average_Daily_Ridership']
            temp=temp.drop(['Average_Daily_Ridership','datetime'],axis=1)

            return temp 
        except Exception as e:
            raise e 


    def save_train_object(self):
        try:

            df=self.start_preparation()
            file_path = os.path.join("artifacts", self.path_to_store)
            to_csv(df,file_path)

            return "saved train file successfully"
        
        except Exception as e:
            raise e





