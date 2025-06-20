import pandas as pd
import os 
from pymongo import MongoClient
from SEPTA.utils.func import to_csv



class DataIngestionComponent:

    def __init__(self,path_to_store_ds: str,mongo_url :str,database_name: str,collection_name: str):
        self.path=path_to_store_ds
        self.mongoid=mongo_url
        self.db_name=database_name
        self.coll_name=collection_name
    
    def ingest_data(self):
        try:
            client = MongoClient(host=self.mongoid)
            db = client[self.db_name]
            collection = db[self.coll_name]

            documents = list(collection.find())
            attributes_data = [doc.get("attributes", {}) for doc in documents]
            df = pd.DataFrame(attributes_data)

            file_path = os.path.join("artifacts", self.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            to_csv(df,file_path)
            return "Successfully retrieved"

        except Exception as e:
            raise e


        