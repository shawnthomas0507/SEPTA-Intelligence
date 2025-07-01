import pandas as pd
import os 
from pymongo import MongoClient
from SEPTA.utils.func import to_csv



class DataIngestionComponent:

    def __init__(self,path_to_store_ridership: str,path_to_store_route: str,file_path_for_route_info:str,mongo_url :str,database_name: str,collection_name_for_ridership: str,collection_name_for_route:str):
        self.path_to_store_ridership=path_to_store_ridership
        self.path_to_store_route=path_to_store_route
        self.mongoid=mongo_url
        self.db_name=database_name
        self.coll_name_for_ridership=collection_name_for_ridership
        self.coll_name_for_route=collection_name_for_route
        self.file_path_for_route_info=file_path_for_route_info
    
    def ingest_data(self):
        try:
            client = MongoClient(host=self.mongoid)
            db = client[self.db_name]
            collection = db[self.coll_name_for_ridership]

            documents = list(collection.find())
            attributes_data = [doc.get("attributes", {}) for doc in documents]
            df = pd.DataFrame(attributes_data)
            df1=pd.read_csv(self.file_path_for_route_info)
            routes_from_df=df['Route'].unique().tolist()
            routes_from_df1=df1['Route'].unique().tolist()
            filtered_routes = [r.strip() for r in routes_from_df if r.strip().isdigit()]
            routes_present=[]
            for i in filtered_routes:
                    if int(i) in routes_from_df1:
                        routes_present.append(i)
            filtered_df = df[df['Route'].isin(routes_present)]
            file_path = os.path.join("artifacts", self.path_to_store_ridership)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            to_csv(filtered_df,file_path)

            return "Successfully retrieved"

        except Exception as e:
            raise e


        