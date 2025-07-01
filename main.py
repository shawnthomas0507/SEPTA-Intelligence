
from SEPTA.components.data_ingestion_component import DataIngestionComponent
from SEPTA.components.data_preparation_component import DataPreparationComponent
from SEPTA.components.forecasting_trainer import Forecast_Trainer
from SEPTA.components.septa_agent import SEPTAAgent
from agents import Agent,Runner,trace, OpenAIChatCompletionsModel,function_tool
from SEPTA.utils.func import load_yaml
import asyncio


"""
common=load_yaml()

ingest=DataIngestionComponent(path_to_store_ridership=common['FILE_PATH_FOR_RIDERSHIP'],path_to_store_route=common['FILE_PATH_FOR_ROUTE'],mongo_url=common['MONGO_URL'],collection_name_for_ridership=common['COLLECTION_NAME_FOR_RIDERSHIP'],collection_name_for_route=common['COLLECTION_NAME_FOR_ROUTE'],database_name=common['DB_NAME'],file_path_for_route_info=common['FILE_PATH_FOR_ROUTE_INFO'])

op=ingest.ingest_data()
print(op)

prepare=DataPreparationComponent(path_to_data=f"Artifacts/{common['FILE_PATH_FOR_RIDERSHIP']}",path_to_store=common['FILE_PATH_TRAIN'])


op=prepare.save_train_object()
print(op)


tr=Forecast_Trainer(path_to_data="C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\train.csv",
                    path_to_store="C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\models")

tr.train_forecaster()

"""

message=input("Enter a query")
sa=SEPTAAgent(path_to_data="C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\train.csv",message=message)





if __name__=="__main__":
    res=sa.run_agent()
    print(res.final_output)
