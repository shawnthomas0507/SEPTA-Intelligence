import yaml
import pandas as pd
from dotenv import load_dotenv
import os 
from agents import Agent,Runner,trace, OpenAIChatCompletionsModel,function_tool
from openai import AsyncOpenAI
from typing import Dict
from openai.types.responses import ResponseTextDeltaEvent
import requests
import asyncio
import joblib
from prophet import Prophet
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from langchain_groq import ChatGroq
from tavily import TavilyClient


def load_yaml():
    try:
        with open("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\SEPTA\\yaml_files\\common.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config 
    except Exception as e:
        raise e



def to_csv(df: pd.DataFrame,file_path:str):
    try:
        df.to_csv(file_path,index=False)
    except Exception as e:
        raise e 
    

@function_tool
def get_forecasts(route_number:int,months_ahead:int):
        """  Forecasts ridership numbers of a given route number in given months_ahead from now """

        print("forecasting")
        df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\train.csv")
        last_date = df['ds'].max()
        model=joblib.load(f"C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\models\\route_{route_number}.pkl")
        print(f"route number is {route_number} and months ahead is {months_ahead}")
        future = model.make_future_dataframe(periods=months_ahead, freq='M')
        forecast = model.predict(future)
        plt.figure(figsize=(20, 6))


        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')

        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                        color='blue', alpha=0.2, label='Confidence Interval')

        forecast_start = last_date
        forecast_end = forecast['ds'].max()

        plt.axvspan(forecast_start, forecast_end, color='orange', alpha=0.1, label='Forecast Period')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        plt.xticks(rotation=45)

        plt.title('Ridership Forecast with Highlighted Forecast Period')
        plt.xlabel('Date')
        plt.ylabel('Ridership')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        print("plotting")
        plt.show()
        
        return {"success":"ok"}



@function_tool
def get_information_about_routes(route_number: str,message: str):
     df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\output.csv")
     extracted=df[df['Route']==route_number]
     csv_text = extracted.to_csv(index=False)
     return {"data":csv_text}


@function_tool
def search_web(question: str):
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API"))
    response = tavily_client.search(question)
    return {"response":response}


