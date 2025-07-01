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
import folium


load_dotenv()



def load_yaml():
    try:
        with open("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\SEPTA\\config\\common.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config 
    except Exception as e:
        raise e

def load_instructions_yaml():
    try:
        with open("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\SEPTA\\config\\instructions.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config 
    except Exception as e:
        raise e


def to_csv(df: pd.DataFrame,file_path:str):
    try:
        df.to_csv(file_path,index=False)
    except Exception as e:
        raise e 
    
def send_text(body:str,pushover_user=os.getenv("PUSHOVER_USER"),pushover_token=os.getenv("PUSHOVER_TOKEN"),pushover_url="https://api.pushover.net/1/messages.json"):
        print(f"push : {body}")
        payload={"user":pushover_user,"token":pushover_token,"message":body}
        requests.post(pushover_url,data=payload)
        return {"status":"success"}



@function_tool
def get_forecasts(route_number:int,months_ahead:int):

        print("forecasting")
        df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\train.csv")
        last_date = df['ds'].max()
        model=joblib.load(f"C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\models\\route_{route_number}.pkl")
        print(f"route number is {route_number} and months ahead is {months_ahead}")
        future = model.make_future_dataframe(periods=months_ahead, freq='M')
        forecast = model.predict(future)
        forecasted_csv=forecast.to_csv(index=False)
        print(forecasted_csv)

        print("this is working")
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
        
        return {"forecasted_data":forecasted_csv}




@function_tool
def get_forecasts_to_identify_risk(route_number:int,months_ahead:int):

        print(" we have entered risk identifying agent and now we are forecasting")
        print("forecasting")
        df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\train.csv")
        last_date = df['ds'].max()
        model=joblib.load(f"C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\models\\route_{route_number}.pkl")
        print(f"route number is {route_number} and months ahead is {months_ahead}")
        future = model.make_future_dataframe(periods=months_ahead, freq='M')
        forecast = model.predict(future)
        
        try:
            forecasted_csv=forecast.to_csv(index=False)
            historical_csv=df[df['Route']==f'{route_number}'].to_csv(index=False)
        except Exception as e:
             print("failed",e)

        print("function working")
        return {"forecasted_data":forecasted_csv,"historical_data":historical_csv}



@function_tool
def get_information_about_routes(route_number: str,message: str):
     df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\output.csv")
     extracted=df[df['Route']==route_number]
     csv_text = extracted.to_csv(index=False)
     return {"data":csv_text}

@function_tool
def search_web(question: str):
    print("searching")
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API"))
        response = tavily_client.search(question)
    except Exception as e:
         raise e
    return {"response":response}



@function_tool
def send_report(route_at_risk:int,reason_for_risk:str):
     report=f"route:{route_at_risk},reason:{reason_for_risk}"
     res=send_text(body=report)
     if res['status']=="success":
          return {"response":"i have sent you the report"}


@function_tool
def get_all_stop_information(route:int):
    print("getting stop information")

    try:
        df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\route_info.csv")
        df=df[['X','Y','FID','GISDBID','Mode','Route','Direction','Stop_Code','Stop','Lat','Lon']]	
        df = df[df['Route'] == route] 
        print(df.to_csv())
        return {"stop_information":df.to_csv()}
    except Exception as e:
         raise e




async def route_identify(agent:Agent,route:str,df:pd.DataFrame):
    extracted=df[df['Route']==route]
    csv_text = extracted.to_csv(index=False)
    result=Runner.run(agent,csv_text)
    return result.final_output




async def identify_risky_routes(agent: Agent):
    df=pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\output.csv")
    tasks=[asyncio.create_task(route_identify(agent,route=route,df=df))for route in df['Route'].unique()]
    results= await asyncio.gather(*tasks)
    return results




