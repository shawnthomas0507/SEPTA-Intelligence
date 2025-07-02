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
from SEPTA.utils.func import get_forecasts,get_information_about_routes,look_up_tavily,get_forecasts_to_identify_risk,load_instructions_yaml,send_report,get_all_stop_information,plot_stops_on_map
load_dotenv()


class SEPTAAgent:
    def __init__(self,path_to_data:str,message:str):
        self.data_source=path_to_data
        self.groq_api_key=os.getenv("GROQ_API_KEY")
        self.groq_base_url="https://api.groq.com/openai/v1"
        self.tavily_api=os.getenv("TAVILY_API")
        self.user_input=message
        self.instructions_yaml_config=load_instructions_yaml()
    
    def set_model(self):
        groq_client=AsyncOpenAI(base_url=self.groq_base_url,api_key=self.groq_api_key)
        llama_3_model =OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile",openai_client=groq_client)
        return llama_3_model
    

    
    def get_data(self)->pd.DataFrame:
        df=pd.read_csv(self.data_source)
        return df 
    
    # <def make_identify_risky_routes_tool(self,risky_routes_agent):
        @function_tool
        async def identify_risky_routes():
            df = pd.read_csv("C:\\Users\\shawn\\OneDrive\\Desktop\\NewProject\\SEPTA_MODEL\\Artifacts\\output.csv")

            async def route_identify(route: str):
                extracted = df[df['Route'] == route]
                csv_text = extracted.to_csv(index=False)
                result = await Runner.run(risky_routes_agent, csv_text)
                print(result.final_output)
                print(f"Finished processing route: {route}")

                return result.final_output

            tasks = [
                asyncio.create_task(route_identify(route))
                for route in df['Route'].unique()
            ]
            results = await asyncio.gather(*tasks)
            return results

        return identify_risky_routes

    
    def set_agents(self):

        instructions=f"{self.instructions_yaml_config['insight_agent']['description']}"

        tools=[get_information_about_routes,look_up_tavily,get_all_stop_information,plot_stops_on_map]

        insight_agent=Agent(
            name="Insight Agent",
            instructions=instructions,
            tools=tools,
            model=self.set_model()
                            )

        tools=[get_forecasts_to_identify_risk,send_report]
        instructions=f"{self.instructions_yaml_config['risky_routes_agent']['description']}"


        risky_routes_agent=Agent(
            name="Risk Identifying Agent",
            instructions=instructions,
            model=self.set_model(),
            tools=tools
        )
        #identify_risky_routes_tool = self.make_identify_risky_routes_tool(risky_routes_agent)


        tools=[get_forecasts]
        instructions=f"{self.instructions_yaml_config['septa_agent']['description']}"

        septa_agent=Agent(
            name="SEPTA_AGENT",
            instructions=instructions,
            model=self.set_model(),
            tools=tools,
            handoffs=[insight_agent,risky_routes_agent]
            )
        
        return septa_agent
    
    def run_agent(self):
        model=self.set_agents()
        result =  asyncio.run(Runner.run(model, self.user_input))
        return result