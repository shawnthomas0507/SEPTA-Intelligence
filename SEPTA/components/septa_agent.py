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
from SEPTA.utils.func import get_forecasts,get_information_about_routes,search_web

load_dotenv()


class SEPTAAgent:
    def __init__(self,path_to_data:str,message:str):
        self.data_source=path_to_data
        self.groq_api_key=os.getenv("GROQ_API_KEY")
        self.groq_base_url="https://api.groq.com/openai/v1"
        self.tavily_api=os.getenv("TAVILY_API")
        self.user_input=message
    
    def set_model(self):
        groq_client=AsyncOpenAI(base_url=self.groq_base_url,api_key=self.groq_api_key)
        llama_3_model =OpenAIChatCompletionsModel(model="llama-3.1-8b-instant",openai_client=groq_client)
        return llama_3_model
    
    def get_data(self)->pd.DataFrame:
        df=pd.read_csv(self.data_source)
        return df 

    
    def set_agents(self):

        instructions="""
        You are an expert agent working for SEPTA (Southeastern Pennsylvania Transportation Authority). 
        You can answer questions about SEPTA routes using the provided data.
        You can add , subtract do anything.

        When a user asks about a specific route:
        1. Extract the route number/ID from their question
        2. Use the get_information_about_routes function to retrieve the data
        3. Analyze the returned data and provide a proper response. 
        4. Only make one function call per question

        If the user asks general questions about the SEPTA route call the search_web function by providing a question to it and use 
        the context provided to answer the user's question.

        Be helpful, informative, and professional in your responses.
        """

        tools=[get_information_about_routes,search_web]

        insight_agent=Agent(
            name="Insight Agent",
            instructions=instructions,
            tools=tools,
            model=self.set_model(),
            handoff_description="Answer user questions about route data"
        )


        tools=[get_forecasts]
        instructions=""" 
        You are an expert agent working for SEPTA. You can forecast ridership data 
            for any route number or answer user queries about routes. 
            When users ask for forecasts, extract the route number
            and months ahead, then use the get_forecasts function. Do not make multiple calls.
            Only forecast for the months that is asked for, nothing else.
            If user asks about insight into routes handoff to the Insight Agent to answer the user query.
            Stick to the given instructions.
        """

        septa_agent=Agent(
            name="SEPTA_AGENT",
            instructions=instructions,
            model=self.set_model(),
            tools=tools,
            handoffs=[insight_agent]
            )
        
        return septa_agent
    
    def run_agent(self):
        model=self.set_agents()
        result =  asyncio.run(Runner.run(model, self.user_input))
        return result