from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agents.loader_agent import loader_tool
from agents.eda_agent import eda_tool
from agents.model_agent import model_tool
from agents.subagent_1 import data_loader_analyser
from agents.subagent_2 import model_selector_tool
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def main():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, openai_api_key=api_key)
    tools = [data_loader_analyser, model_selector_tool]

    controller = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        verbose=True
    )

    filepath = input("Enter CSV dataset filepath: ")
    result = controller.invoke({
        "input": {
            "filepath": filepath  
        }
    })
    print("\n===== subagent 1 RESULT =====")
    print(result)

    print("EDA completed successfully. Proceeding to model training and evaluation...")
    model_result = controller.invoke({
        "input": {
            "filepath": filepath,
            "target": result['output']['target']
        }
    })

    print("\n===== MODEL RESULT =====")
    print(model_result)

    
if __name__ == "__main__":
    main()
