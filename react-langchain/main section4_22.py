from dotenv import load_dotenv
load_dotenv()
from langchain.agents import tool #will be using this decorator above "get_text_length" func
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import List, Union
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.schema import AgentAction
from langchain.schema import AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str

@tool
def get_text_length(text:str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")

if __name__ == "__main__":
    print("Hell ReAct LangChain!")
    tools = [get_text_length]

    template= """ 
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names=", ".join([t.name for t in tools])
    )
    
    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    intermediate_steps = []
     
    agent = (
        {
            "input": lambda x: x["input"], 
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "what is the length in characters of the text: DOG ?", 
         "agent_scratchpad": intermediate_steps}
    )
    
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))
    
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "what is the length in characters of the text: DOG?", 
         "agent_scratchpad": intermediate_steps}
    )
    
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)