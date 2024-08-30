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

# Tool decotrator is a LangChain utility function that will take this function
# and create a langchain tool from it. Its going to plug in the name of the function,
# what it receives as arguments, what is returns, its description, and populate it
# in the LangChain tool class and this will be used by the LLM reasoning engine to decide
# wheher to tuse this tool.
@tool
def get_text_length(text:str) -> int:
    # We will start by defining what this function is going to do
    # description is very important because this is going to help the LLM decide
    # if its going to use this tool or not in its reasoning agent 
    # And we will soon see how this actually works.
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
    #print(get_text_length(text="Dog"))

    # List of LC tools, which is not going to be populated only with "get_text_length"

    tools = [get_text_length]

    # Now this list of tools, we are going to supply to our ReAct agent
    # Previously: The agent selected the correct tool to use from a prompt and it was sort 
    # of like magic. So, now we are going to dive deep and to see how its actually implemented
    # and whats happening under the hood.

    # used from LangChain Hub which is part of LangSmith. LC hub is a marketplace from prompts.
    # Its all free and people share their very well-crafted prompts, including the prompts for 
    # this React agent. Now below prompt was created by Harrison Chase, the creator of LangChain.
    # Every can submit prompts and share them on the LangChain Hub.
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
    Thought:
    """

    # Ref page 24 and 25 in my notes (good doc: https://docs.google.com/document/d/1RjIyz6naTouo4sFrAxAe7B24g1hav0rukqTLaKD4M-A/edit?pli=1)
    # From quick analysis of above prompt defined in "template" we can calssify this prompt as chain of 
    # thought prompt bcoz we are asking the LLM to tell us how its thinking and how its 
    # coming up with its answer. Its also a few shot prompt bcoz we supply examples and telling the LLM
    # how do we want our format otput to be with. And this is an implementation of the ReactPaper
    # reasoning and acting.

    #from above template, we want to create a LangCahin prompt template
    # Partial: Will pouplate and plug in the values of the placeholders that we
    # already have bcoz we know we have the tools and tools names. So, we can plug it
    # into our prompt right now and the input variable placeholder will come dynamically
    # when we run this chain (so, its going to be from the user).
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names=", ".join([t.name for t in tools])
    )
    
    # We want to plug-in stop argument with "\nObservation" it will tell llm to stop
    # generating words and to finish working once its outputted the "\nObservation"
    # token. Why do we need it? bcoz if wont put the stop token then the LLM would continue
    # to generate text and its going to guess one word after anohter observation.
    # And the observation is the result of the tool and this is something that will come
    # from running our tool. and if it will come from the LLM then it will simply have hallucinations.
    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])

    # Below pipe symbol is related to using LangChain expression Language (LCEL)
    # Pipe operator takes the output from left-side and plugs it into the input of
    # the right side. Here its going to run the prompt step which is going to
    # output us a prompt value and its going to input into the LLM. Bcoz LLMs receives
    # a prompt value.
    # But, below prompt is not complete yet, bcoz we havent supplied it with our question
    # that we want to ask our agent to do. So, we need to plug it in its input.
    # And what does it input? It comes in the form of dictionary with the keyword, which is 
    # going to relace the placeholders and the value is going to be whats going to replace
    # those placeholders. So, we partialy initialized this prompt with the tools and tool names,
    # we only need to populate the input placeholder. So we are going to input our
    # prompt with a dictionary thats going to contain the key of input and its value is going
    # to hold eventaully the question that we are going to be asking out agent.

    # Bcoz we want everything to be dynamic, we are not going to pass the prompt that we are going
    # to send the agent. We are going to pass it when we invoke all of this chain. 
    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    # The way that we write here the prompt is super important bcoz our agent is going to receive it
    # and its going from that determine which tool to use and what input will it receive. And even it 
    # will affect the output parsing later when we parse the results of the LLM.
    res = agent.invoke({"input": "what is the length of 'DOG' in characters?'"})

    # Below is what the LLM responded, and this is actually the reasoning engine. The LLM got input prompt
    # and it respoded us with eactly what needs to be run, what tool needs to be selected.
    # And this covers in the ReAct algorithm, the query part, which we plug into the agent and how does the 
    # agne comprise and elobrate prompt which is sent to the LLM - Then the LLM functions as reasoning agent
    # to select the correct tool and returns us a response containing all the information about which tools it 
    # selected and need to be run and now we need to go an parse this output that we just saw belwo.

    # OUTPUT (before adding ReActSingleInputOuputPareser): content="To find the length of the text 'DOG' in characters, I can use the get_text_length tool."
    # OUTPUT (with ReActSingleInputOuputPareser): tool='get_text_length' tool_input="'DOG'" log="I can use the get_text_length tool to find the length of the text 'DOG'. \nAction: get_text_length\nAction Input: 'DOG'"
    print(res)

    # Now we take the output of the LLM in the ReAct agent and simply parse it into the components
    # that we need. So, above ouput we want to pass into something which is called
    # ReActSingleInputOuputParser.

    # We are done with parsing part, now we want to go to execution part
    # Modifing line 118:
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "what is the length in characters of the text: DOG ?"})
    print(agent_step)

    # Now if the agent_step is the type of AgentAction, it will hold all the information of the tool we want to run and
    # execute and get our observation, which is the result of this tool. So, we want to write this logic.

    # If we get back from the output parser the agent_step, which is going to be an instance of class AgentAction,
    # We want to extrapolate the tools to use. So, we will get the tool name, and from the tool name we are going to
    # find from the list of tools, the tools to run. 
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
    
    # OUTPUT
    # Hell ReAct LangChain!
    # WARNING! stop is not default parameter.
    #                 stop was transferred to model_kwargs.
    #                 Please confirm that stop is what you intended.
    # tool='get_text_length' tool_input="'DOG'" log="I can use the get_text_length tool to find the length of the text 'DOG'. \nAction: get_text_length\nAction Input: 'DOG'"
    # tool='get_text_length' tool_input='DOG' log='I need to find the length of the text "DOG" in characters.\nAction: get_text_length\nAction Input: "DOG"'
    # get_text_length enter with text='DOG'
    # observation=3

    # From above OUTPUT we see observation resulting to number 5. Which is the length of the input text 'DOG'
    # which results in get text length = 3

    # Section 4.21: In this video we reviewed a massive part of ReAct implementation of LangChain.
    # We saw how we can start from a user query to count the number of characters in the word DOG.
    # How we plugged it into the agent, then the agent wrapped it up in a special LLM call of ReAct algorithm,
    # sent this LLM cal to GPT-3.5
    # We got back a response from the LLM, which had the thought and all the information of the tool selection.
    # This was actually the reasoning engine of the LLM. We then saw LC parses all of this information and all of this 
    # response, and we swa them how LC transformed it into a tool to run and to execute that tool.