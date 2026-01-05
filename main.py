from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

information = """
LangChain is a framework for developing applications powered by language models. 
It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
"""

# summary_template = "Summarize the following text:\n\n{information}\n\nSummary:"

# summary_prompt_template = PromptTemplate(
#     input_variables=["information"], 
#     template=summary_template
#     )

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOllama(temperature=0, model="gemma3:270m")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

# chain = summary_prompt_template | llm # Runnable object, setup the first one and the pass it to the second one.
# response = chain.invoke(input={"information": information})
# print(response.content)

def main():
    print("Running langchain Agent!")
    result = agent.invoke({"messages": HumanMessage(content="search for 10 job positions for an AI engineer"
    " check only for the last week, in colombia or a remote position and summarize the results.")})
    print(result)


if __name__ == "__main__":
    main()
