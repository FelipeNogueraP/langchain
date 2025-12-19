from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    LangChain is a framework for developing applications powered by language models. 
    It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
    """

    summary_template = "Summarize the following text:\n\n{information}\n\nSummary:"

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
        )

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm # Runnable object, setup the first one and the pass it to the second one.
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
