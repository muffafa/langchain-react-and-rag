import os
from dotenv import load_dotenv

# import googlegenai from langchain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

from langchain.tools.render import render_text_description
from langchain.agents import tool

# from langchain.agents import initialize_agent, AgentType

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""

    # stripping away non alphabetic characters just in case
    text = text.strip("'\n").strip('""')
    print(f"get_text_length enter with {text=}")
    return len(text)


def try_react_agent(llm):
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
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
        Thought:{{}}
    """

    # Correct the render_text_description call by passing the tools argument
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),  # Pass the tools argument here
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = {"input": lambda x: x["input"]} | prompt | llm | StrOutputParser()

    res = agent.invoke({"input": "How many characters are in the word 'hello'?"})

    print(res)


def translate(llm):

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to Turkish. Translate the user sentence.",
        ),
        ("human", "GIST OF LANGCHAIN"),
    ]

    ai_msg = llm.invoke(messages)

    print(ai_msg.content)


def suummarize(llm):
    summary_template = """
    You are a very good assistant that summarizes texts.
    Please summarize the information in 1 sentece: {information}
    """

    # Create a PromptTemplate object from the template and input variables
    summary_prompt_template = PromptTemplate.from_template(summary_template)

    summary_chain = summary_prompt_template | llm | StrOutputParser()

    summary_information = """
            LangChain is a framework for developing applications powered by language models.
            It provides a simple interface for interacting with a variety of language models,
            including OpenAI's GPT-3 and GPT-4, Anthropic's Claude, and more.
            LangChain also provides a range of tools and utilities for common tasks in language model applications,
            such as text generation, question answering, and summarization.
            """

    summary_message = summary_chain.invoke(input={"information": summary_information})

    print(summary_message)


if __name__ == "__main__":
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-exp-0827",
        stop=["\nObservation"],
        api_key=os.environ["GOOGLE_API_KEY"],
        # other params...
    )

    # llm = ChatOllama(model="tinyllama", temperature=0)

    # translate(llm)
    try_react_agent(llm)
