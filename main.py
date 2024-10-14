import os
from dotenv import load_dotenv

# import googlegenai from langchain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

# from langchain.agents import initialize_agent, AgentType


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-exp-0827",
        api_key=os.environ["GOOGLE_API_KEY"],
        # other params...
    )

    # llm = ChatOllama(model="tinyllama", temperature=0)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    query = "Yapay zeka bizi ele mi geçirecek?"

    # ------- standart query --------

    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke({})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # ------- rag langchainhub --------

    # retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # combine_docs_chain = create_stuff_documents_chain(llm, retrival_qa_chat_prompt)
    # retrival_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )

    # result2 = retrival_chain.invoke(input={"input": query})

    # print(result2)

    # ------- custom template rag --------

    template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer
        {context}
        Question: {question}
        Helpful Answer:
    """

    custom_rag_propmpt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_propmpt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)
