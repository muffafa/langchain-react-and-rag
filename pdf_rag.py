import os
from dotenv import load_dotenv

# import googlegenai from langchain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

# from langchain.agents import initialize_agent, AgentType

if __name__ == "__main__":
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-exp-0827",
        api_key=os.environ["GOOGLE_API_KEY"],
        # other params...
    )

    # Load the document
    pdf_path = "/Users/muffafa/Desktop/react-langchain-final-0/transcription_result.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator="\n"
    )

    docs = text_splitter.split_documents(documents=documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_transcription")

    new_vectorestore = FAISS.load_local(
        "faiss_index_transcription", embeddings, allow_dangerous_deserialization=True
    )

    query = "Yapay zeka bizi ele mi geçirecek?"

    retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrival_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)
