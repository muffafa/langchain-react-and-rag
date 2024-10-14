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


# from langchain.agents import initialize_agent, AgentType

if __name__ == "__main__":
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-exp-0827",
        api_key=os.environ["GOOGLE_API_KEY"],
        # other params...
    )

    loader = TextLoader("blog.txt")
    document = loader.load()

    print("...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"Split into {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    PineconeVectorStore.from_documents(
        texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
