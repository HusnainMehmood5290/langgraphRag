from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


filePath="Customer Service Dept. Manual.pdf"
embeddings=HuggingFaceEmbeddings()
vector_store=Chroma(collection_name="Chatbot",embedding_function=embeddings,persist_directory="vector_db_dir")


def laod_and_split(filePath):
    loader=PyPDFLoader(file_path=filePath)
    docs=loader.load()
    spliter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        )
    splitted_docs=spliter.split_documents(docs)
    return splitted_docs


def add_docs(splitted_docs):
    vector_store.add_documents(splitted_docs)


if __name__=="__main__":
    docs=laod_and_split(filePath)
    add_docs(docs)
    vector_store.similarity_search("EDA")