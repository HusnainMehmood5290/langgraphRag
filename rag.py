from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.documents import Document 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"


filePath="Customer Service Dept. Manual.pdf"
embeddings=HuggingFaceEmbeddings()
vector_store=Chroma(collection_name="Chatbot",embedding_function=embeddings,persist_directory="vector_db_dir")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=None, timeout=None, max_retries=2)


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
    prompt=hub.pull("rlm/rag-prompt")

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}


    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()


    response = graph.invoke({"question": "What is EDA stand for?"})
    print(response["answer"])