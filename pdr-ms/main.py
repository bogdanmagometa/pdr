from typing import List
import dotenv

dotenv.load_dotenv(override=True);

from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langserve import add_routes
import fastapi
from pydantic import BaseModel, Field

# Construct retriever
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma(embedding_function=embeddings)
docstore = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000), 
)

# Load ruels document and add to retriever
def load_using_webbaseloader(url) -> List[Document]:
    loader = WebBaseLoader(url)

    docs = loader.load()
    
    return docs
docs = load_using_webbaseloader("https://zakon.rada.gov.ua/laws/show/1306-2001-%D0%BF/print")
retriever.add_documents(docs)

# Combine prompt, chat model and retriever into QA chain
input_context_prompt = ChatPromptTemplate.from_messages([
    ('system', """Ти знавець правил дорожнього руху (ПДР) в Україні та розмовляєш виключно українською. Відповідай на питання беручи до уваги наступний контекст:\n\n{context}"""),
    ('user', """{input}"""),
    ]
)
chat_model = ChatOpenAI()
qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(chat_model, input_context_prompt))

app = fastapi.FastAPI(title="QA microservice")

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: str

add_routes(
    app,
    qa_chain.with_types(input_type=Input, output_type=Output),
    path="/pdr_qa",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
