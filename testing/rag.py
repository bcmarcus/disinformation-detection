import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import re

# Load the tokenizer and model for microsoft/Phi-3-mini-4k-instruct
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_length=1024
)

# Create the transformers pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Create a LangChain LLM wrapper
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to get Wikipedia content (modified from the original)
def get_wikipedia_content(title):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find(id="mw-content-text").find(class_="mw-parser-output")
        
        # Remove unwanted elements
        for unwanted in content(["table", "div", "script", "style"]):
            unwanted.decompose()
        
        # Extract text and clean it
        text = ' '.join(p.text for p in content.find_all('p'))
        text = re.sub(r'\[[\d\s]+\]', '', text)  # Remove citation numbers
        return text
    return ""

# Function to create vector store from Wikipedia content
def create_vector_store(titles):
    documents = []
    for title in titles:
        content = get_wikipedia_content(title)
        documents.append(content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents(documents)
    
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Create the vector store (example titles)
titles = ["Artificial Intelligence", "Machine Learning", "Natural Language Processing"]
vector_store = create_vector_store(titles)

# Create a retrieval QA chain
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Function to get answer
def get_answer(question):
    result = qa_chain({"query": question})
    return result["result"]

# Example usage
question = "What is the difference between AI and ML?"
answer = get_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")