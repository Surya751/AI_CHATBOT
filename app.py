import json
import os
import boto3
import pandas as pd
from io import StringIO
from fastapi import FastAPI, Body, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from passlib.context import CryptContext

load_dotenv()

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
mongo_uri = os.getenv('MONGO_URI')

app = FastAPI()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["user_db"]
users_collection = db["users"]

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# CORS Middleware
origins = [
    "http://localhost:3000",
    "http://192.168.5.55:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Bedrock Configuration
bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Models
class QueryModel(BaseModel):
    query: str

class UserRegisterModel(BaseModel):
    username: str
    password: str
    email: EmailStr

# New LoginRequestModel
class LoginRequestModel(BaseModel):
    username: str
    password: str

# Utility functions for authentication
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    return users_collection.find_one({"username": username})

# Register endpoint
@app.post("/register")
async def register(user: UserRegisterModel):
    if get_user(user.username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    user_data = {
        "username": user.username,
        "password": hash_password(user.password),
        "email": user.email
    }
    users_collection.insert_one(user_data)
    return {"message": "User registered successfully"}

# Login endpoint
@app.post("/login")
async def login(request: LoginRequestModel):
    user = get_user(request.username)
    if not user or not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return {"message": "Login successful"}

def data_ingestion():
    s3_client = boto3.client(
        's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
    )
    bucket_name = 'bedrock-sampledata'
    
    object_keys = [
        'data/Customer-support-sample-data.csv',
        'data/oil-gas-components-dataset.csv',
        'data/purchase-history-dataset.csv'
    ]
    
    all_documents = []
    
    for object_key in object_keys:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        df = pd.read_csv(StringIO(csv_content))
        text = df.to_string(index=False)
        
        documents = [Document(page_content=text)]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
        
        all_documents.extend(docs)
    
    return all_documents

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama3_llm():
    llm = BedrockLLM(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={
        "max_gen_len": 450
    })
    return llm

prompt_template = """
Human: Use the following context to provide an accurate answer. 
If the answer is related to numbers or price or quantity or calculations, make sure to provide exact figures. 
If the answer is unknown, say "I don't know" without trying to make up an answer.

{context}

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

@app.post("/ask-bot")
async def ask_question(query: str = Body(...)):
    print(f"Received query: {query}") 
    docs = data_ingestion()
    get_vector_store(docs)
    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_llama3_llm()
    response = get_response_llm(llm, faiss_index, query)
    return {"response": response}

@app.post("/update-vectors")
async def update_vectors():
    docs = data_ingestion()
    get_vector_store(docs)
    return {"message": "Vectors updated successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
