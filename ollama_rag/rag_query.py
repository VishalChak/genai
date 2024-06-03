import glob, os
import uuid
import chromadb
from ollama import Client
from flask import Flask, request
from chromadb.config import Settings

import langchain_community
import langchain_text_splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

model = "phi3"
embedding_model = "all-MiniLM-L6-v2"
server = "host.docker.internal"
collection_name = "my_collection"

embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
chroma_client = chromadb.HttpClient(host=server, port=8000)
collection = chroma_client.get_or_create_collection(name=collection_name)
def load_documents(data_fol):
    for file_name in glob.glob(os.path.join(data_fol, "*.pdf")):
        loader = PyPDFLoader(file_name)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        for doc in docs:
            collection.add(ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content)


def extract_context(query):
    db = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_function, )
    docs = db.similarity_search(query)
    fullcontent =''
    for doc in docs:
        fullcontent ='. '.join([fullcontent,doc.page_content])

    return fullcontent
    
def get_system_message_rag(content):
        return f"""You are an expert consultant helping executive advisors to get relevant information from internal documents.

        Generate your response by following the steps below:
        1. Recursively break down the question into smaller questions.
        2. For each question/directive:
            2a. Select the most relevant information from the context in light of the conversation history.
        3. Generate a draft response using selected information.
        4. Remove duplicate content from draft response.
        5. Generate your final response after adjusting it to increase accuracy and relevance.
        6. Do not try to summarise the answers, explain it properly.
        6. Only show your final response! 
        
        Constraints:
        1. DO NOT PROVIDE ANY EXPLANATION OR DETAILS OR MENTION THAT YOU WERE GIVEN CONTEXT.
        2. Don't mention that you are not able to find the answer in the provided context.
        3. Don't make up the answers by yourself.
        4. Try your best to provide answer from the given context.

        CONTENT:
        {content}
        """

def get_ques_response_prompt(question):
    return f"""
    ==============================================================
    Based on the above context, please provide the answer to the following question:
    {question}
    """

def generate_rag_response(content,question):
    client = Client(host=server)
    stream = client.chat(model=model, messages=[
    {"role": "system", "content": get_system_message_rag(content)},            
    {"role": "user", "content": get_ques_response_prompt(question)}
    ],stream=True)
    # print(get_system_message_rag(content))
    # print(get_ques_response_prompt(question))
    # print("####### THINKING OF ANSWER............ ")
    full_answer = ''
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        full_answer =''.join([full_answer,chunk['message']['content']])

    return full_answer

app = Flask(__name__) 
@app.route('/query', methods=['POST'])
def respond_to_query():
    if request.method == 'POST':
        data = request.get_json()
        # Assuming the query is sent as a JSON object with a key named 'query'
        query = data.get('query')
        content = extract_context(query)
        # print(content)
        response = f'This is the response to your query from local RAG:\n {generate_rag_response(content , query)}'
        return response
        # return "Hello"

if __name__ == '__main__':
    load_documents("data")
    app.run(debug=True, host='0.0.0.0')