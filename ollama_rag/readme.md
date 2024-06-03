## 1. Start Ollama server in local.
curl -fsSL https://ollama.com/install.sh | sh
ollama run <model_name>   [phi3, mistral , etc]
## 2. Start ChomaDB using docker:
docker run -p 8000:8000 chromadb/chroma
## 3. Keep pdf file in data folder
## 4. change configs (models) in rag_query.py file if you want
## 4. build docker image:
    docker build -t rag .
## 5. Run docker container.
    docker run -it -p 5000:5000 -v <your_path>:/app/data rag


## 6. Test 
     curl -X POST -H "Content-Type: application/json" -d '{"query": "what is NLP"}' http://0.0.0.0:5000/query