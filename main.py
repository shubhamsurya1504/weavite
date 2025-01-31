import weaviate
from weaviate.auth import AuthApiKey
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import embed_anything
from embed_anything import EmbeddingModel, WhichModel
from typing import Dict, List, Union
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Initialize Weaviate client
client = weaviate.Client(
    url="https://totxgehfsvinwfz33rkxew.c0.australia-southeast1.gcp.weaviate.cloud",
    auth_client_secret=AuthApiKey("8jGoywAvKvRLlpBYHgGs0bR4AE6A0Z5CklJB"),
)

# Initialize the Jina embedding model
jina_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)

class JinaEmbeddingWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return embed_anything.embed_query([text], self.model)[0].embedding

jina_embeddings = JinaEmbeddingWrapper(jina_model)

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_documents(request: QueryRequest) -> Dict[str, Union[str, List[dict]]]:
    try:
        query_vector = jina_embeddings.embed_query(request.query)
        
        result = (
            client.query.get(
                "Nishat",
                ["text"]
            )
            .with_hybrid(
                query=request.query,
                vector=query_vector,
                alpha=0.5,
                properties=["text"]
            )
            .with_additional(["score"])
            .with_limit(5)
            .do()
        )
        
        if not result["data"]["Get"]["Nishat"]:
            return {"message": "No relevant information found for your query"}
        
        # Fixed: Convert score to float before comparison
        filtered_results = [
            {
                "text": item["text"],
                "score": item.get("_additional", {}).get("score", 0)
            }
            for item in result["data"]["Get"]["Nishat"]
            if float(item.get("_additional", {}).get("score", 0)) > 0.7
        ]
        
        if not filtered_results:
            return {"message": "No relevant information found for your query"}
            
        # Sort results by score in descending order
        filtered_results.sort(key=lambda x: float(x["score"]), reverse=True)
        return {"results": filtered_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 