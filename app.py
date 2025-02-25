from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import get_retriever
from generation import get_answer_chain
from query_processing import extract_medical_entities

# Initialize FastAPI application
app = FastAPI()

# Initialize the document retriever and answer generation chain
retriever = get_retriever()
answer_chain = get_answer_chain()

# Define the request model for query processing
class QueryRequest(BaseModel):
    text: str

@app.post("/process_query")
async def process_query(request: QueryRequest):
    """
    Process an incoming medical query:
    1. Extract medical entities from the query
    2. Retrieve relevant documents
    3. Generate an answer based on the documents
    """
    # Get the query text from the request
    query = request.text
    
    # Extract medical entities from the query
    entities = extract_medical_entities(query)
    
    # Retrieve relevant documents based on the query
    docs = retriever.invoke(query)
    
    # Generate an answer using the retrieved documents
    answer = answer_chain({"context": docs, "question": query})
    
    # Return the processed results
    return {
        "answer": answer,
        "entities": entities,
        "sources": [{"content": doc.page_content} for doc in docs]
    }