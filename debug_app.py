from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import sys

# Define the request model
class QueryRequest(BaseModel):
    text: str

# Initialize FastAPI application
app = FastAPI(title="Debug API")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch and log all errors"""
    print(f"Exception occurred: {str(exc)}")
    print(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )

@app.post("/test")
async def test_endpoint(request: QueryRequest):
    """Simple test endpoint that just echoes the request"""
    return {"message": "Test successful", "received_text": request.text}

@app.get("/")
async def root():
    return {"message": "Debug API is running"} 