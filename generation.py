import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def format_docs(docs):
    """Format retrieved documents for prompt input"""
    return "\n\n".join([f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])

# Define the prompt template for medical question answering
prompt_template = """
You are a medical expert. Answer the question using ONLY the context below. Cite sources like [1], [2]. If unsure, say so.

Context:
{context}

Question: {question}

Answer:
"""

def get_answer_chain():
    """
    Create a function that generates medical answers using Gemini:
    1. Uses a medical expert prompt
    2. Formats context and question
    3. Generates response with controlled parameters
    """
    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_answer(input_data):
        # Format the prompt with context and question
        formatted_prompt = PromptTemplate.from_template(prompt_template).format(
            context=input_data["context"],
            question=input_data["question"]
        )
        
        # Generate response with controlled parameters
        response = model.generate_content(
            formatted_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,    # More deterministic outputs
                top_p=0.15       # More focused on likely tokens
            )
        )
        
        return response.text

    return generate_answer