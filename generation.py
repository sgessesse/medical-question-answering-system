import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from retrieval import get_source_info

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def format_docs(docs):
    """
    Format retrieved documents for prompt input with enhanced metadata for better citations
    Returns both formatted context and source information
    """
    formatted_docs = []
    source_info = []
    
    for i, doc in enumerate(docs):
        # Get source information
        source_data = get_source_info(doc)
        source_name = source_data["source_name"]
        chunk_id = source_data["chunk_id"]
        
        # Extract section and page information if available
        section = doc.metadata.get("section", "")
        page = doc.metadata.get("page", "")
        location_info = ""
        
        if section:
            location_info += f", Section: {section}"
        if page:
            location_info += f", Page: {page}"
            
        # Format document with citation info
        formatted_docs.append(f"[Document {i+1}]\n{doc.page_content}")
        
        # Store source information for citation
        source_info.append({
            "id": i+1,
            "source": source_name,
            "location": location_info.strip(", ")
        })
    
    return "\n\n".join(formatted_docs), source_info

# Define the improved prompt template for medical question answering
prompt_template = """
You are an expert medical assistant providing accurate and helpful information. Answer the user's question based ONLY on the provided context.

Guidelines:
1. Be thorough in your explanation when medical concepts need clarification.
2. Use simple language while maintaining medical accuracy.
3. Cite your sources properly using the format [Source #].
4. If information is missing or unclear, acknowledge this explicitly.
5. Organize complex answers with clear structure.
6. NEVER make up information not provided in the context.

Context:
{context}

Question: {question}

When citing, use these sources:
{sources}

Answer:
"""

def get_answer_chain():
    """
    Create a function that generates medical answers using Gemini:
    1. Uses an improved medical expert prompt
    2. Formats context and question with better structure
    3. Includes detailed source information for citations
    4. Generates response with carefully tuned parameters
    """
    # Initialize Gemini model with improved model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_answer(input_data):
        # Get question from input
        question = input_data["question"]
        
        # Format documents and extract source information
        context_docs = input_data["context"]
        formatted_context, source_info = format_docs(context_docs)
        
        # Format source information for citations
        sources_text = "\n".join([
            f"[Source {s['id']}]: {s['source']}{' - ' + s['location'] if s['location'] else ''}"
            for s in source_info
        ])
        
        # Format the prompt with context, question and source info
        formatted_prompt = PromptTemplate.from_template(prompt_template).format(
            context=formatted_context,
            question=question,
            sources=sources_text
        )
        
        # Generate response with carefully tuned parameters
        response = model.generate_content(
            formatted_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,    # Slightly increased for more natural explanations
                top_p=0.2,         # Slightly increased for more varied responses
                top_k=40,          # Consider more tokens for better explanations
                max_output_tokens=800  # Allow for more detailed responses
            )
        )
        
        return response.text

    return generate_answer