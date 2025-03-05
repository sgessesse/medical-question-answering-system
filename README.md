# Medical Question Answering System

A specialized AI-powered system for answering medical questions using a combination of retrieval-augmented generation, medical entity extraction, and state-of-the-art language models.

![Medical QA System](https://img.shields.io/badge/Medical%20QA-System-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

## Features

- **Advanced Document Retrieval**: Uses semantic search with FAISS vector database to find relevant medical information
- **Medical Entity Recognition**: Extracts medical terms from queries using specialized biomedical NER
- **Query Expansion**: Improves search by expanding queries with recognized medical entities
- **Evidence-Based Answers**: Generates answers with direct citations to medical sources
- **User-Friendly Interface**: Clean Streamlit UI with organized display of results and sources
- **Production-Ready API**: FastAPI backend with proper error handling and documentation

## Architecture

The system integrates multiple components:

1. **Query Processing**: Analyzes and expands medical queries
2. **Document Retrieval**: Finds relevant sections from medical textbooks
3. **Answer Generation**: Creates comprehensive, evidence-based answers
4. **User Interface**: Presents information in an accessible way

![Architecture Diagram](architecture_diagram.png)

## Getting Started

### Prerequisites

- Python 3.10+
- FastAPI
- Streamlit
- LangChain
- Hugging Face Transformers
- FAISS vector database
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sgessesse/medical-question-answering-system.git
cd medical-question-answering-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
GOOGLE_API_KEY=your_gemini_api_key
```

4. Download the medical textbooks dataset:
The system uses a medical textbooks dataset from Hugging Face. You need to download it and place it in a `sample_docs` directory:
```bash
# Create a sample_docs directory
mkdir -p sample_docs

# Download the dataset from Hugging Face
# Visit: https://huggingface.co/datasets/Gaoj124/medical_textbook_en
# Follow the instructions to download the textbooks and save them in the sample_docs folder
```

5. Create the FAISS index:
Once you have downloaded the textbooks, you need to create the FAISS index:
```bash
# Run the data ingestion script to create the FAISS index
python data_ingestion.py
```
This step is required before running the application as the FAISS index is not included in the repository.

6. Run the system:
```bash
# Start the FastAPI backend
python -m uvicorn app:app --reload --port 8000

# In a separate terminal, start the Streamlit UI
python -m streamlit run streamlit_app.py
```

7. Access the application:
   - API: http://localhost:8000
   - UI: http://localhost:8501

## Usage

1. Enter a medical question in the text area
2. Click "Submit Query"
3. View the detailed answer with source citations
4. Expand source sections for more information

### Example Questions

- "What are the symptoms of diabetes?"
- "How is rheumatoid arthritis diagnosed?"
- "What are the side effects of statins?"
- "What treatments are available for depression?"

## Project Structure

```
medical-question-answering-system/
├── app.py                 # FastAPI backend application
├── streamlit_app.py       # Streamlit frontend interface
├── data_ingestion.py      # Document loading and indexing
├── retrieval.py           # Vector search and document retrieval
├── generation.py          # Answer generation with Gemini
├── query_processing.py    # Medical entity extraction and query expansion
├── faiss_index/           # Vector database (not in repo, created on setup)
├── sample_docs/           # Medical textbook resources (not in repo)
├── .env                   # Environment variables (not in repo)
└── requirements.txt       # Project dependencies
```

## Data Sources

The system uses medical textbooks from the [Hugging Face medical_textbook_en dataset](https://huggingface.co/datasets/Gaoj124/medical_textbook_en), including:
- Harrison's Internal Medicine
- Gray's Anatomy
- DSM-5 Psychiatry
- Robbins Pathology
- And many other medical references

You must download these textbooks from the provided link and place them in the `sample_docs` directory to recreate the FAISS index for the retrieval system to work.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The [LangChain](https://github.com/hwchase17/langchain) team for the retrieval framework
- [Hugging Face](https://huggingface.co/) for NLP models and embedding support
- [The Biomedical NER](https://huggingface.co/d4data/biomedical-ner-all) model from d4data
- [The medical_textbook_en dataset](https://huggingface.co/datasets/Gaoj124/medical_textbook_en) used for the knowledge base