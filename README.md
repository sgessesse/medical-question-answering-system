# Medical Question Answering System

A modular Retrieval Augmented Generation (RAG) system designed for medical question answering, built with a multi-agent architecture that separates query processing, retrieval, and generation components.

## Overview

This project implements a RAG system specifically tailored for medical domain questions, using a collection of medical textbooks from [Hugging Face's medical_textbook_en dataset](https://huggingface.co/datasets/Gaoj124/medical_textbook_en). The system is built with modularity in mind, making it easily adaptable to other domains such as legal or technical documentation.

### Key Features

- **Modular Architecture**: Separate agents for query processing, retrieval, and generation
- **Medical Entity Recognition**: Identifies medical terms and concepts in queries
- **Efficient Document Retrieval**: Uses FAISS for fast similarity search
- **Context-Aware Generation**: Generates answers based on retrieved medical literature
- **User-Friendly Interface**: Streamlit-based web interface

## Architecture

The system consists of several key components:

1. **Query Processing** (`query_processing.py`): Uses biomedical NER to extract medical entities
2. **Document Retrieval** (`retrieval.py`): FAISS-based retrieval system
3. **Answer Generation** (`generation.py`): Uses Google's Gemini model for answer generation
4. **Data Ingestion** (`data_ingestion.py`): Processes and indexes medical documents
5. **API Server** (`app.py`): FastAPI-based backend server
6. **Web Interface** (`streamlit_app.py`): Streamlit-based user interface

## Setup

1. Clone the repository: 
```bash
git clone https://github.com/sgessesse/medical-question-answering-system.git
cd medical-question-answering-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```bash
GOOGLE_API_KEY=your_api_key_here
```

4. Create the document index:
```bash
python data_ingestion.py
```
Note: This step is required before running the application as the FAISS index is not included in the repository due to size constraints.

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

2. In a new terminal, start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Enter your medical question in the text input field
2. Click "Submit Query" or press Enter
3. View the generated answer, detected medical entities, and source documents

## Adapting to Other Domains

The modular nature of this system makes it adaptable to other domains. To adapt:

1. Replace the document dataset in `data_ingestion.py`
2. Update the NER model in `query_processing.py`
3. Adjust the prompt template in `generation.py`
4. Modify the UI labels in `streamlit_app.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical textbook dataset from [Hugging Face](https://huggingface.co/datasets/Gaoj124/medical_textbook_en)
- Biomedical NER model from d4data
- Google's Gemini model for text generation