from transformers import pipeline

def extract_medical_entities(query):
    """
    Extract medical entities from the query text:
    1. Initialize biomedical NER model
    2. Process the query
    3. Format the results
    """
    # Initialize Named Entity Recognition pipeline with biomedical model
    ner_pipeline = pipeline(
        "ner", 
        model="d4data/biomedical-ner-all",
        tokenizer="d4data/biomedical-ner-all",
        aggregation_strategy="simple"  # Combine subwords into single entities
    )
    
    # Process the query and extract entities
    entities = ner_pipeline(query)
    
    # Format the results
    return [{"word": e["word"], "type": e["entity_group"]} for e in entities]