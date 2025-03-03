from transformers import pipeline
import re

def extract_medical_entities(query):
    """
    Extract medical entities from the query text:
    1. Initialize biomedical NER model
    2. Process the query with specialized entity recognition
    3. Format the results with improved type mapping
    """
    # Initialize Named Entity Recognition pipeline with biomedical model
    ner_pipeline = pipeline(
        "ner", 
        model="d4data/biomedical-ner-all",
        tokenizer="d4data/biomedical-ner-all",
        aggregation_strategy="simple",  # Combine subwords into single entities
        device=-1  # Use CPU (-1) or GPU (0)
    )
    
    # Process the query and extract biomedical entities
    entities = ner_pipeline(query)
    
    # Map entity types to more readable formats
    entity_type_map = {
        "DISEASE": "Disease",
        "CHEMICAL": "Chemical/Drug",
        "GENE": "Gene",
        "SPECIES": "Species",
        "DNA": "DNA",
        "CELL_LINE": "Cell Line",
        "CELL_TYPE": "Cell Type",
        "RNA": "RNA",
        "PROTEIN": "Protein"
    }
    
    # Format the results with improved type readability
    return [
        {
            "word": e["word"],
            "type": entity_type_map.get(e["entity_group"], e["entity_group"])
        } 
        for e in entities
    ]

def expand_query(query, entities):
    """
    Use detected entities to expand the query for better retrieval.
    This can be helpful when the original query is too short or lacks specificity.
    """
    # Base case: if no entities found or query is long enough, return original
    if not entities or len(query.split()) > 8:
        return query
        
    # Get disease and chemical entities which are most relevant for search
    relevant_entities = [
        e["word"] for e in entities 
        if e["type"] in ["Disease", "Chemical/Drug", "Gene", "Protein"]
    ]
    
    # If we have relevant entities, create an expanded query
    if relevant_entities:
        # Join the original query with specific entities for better retrieval
        expanded = f"{query} {' '.join(relevant_entities)}"
        return expanded
        
    return query