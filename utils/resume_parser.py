import PyPDF2
from docx import Document
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages])
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")
    return text

def extract_key_details(text):
    doc = nlp(text)
    skills = []
    experiences = []
    
   
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:  
            skills.append(ent.text)
    
    
    experiences = re.findall(r'\b(\d+\+? years? of .*? experience)\b', text, re.IGNORECASE)
    
    return {
        "skills": list(set(skills)),
        "experience": experiences,
        "text": text
    }
