import os
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
import re


from google.generativeai.types import HarmCategory, HarmBlockThreshold

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


llm = genai.GenerativeModel("gemini-2.0-flash")


prompt_template = """
Generate {count} interview questions with clear category labels based on:
Skills: {skills}
Experience: {experience}
Resume Text: {resume_text}

IMPORTANT: Start each question directly with one of these exact labels:
"[TECHNICAL] ", "[SITUATIONAL] ", or "[BEHAVIORAL] "

Do not include any introductory text or numbering. 
Simply list the questions one per line.

Example format:
[TECHNICAL] How would you optimize a MongoDB query for...
[BEHAVIORAL] Describe a time when you had to resolve a team conflict...
[SITUATIONAL] What would you do if you discovered a critical bug in production...

Generate a mix of approximately:
- 50% Technical questions
- 30% Behavioral questions
- 20% Situational questions

Focus on questions relevant to the candidate's background.
"""

def generate_interview_questions(details, count=10):
    prompt = PromptTemplate.from_template(prompt_template)
    prompt_text = prompt.format(
        skills=", ".join(details["skills"]),
        experience=", ".join(details["experience"]),
        resume_text=details["text"][:3000],
        count=count
    )
    response = llm.generate_content(prompt_text)
    return parse_labeled_questions(response.text)

def parse_labeled_questions(response_text):
    questions = []
    categories = {
        "technical": [],
        "situational": [],
        "behavioral": []
    }
    
    
    pattern = r"\[(TECHNICAL|SITUATIONAL|BEHAVIORAL)\]\s*(.+?)(?=\n\[|\Z)"
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for category, question in matches:
        category = category.lower()
        question = question.strip()  
        if question:
            categories[category].append(question)
            questions.append((category, question))
    

    if not questions:
        for line in response_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            
            if line.lower().startswith("[technical]"):
                question = line[len("[technical]"):].strip()
                categories["technical"].append(question)
                questions.append(("technical", question))
            elif line.lower().startswith("[situational]"):
                question = line[len("[situational]"):].strip()
                categories["situational"].append(question)
                questions.append(("situational", question))
            elif line.lower().startswith("[behavioral]"):
                question = line[len("[behavioral]"):].strip()
                categories["behavioral"].append(question)
                questions.append(("behavioral", question))
    
    return {
        "all_questions": questions,  
        "categorized": categories    
    }


if __name__ == "__main__":
    sample_details = {
        "skills": ["Python", "Machine Learning", "MongoDB"],
        "experience": ["Data Scientist at XYZ", "ML Engineer at ABC"],
        "text": "Experienced data scientist with 5 years in ML projects..."
    }
    
    result = generate_interview_questions(sample_details)
    
    print("All Questions with Labels:")
    for category, question in result["all_questions"]:
        print(f"[{category.upper()}] {question}")
    
    print("\nCategorized Questions:")
    print("Technical:", result["categorized"]["technical"])
    print("Situational:", result["categorized"]["situational"])
    print("Behavioral:", result["categorized"]["behavioral"])
