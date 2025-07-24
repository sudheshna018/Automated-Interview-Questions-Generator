from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
from werkzeug.utils import secure_filename
from utils.resume_parser import extract_text, extract_key_details
from utils.question_generator import generate_interview_questions
import google.generativeai as genai
import tempfile
import base64
from pydub import AudioSegment
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import json
import re
from functools import wraps
import uuid
from datetime import datetime
import logging
import traceback
from bson.objectid import ObjectId

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
app.secret_key = 'your_secret_key_here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HF_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HF_API_TOKEN = os.getenv('HF_API_TOKEN', 'your_hf_token')
MONGO_URI = os.getenv('MONGO_URI', 'your_mongo_uri')

mongo_client = MongoClient(MONGO_URI)
db = mongo_client['interview_app']
users_collection = db['users']
interview_sessions_collection = db['interview_sessions']

genai.configure(api_key='YOUR_GOOGLE_API_KEY')
evaluation_model = genai.GenerativeModel('gemini-2.0-flash')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    return text.strip()

def transcribe_with_huggingface(audio_file_path):
    try:
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "audio/wav"
        }
        response = requests.post(HF_API_URL, headers=headers, data=audio_data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "No transcription available")
        elif response.status_code == 503:
            return "Error: Model is loading, please try again later."
        elif response.status_code == 400:
            return "Error: Invalid audio format or data."
        elif response.status_code in (401, 403):
            return "Error: Invalid API token or access denied."
        elif response.status_code == 422:
            return "Error: Unprocessable entity, check audio file."
        else:
            return f"Error: Transcription failed with status {response.status_code}"
    except Exception as e:
        return f"Error: Transcription failed - {str(e)}"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/app', methods=['GET', 'POST'])
@login_required
def index():
    questions = None
    scroll_to_questions = False
    if request.method == 'POST':
        if 'resume' not in request.files:
            return redirect(request.url)
        file = request.files['resume']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text = extract_text(filepath)
            details = extract_key_details(text)
            num_questions = request.form.get('num_questions', 10)
            try:
                num_questions = int(num_questions)
            except Exception:
                num_questions = 10
            result = generate_interview_questions(details, count=num_questions)
            scroll_to_questions = True
            return render_template('index.html', categorized=result["categorized"], scroll_to_questions=scroll_to_questions)
    return render_template('index.html', questions=questions, scroll_to_questions=scroll_to_questions)

@app.route('/mock-interview', methods=['GET', 'POST'])
@login_required
def mock_interview():
    questions = None
    session_id = None
    if request.method == 'POST':
        if 'resume' not in request.files:
            return redirect(request.url)
        file = request.files['resume']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text = extract_text(filepath)
            details = extract_key_details(text)
            questions = generate_interview_questions(details, count=10)
            session_id = str(uuid.uuid4())
            question_list = [
                {"index": i+1, "category": cat.lower(), "question": q}
                for i, (cat, q) in enumerate(questions.get("all_questions", []))
            ]
            interview_sessions_collection.insert_one({
                "session_id": session_id,
                "user_id": session['user_id'],
                "timestamp": datetime.utcnow().isoformat(),
                "resume_filename": filename,
                "job_title": None,
                "seniority": None,
                "questions": question_list,
                "answers": [],
                "feedback": [],
                "summary": {"total_score": 0, "completion_status": "partial"}
            })
            session['current_session_id'] = session_id
    return render_template('mock_interview.html', questions=questions, session_id=session_id)

@app.route('/transcribe-audio', methods=['POST'])
@login_required
def transcribe_audio():
    try:
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        temp_audio_path = None
        wav_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_path = temp_audio_file.name
        audio = AudioSegment.from_file(temp_audio_path)
        wav_path = temp_audio_path + ".wav"
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        transcript_text = transcribe_with_huggingface(wav_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        if 'current_session_id' in session:
            question_index = request.json.get('question_index')
            if question_index:
                interview_sessions_collection.update_one(
                    {"session_id": session['current_session_id']},
                    {"$push": {
                        "answers": {
                            "question_index": int(question_index),
                            "answer_text": clean_text(transcript_text),
                            "is_audio": True
                        }
                    }}
                )
        return jsonify({"transcript": transcript_text})
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route('/evaluate-interview', methods=['POST'])
@login_required
def evaluate_interview():
    if 'current_session_id' not in session:
        logging.error("Current session ID not found in session")
        return jsonify({"error": "Session not found"}), 400
    
    data = request.get_json()
    all_questions = data.get('all_questions', [])
    answers = data.get('answers', {})
    results = []
    total_score = 0
    evaluated_count = 0
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    for i, question in enumerate(all_questions):
        answer = answers.get(f'answer{i+1}', '').strip()
        question_index = i + 1
        if answer:
            clean_answer = clean_text(answer)
            prompt = f"""Evaluate this interview answer on a scale of 1-10:
Question: {question}
Answer: {clean_answer}
Evaluate based on:
- Technical accuracy (40%)
- Clarity of communication (30%)
- Relevance and completeness (30%)
Provide your response in VALID JSON format ONLY:
{{
    "score": <number>,
    "feedback": "<feedback text>"
}}
"""
            try:
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                print(f"Raw response for question {question_index}: {response_text}")
                response_text = response_text.replace('```json', '').replace('```', '')
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    evaluation = json.loads(response_text[json_start:json_end])
                    score = min(max(float(evaluation.get('score', 0)), 0), 10)
                    feedback = evaluation.get('feedback', 'No feedback provided')
                else:
                    score_match = re.search(r'score["\s:]+(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
                    score = float(score_match.group(1)) if score_match else 5.0
                    feedback = 'Evaluation failed.'
            except Exception as e:
                logging.error(f"Error evaluating answer for question {question_index}: {str(e)}")
                score = 5.0
                feedback = "Evaluation error occurred."
            results.append({
                "question": question,
                "answer": answer,
                "score": score,
                "feedback": feedback
            })
            total_score += score
            evaluated_count += 1
            interview_sessions_collection.update_one(
                {"session_id": session['current_session_id']},
                {"$push": {
                    "answers": {
                        "question_index": question_index,
                        "answer_text": clean_answer,
                        "is_audio": False
                    },
                    "feedback": {
                        "question_index": question_index,
                        "score": score,
                        "feedback_text": feedback
                    }
                }}
            )
            print(f"Updated session {session['current_session_id']} with answer and feedback for question {question_index}")
        else:
            results.append({
                "question": question,
                "answer": "",
                "score": 0,
                "feedback": "Question not attempted"
            })
            interview_sessions_collection.update_one(
                {"session_id": session['current_session_id']},
                {"$push": {
                    "answers": {
                        "question_index": question_index,
                        "answer_text": "",
                        "is_audio": False
                    },
                    "feedback": {
                        "question_index": question_index,
                        "score": 0,
                        "feedback_text": "Question not attempted"
                    }
                }}
            )
            print(f"Updated session {session['current_session_id']} with not attempted for question {question_index}")
    
    try:
        avg_score = round((total_score / evaluated_count) * 10, 1) if evaluated_count > 0 else 0
    except ZeroDivisionError:
        avg_score = 0
    interview_sessions_collection.update_one(
        {"session_id": session['current_session_id']},
        {"$set": {
            "summary": {
                "total_score": avg_score,
                "completion_status": "completed" if evaluated_count > 0 else "partial"
            }
        }}
    )
    print(f"Updated summary for session {session['current_session_id']}: total_score={avg_score}, status={'completed' if evaluated_count > 0 else 'partial'}")
    
    return jsonify({
        "total_score": avg_score,
        "details": results
    })

@app.route('/history', methods=['GET'])
@login_required
def history():
    user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
    sessions = interview_sessions_collection.find({"user_id": session['user_id']}).sort("timestamp", -1)
    return render_template('history.html', sessions=sessions, username=user.get('username'), email=user.get('email'))

def clean_text(text):
    """Clean text of control characters and other problematic elements"""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F700-\U0001F77F"  
                               u"\U0001F780-\U0001F7FF"  
                               u"\U0001F800-\U0001F8FF"  
                               u"\U0001F900-\U0001F9FF"  
                               u"\U0001FA00-\U0001FA6F"  
                               u"\U0001FA70-\U0001FAFF"  
                               u"\U00002702-\U000027B0"  
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...',
        '\n': ' ',  
        '\r': '',
        '\t': ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

@app.route('/generate-job-questions', methods=['POST'])
@login_required
def generate_job_questions():
    try:
        job_title = request.form.get('jobTitle')
        seniority = request.form.get('seniority')
        job_description = request.form.get('jobDescription')
        num_questions = request.form.get('num_questions', 15)
        try:
            num_questions = int(num_questions)
        except Exception:
            num_questions = 15

        if not job_title or not seniority or not job_description:
            return render_template('index.html', 
                                error='Please fill in all fields (Job Title, Seniority, and Job Description)')

        prompt = f"""
You are an API. Respond ONLY with a valid JSON object and nothing else.

Generate exactly {num_questions} interview questions for a {seniority} {job_title} position.

Job Description:
{job_description}

Divide the questions as follows (total must be {num_questions}):
- Technical: ~50%
- Behavioral: ~30%
- Situational: ~20%

Format your response as a JSON object with these exact keys:
{{
    "technical": [list of technical questions],
    "behavioral": [list of behavioral questions],
    "situational": [list of situational questions]
}}

Do NOT include any explanations, code blocks, or extra text. Only output the JSON object.
"""

        try:
            response = evaluation_model.generate_content(prompt)
            response_text = response.text.strip()
            print("Gemini raw response:", response_text)  

            
            import re
            response_text = re.sub(r"^```json|```$", "", response_text, flags=re.MULTILINE).strip()

            
            match = re.search(r'({.*})', response_text, re.DOTALL)
            if match:
                try:
                    questions_json = json.loads(match.group(1))
                except Exception as e:
                    print("JSON decode error:", e)
                    return render_template('index.html', error='Failed to parse generated questions. Please try again. (Raw response: ' + response_text[:500] + ')')
            else:
                print("Failed to parse JSON from Gemini response:", response_text)
                return render_template('index.html', error='Failed to parse generated questions. Please try again. (Raw response: ' + response_text[:500] + ')')

            
            return render_template('index.html', 
                                categorized=questions_json,
                                job_title=job_title,
                                seniority=seniority,
                                scroll_to_questions=True)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return render_template('index.html', 
                                error='Failed to parse generated questions. Please try again.')
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return render_template('index.html', 
                                error=f'Failed to generate questions: {str(e)}')

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return render_template('index.html', 
                             error='An unexpected error occurred. Please try again.')
    
@app.route('/generate-answer', methods=['POST'])
@login_required
def generate_answer():
    try:
        logging.debug("Received request for answer generation")
        data = request.get_json()
        question = data.get('question')
        category = data.get('category')
        logging.debug(f"Question: {question}, Category: {category}")

        if not question or not category:
            logging.error("Missing question or category")
            return jsonify({"error": "Question and category are required"}), 400

        
        try:
            logging.debug("Initializing Gemini model")
            model = genai.GenerativeModel('gemini-2.0-flash')
            logging.debug("Model initialized successfully")
            
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            logging.debug("Configuration settings prepared")

        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Answer generation service unavailable: {str(e)}"}), 500

        
        if category == 'technical':
            prompt = f"""Generate a concise technical answer for this interview question. Keep it focused and brief while covering the essential points.
Question: {question}

Provide a clear, focused response that:
- Demonstrates key technical concepts
- Uses 2-3 sentences for explanation
- Includes one brief example if relevant
- Stays under 150 words"""

        elif category == 'behavioral':
            prompt = f"""Generate a brief STAR method answer for this behavioral interview question. Keep it focused and concise.
Question: {question}

Structure a brief response using STAR:
- Situation: One sentence context
- Task: One sentence goal
- Action: 1-2 sentences on what you did
- Result: One sentence outcome
Keep the entire response under 150 words."""

        else:  # situational
            prompt = f"""Generate a concise strategic answer for this situational interview question. Focus on key points only.
Question: {question}

Provide a brief response that:
- States your approach in 1-2 sentences
- Lists 2-3 key steps you would take
- Mentions one key consideration
- Concludes with expected outcome
Keep the entire response under 150 words."""

        logging.debug(f"Generated prompt for category: {category}")

        try:
            logging.debug("Attempting to generate content")
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            logging.debug("Content generation completed")
            
            if response and response.text:
                answer = response.text.strip()
                logging.debug("Successfully generated answer")
                return jsonify({"answer": answer})
            else:
                logging.error("Empty response from model")
                return jsonify({"error": "Failed to generate answer - Empty response"}), 500

        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('index'))  
    error = None
    if request.method == 'POST':
        login_identifier = request.form.get('username')
        password = request.form.get('password')
        user = users_collection.find_one({'$or': [{'username': login_identifier}, {'email': login_identifier}]})
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']        
            session['user_id'] = str(user['_id'])     
            return redirect(url_for('index'))
        else:
            error = 'Invalid username, email, or password.'
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user' in session:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username').strip()
        email = request.form.get('email').strip()
        confirm_email = request.form.get('confirm_email').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not confirm_email or not password or not confirm_password:
            error = 'All fields are required.'
        elif email != confirm_email:
            error = 'Email addresses do not match.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
            error = 'Username or email already exists.'
        else:
            hashed_password = generate_password_hash(password)
            users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})
            return redirect(url_for('login')) 

    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    return redirect(url_for('landing'))


@app.before_request
def require_login():
    if request.endpoint in ['login', 'signup', 'static', 'landing', None]:
        return  
    if 'user_id' not in session:
        return redirect(url_for('login'))

@app.route('/test')
def test():
    return "Test route works!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
