import os
import base64
import tempfile
from flask import Flask, render_template, jsonify, request, send_file
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import *
import google.genai as genai
from gtts import gTTS
from google.genai import types
import logging

# ── Flask App & Environment ─────────────────────────────
app = Flask(__name__)
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Rate Limiting (FIXED for latest flask-limiter) ───────
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Correct initialization for newer versions of flask-limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30 per hour"]
)
limiter.init_app(app)   # ← This is the important part

# ── API Keys ─────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY     = os.environ.get('GROQ_API_KEY')
GEMINI_API_KEY   = os.environ.get('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"]     = GROQ_API_KEY

# ── Gemini client (voice + vision) ──────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── RAG pipeline (text Q&A) ─────────────────────────────
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medibot",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)

# ── Vision system prompt ────────────────────────────────
VISION_SYSTEM_PROMPT = """You are a professional medical assistant helping analyze medical images.
What do you observe in this image medically? If you identify any conditions, suggest possible 
diagnoses and general remedies. Do not use special characters or markdown formatting.
Respond in one clear paragraph as if speaking directly to a patient.
Start with 'Based on what I can see...' Keep your answer concise (2-3 sentences)."""

# ── Routes ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
@limiter.limit("30 per hour")
def chat():
    """Handle plain-text messages via RAG pipeline."""
    try:
        user_msg = request.form.get("msg", "").strip()
        if not user_msg:
            return "No message received.", 400
        logging.info(f"[TEXT] {user_msg}")
        answer = rag_chain.invoke(user_msg)
        return str(answer)
    except Exception as e:
        logging.exception("Error in /get")
        return f"Error: {str(e)}", 500


@app.route("/transcribe", methods=["POST"])
@limiter.limit("30 per hour")
def transcribe():
    """Transcribe uploaded audio using Gemini STT."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file"}), 400

        audio_file = request.files['audio']
        suffix = os.path.splitext(audio_file.filename)[-1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            audio_data = base64.standard_b64encode(f.read()).decode("utf-8")
        os.unlink(tmp_path)

        mime = "audio/webm"
        if suffix == ".mp3":  mime = "audio/mpeg"
        elif suffix == ".wav": mime = "audio/wav"
        elif suffix == ".ogg": mime = "audio/ogg"

        result = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this audio word-for-word in English. Only provide the transcription, nothing else:",
                types.Part.from_bytes(data=base64.b64decode(audio_data), mime_type=mime)
            ]
        )
        text = result.text.strip()
        logging.info(f"[STT] {text}")
        return jsonify({"text": text})

    except Exception as e:
        logging.exception("Error in /transcribe")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze_image", methods=["POST"])
@limiter.limit("10 per hour")
def analyze_image():
    """Analyze an uploaded medical image using Gemini Vision."""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        image_b64   = base64.b64encode(image_bytes).decode("utf-8")

        patient_query = request.form.get("query", "").strip()
        full_prompt   = VISION_SYSTEM_PROMPT
        if patient_query:
            full_prompt += f"\n\nPatient describes: {patient_query}"

        result = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                full_prompt,
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg")
            ]
        )
        answer = result.text.strip()
        logging.info(f"[VISION] {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        logging.exception("Error in /analyze_image")
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
@limiter.limit("30 per hour")
def speak():
    """Convert text to speech using gTTS and return the MP3."""
    try:
        data = request.get_json()
        text = (data or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name

        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(tmp_path)

        return send_file(tmp_path, mimetype="audio/mpeg", as_attachment=False,
                         download_name="response.mp3")

    except Exception as e:
        logging.exception("Error in /speak")
        return jsonify({"error": str(e)}), 500


# ── Run Server ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)