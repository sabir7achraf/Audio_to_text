from flask import Flask, request, jsonify, render_template ,send_from_directory
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_cors import CORS
import os
import json
import numpy as np
from audio_processor import ArabicAudioProcessor
from evaluator import ArabicReadingEvaluator
from AzurePronunciationCorrector import AzurePronunciationCorrector, PronunciationError  # NEW: Import AzurePronunciationCorrector

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'ogg', 'mp3', 'm4a'}
AUDIO_CORRECTIONS_FOLDER = 'audio_corrections'  # NEW: Folder for pronunciation correction audio files

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_CORRECTIONS_FOLDER'] = AUDIO_CORRECTIONS_FOLDER
CORS(app, origins=["http://localhost:3000"])
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost:3306/agentiai'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create upload and audio corrections folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_CORRECTIONS_FOLDER, exist_ok=True)  # NEW: Create audio corrections folder

db = SQLAlchemy(app)
processor = ArabicAudioProcessor()

# NEW: Initialize AzurePronunciationCorrector
try:
    AZURE_SPEECH_KEY = "F2d84URhnc9EhA8UG2QCbAAEWzGIeVnbgmQpyWC21OlXytPzbPxYJQQJ99BEACYeBjFXJ3w3AAAYACOGbTan"
    # AZURE_SPEECH_KEY =os.getenv('AZURE_SPEECH_KEY', 'your-azure-speech-key-here')  # Replace with actual key or use env var
    AZURE_REGION = os.getenv('AZURE_REGION', 'eastus')  # Replace with actual region or use env var
    pronunciation_corrector = AzurePronunciationCorrector(
        subscription_key=AZURE_SPEECH_KEY,
        region=AZURE_REGION,
        language='ar-SA'
    )
    print("✅ Azure Pronunciation Corrector initialized successfully")
except Exception as e:
    pronunciation_corrector = None
    print(f"❌ Failed to initialize Azure Pronunciation Corrector: {e}")

try:
    gemini_api_key = "AIzaSyBDk0RlHr-rHMqePNcEWKz1C9cz7cHgiDk"
    # gemini_api_key = os.getenv('Api_gemini')  # Use env var for security
    if gemini_api_key:
        reading_evaluator = ArabicReadingEvaluator(api_key=gemini_api_key)
        print("✅ Reading evaluator initialized successfully")
    else:
        reading_evaluator = None
        print("⚠️ GEMINI_API_KEY not found. Reading evaluation will not be available.")
except Exception as e:
    reading_evaluator = None
    print(f"❌ Failed to initialize reading evaluator: {e}")

# Updated Recorder model to store pronunciation correction results
class Recorder(db.Model):
    __tablename__ = 'recorder'
    id = db.Column(db.Integer, primary_key=True)
    id_eleve = db.Column(db.Integer, nullable=False)
    idTexte = db.Column(db.Integer, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    pronunciation_corrections = db.Column(db.Text, nullable=True)  # NEW: Store JSON of pronunciation corrections
    date_enregistrement = db.Column(db.DateTime, default=datetime.utcnow)

class Texte(db.Model):
    __tablename__ = 'texte'
    idTexte = db.Column(db.Integer, primary_key=True)
    texteContent = db.Column(db.Text, nullable=False)

with app.app_context():
    db.create_all()

    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/upload", methods=["POST"])
def upload_and_transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            id_eleve = int(request.form.get("id_eleve", 1))
            idTexte = int(request.form.get("idTexte", 1))
            
            # Retrieve original text from Texte table
            texte = Texte.query.filter_by(idTexte=idTexte).first()
            if not texte:
                return jsonify({"error": f"Texte avec idTexte={idTexte} non trouvé"}), 404
            
            # Create database record
            record = Recorder(id_eleve=id_eleve, idTexte=idTexte, file_path=filepath)
            db.session.add(record)
            db.session.commit()
            
            # Process audio with quality analysis
            result = processor.process_audio(record.file_path)
            
            # Debug: Print result type and content
            print(f"DEBUG - Type de result: {type(result)}")
            print(f"DEBUG - Contenu de result: {result}")
            
            response_data = {
                "success": False,
                "record_id": record.id,
                "message": "",
                "transcription": "",
                "quality_analysis": {},
                "pronunciation_corrections": {}  # NEW: Add pronunciation corrections
            }
            
            if isinstance(result, str):
                # Old version: just transcription
                record.transcription = result
                response_data["transcription"] = result
                response_data["success"] = True
                response_data["message"] = "Fichier enregistré et transcrit avec succès"
                
                # NEW: Apply pronunciation correction if available
                if pronunciation_corrector and texte:
                    corrections = pronunciation_corrector.correct_pronunciation(
                        original_text=texte.texteContent,
                        transcribed_text=result,
                        audio_output_dir=AUDIO_CORRECTIONS_FOLDER
                    )
                    record.pronunciation_corrections = json.dumps(corrections, ensure_ascii=False)
                    response_data["pronunciation_corrections"] = corrections
            
            elif isinstance(result, dict):
                quality_analysis = convert_numpy_types(result.get("quality_analysis", {}))
                response_data["quality_analysis"] = quality_analysis
                
                if result.get("success", False):
                    record.transcription = result.get("transcription", "")
                    response_data["transcription"] = result.get("transcription", "")
                    response_data["success"] = True
                    response_data["message"] = "Fichier enregistré et transcrit avec succès"
                    
                    # NEW: Apply pronunciation correction if available
                    if pronunciation_corrector and texte:
                        corrections = pronunciation_corrector.correct_pronunciation(
                            original_text=texte.texteContent,
                            transcribed_text=result.get("transcription", ""),
                            audio_output_dir=AUDIO_CORRECTIONS_FOLDER
                        )
                        record.pronunciation_corrections = json.dumps(corrections, ensure_ascii=False)
                        response_data["pronunciation_corrections"] = corrections
                else:
                    response_data["success"] = False
                    response_data["message"] = "Qualité audio insuffisante"
                    response_data["errors"] = quality_analysis.get("errors", [])
                    response_data["warnings"] = quality_analysis.get("warnings", [])
                    response_data["error_details"] = result.get("error", "Erreur inconnue")
                    db.session.commit()
                    return jsonify(response_data), 422
            
            db.session.commit()
            return jsonify(response_data)
            
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Format de fichier non autorisé"}), 400


@app.route("/uploadd", methods=["POST"])
def upload_and_transcribee():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            id_eleve = int(request.form.get("id_eleve", 1))
            idTexte = int(request.form.get("idTexte", 1))
            
            # Retrieve original text from Texte table
            texte = Texte.query.filter_by(idTexte=idTexte).first()
            if not texte:
                return jsonify({"error": f"Texte avec idTexte={idTexte} non trouvé"}), 404
            
            # Create database record
            record = Recorder(id_eleve=id_eleve, idTexte=idTexte, file_path=filepath)
            db.session.add(record)
            db.session.commit()
            
            # Process audio (transcription only, no quality analysis)
            transcription = processor.transcribe_audio(record.file_path)
            
            # Debug: Print transcription
            print(f"DEBUG - Transcription: {transcription}")
            
            response_data = {
                "success": True,
                "record_id": record.id,
                "message": "Fichier enregistré et transcrit avec succès",
                "transcription": transcription,
                "pronunciation_corrections": {}
            }
            
            # Store transcription in database
            record.transcription = transcription
            
            # Apply pronunciation correction if available
            if pronunciation_corrector and texte:
                corrections = pronunciation_corrector.correct_pronunciation(
                    original_text=texte.texteContent,
                    transcribed_text=transcription,
                    audio_output_dir=AUDIO_CORRECTIONS_FOLDER
                )
                record.pronunciation_corrections = json.dumps(corrections, ensure_ascii=False)
                response_data["pronunciation_corrections"] = corrections
            
            db.session.commit()
            return jsonify(response_data)
            
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Format de fichier non autorisé"}), 400        

@app.route("/analyze_quality/<int:record_id>", methods=["GET"])
def analyze_audio_quality(record_id):
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    try:
        quality_result = processor.quality_analyzer.analyze_audio_quality(record.file_path)
        quality_result = convert_numpy_types(quality_result)
        return jsonify({
            "record_id": record.id,
            "quality_analysis": quality_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/audio_corrections/<path:filename>')
def serve_audio_corrections(filename):
    return send_from_directory(app.config['AUDIO_CORRECTIONS_FOLDER'], filename)

@app.route("/retry_transcription/<int:record_id>", methods=["POST"])
def retry_transcription(record_id):
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    try:
        transcription = processor.transcribe_audio(record.file_path)
        record.transcription = transcription
        
        # NEW: Re-run pronunciation correction if available
        if pronunciation_corrector:
            texte = Texte.query.filter_by(idTexte=record.idTexte).first()
            if texte:
                corrections = pronunciation_corrector.correct_pronunciation(
                    original_text=texte.texteContent,
                    transcribed_text=transcription,
                    audio_output_dir=AUDIO_CORRECTIONS_FOLDER
                )
                record.pronunciation_corrections = json.dumps(corrections, ensure_ascii=False)
        
        db.session.commit()
        return jsonify({
            "success": True,
            "message": "Transcription forcée réussie",
            "record_id": record.id,
            "transcription": transcription,
            "pronunciation_corrections": corrections if pronunciation_corrector and texte else {}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluer_lecture_diacritisee/<int:record_id>", methods=["GET"])
def evaluer_lecture_diacritisee_endpoint(record_id):
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    if not record.transcription:
        return jsonify({
            "error": "Aucune transcription disponible pour cet enregistrement.",
            "record_id": record.id
        }), 400
    
    texte = Texte.query.filter_by(idTexte=record.idTexte).first()
    if not texte:
        return jsonify({"error": f"Texte original non trouvé"}), 404
    
    # NEW: Use AzurePronunciationCorrector for diacritized evaluation
    score = 0
    corrections = {}
    if pronunciation_corrector:
        corrections = pronunciation_corrector.correct_pronunciation(
            original_text=texte.texteContent,
            transcribed_text=record.transcription,
            audio_output_dir=AUDIO_CORRECTIONS_FOLDER
        )
        # Calculate score based on number of errors
        total_words = len(texte.texteContent.split())
        error_count = corrections.get("total_errors", 0)
        score = max(0, 100 - (error_count / total_words * 100)) if total_words > 0 else 0
        record.pronunciation_corrections = json.dumps(corrections, ensure_ascii=False)
        db.session.commit()
    
    return jsonify({
        "record_id": record.id,
        "idTexte": record.idTexte,
        "score_lecture": score,
        "transcription": record.transcription,
        "texte_original": texte.texteContent,
        "pronunciation_corrections": corrections
    })

@app.route("/evaluate_reading/<int:record_id>", methods=["POST"])
def evaluate_reading_endpoint(record_id):
    if not reading_evaluator:
        return jsonify({
            "error": "خدمة تقييم القراءة غير متوفرة. يرجى التأكد من إعداد مفتاح Gemini API."
        }), 503
    
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    if not record.transcription:
        return jsonify({
            "error": "لا توجد نسخة نصية متاحة لهذا التسجيل.",
            "record_id": record.id
        }), 400
    
    texte = Texte.query.filter_by(idTexte=record.idTexte).first()
    if not texte:
        return jsonify({"error": f"النص الأصلي مع معرف {record.idTexte} غير موجود"}), 404
    
    try:
        # Evaluate reading using LLM
        evaluation = reading_evaluator.evaluate_reading(
            transcription=record.transcription,
            original_text=texte.texteContent
        )
        
        # NEW: Add pronunciation corrections
        pronunciation_corrections = {}
        if pronunciation_corrector:
            pronunciation_corrections = pronunciation_corrector.correct_pronunciation(
                original_text=texte.texteContent,
                transcribed_text=record.transcription,
                audio_output_dir=AUDIO_CORRECTIONS_FOLDER
            )
            record.pronunciation_corrections = json.dumps(pronunciation_corrections, ensure_ascii=False)
            db.session.commit()
        
        evaluation_data = {
            "record_id": record.id,
            "student_id": record.id_eleve,
            "text_id": record.idTexte,
            "evaluation": {
                "overall_score": evaluation.overall_score,
                "level": evaluation.level.value,
                "scores": {
                    "pronunciation": evaluation.pronunciation_score,
                    "fluency": evaluation.fluency_score,
                    "accuracy": evaluation.accuracy_score,
                    "comprehension": evaluation.comprehension_score
                },
                "feedback": evaluation.feedback,
                "strengths": evaluation.strengths,
                "areas_to_improve": evaluation.areas_to_improve,
                "suggestions": evaluation.suggestions,
                "detailed_feedback": evaluation.detailed_feedback
            },
            "texts": {
                "original": texte.texteContent,
                "transcribed": record.transcription
            },
            "pronunciation_corrections": pronunciation_corrections,  # NEW: Include pronunciation corrections
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify({
            "success": True,
            "message": "تم تقييم القراءة بنجاح",
            "data": evaluation_data
        })
        
    except Exception as e:
        return jsonify({
            "error": f"حدث خطأ أثناء تقييم القراءة: {str(e)}",
            "record_id": record.id
        }), 500

@app.route("/test_evaluate_reading/<int:record_id>", methods=["POST"])
def test_evaluate_reading(record_id):
    if not reading_evaluator:
        return jsonify({
            "error": "خدمة تقييم القراءة غير متوفرة. يرجى التأكد من إعداد مفتاح Gemini API."
        }), 503
    
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    if not record.transcription:
        return jsonify({
            "error": "لا توجد نسخة نصية متاحة لهذا التسجيل.",
            "record_id": record.id
        }), 400
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "لم يتم إرسال بيانات JSON"}), 400
    
    original_text = data.get('original_text', '').strip()
    if not original_text:
        return jsonify({"error": "النص الأصلي مطلوب"}), 400
    
    try:
        evaluation = reading_evaluator.evaluate_reading(
            transcription=record.transcription,
            original_text=original_text
        )
        
        # NEW: Add pronunciation corrections
        pronunciation_corrections = {}
        if pronunciation_corrector:
            pronunciation_corrections = pronunciation_corrector.correct_pronunciation(
                original_text=original_text,
                transcribed_text=record.transcription,
                audio_output_dir=AUDIO_CORRECTIONS_FOLDER
            )
            record.pronunciation_corrections = json.dumps(pronunciation_corrections, ensure_ascii=False)
            db.session.commit()
        
        evaluation_data = {
            "record_id": record.id,
            "student_id": record.id_eleve,
            "text_id": record.idTexte,
            "evaluation": {
                "overall_score": evaluation.overall_score,
                "level": evaluation.level.value,
                "scores": {
                    "pronunciation": evaluation.pronunciation_score,
                    "fluency": evaluation.fluency_score,
                    "accuracy": evaluation.accuracy_score,
                    "comprehension": evaluation.comprehension_score
                },
                "feedback": evaluation.feedback,
                "strengths": evaluation.strengths,
                "areas_to_improve": evaluation.areas_to_improve,
                "suggestions": evaluation.suggestions,
                "detailed_feedback": evaluation.detailed_feedback
            },
            "texts": {
                "original": original_text,
                "transcribed": record.transcription
            },
            "pronunciation_corrections": pronunciation_corrections,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify({
            "success": True,
            "message": "تم تقييم القراءة بنجاح",
            "data": evaluation_data
        })
        
    except Exception as e:
        return jsonify({
            "error": f"حدث خطأ أثناء تقييم القراءة: {str(e)}",
            "record_id": record.id
        }), 500

@app.route("/evaluate_reading_quick", methods=["POST"])
def evaluate_reading_quick():
    if not reading_evaluator:
        return jsonify({
            "error": "خدمة تقييم القراءة غير متوفرة"
        }), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "لم يتم إرسال بيانات JSON"}), 400
    
    transcription = data.get('transcription', '').strip()
    original_text = data.get('original_text', '').strip()
    
    if not transcription or not original_text:
        return jsonify({
            "error": "النص المنطوق والنص الأصلي مطلوبان"
        }), 400
    
    try:
        evaluation = reading_evaluator.evaluate_reading(transcription, original_text)
        
        # NEW: Add pronunciation corrections
        pronunciation_corrections = {}
        if pronunciation_corrector:
            pronunciation_corrections = pronunciation_corrector.correct_pronunciation(
                original_text=original_text,
                transcribed_text=transcription,
                audio_output_dir=AUDIO_CORRECTIONS_FOLDER
            )
        
        return jsonify({
            "success": True,
            "evaluation": {
                "overall_score": evaluation.overall_score,
                "level": evaluation.level.value,
                "scores": {
                    "pronunciation": evaluation.pronunciation_score,
                    "fluency": evaluation.fluency_score,
                    "accuracy": evaluation.accuracy_score,
                    "comprehension": evaluation.comprehension_score
                },
                "feedback": evaluation.feedback,
                "strengths": evaluation.strengths,
                "areas_to_improve": evaluation.areas_to_improve,
                "suggestions": evaluation.suggestions
            },
            "pronunciation_corrections": pronunciation_corrections
        })
        
    except Exception as e:
        return jsonify({
            "error": f"حدث خطأ أثناء التقييم: {str(e)}"
        }), 500

# NEW: Endpoint to retrieve audio feedback files
@app.route("/get_audio_feedback/<int:record_id>", methods=["GET"])
def get_audio_feedback(record_id):
    if not pronunciation_corrector:
        return jsonify({
            "error": "خدمة تصحيح النطق غير متوفرة. يرجى التأكد من إعداد مفتاح Azure Speech."
        }), 503
    
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    if not record.pronunciation_corrections:
        return jsonify({
            "error": "Aucune correction de prononciation disponible pour cet enregistrement.",
            "record_id": record.id
        }), 400
    
    try:
        corrections = json.loads(record.pronunciation_corrections)
        audio_files = {
            "corrected_text_audio": corrections.get("corrected_text_audio"),
            "feedback_audio": corrections.get("feedback_audio"),
            "individual_corrections": [
                error["audio_file"] for error in corrections.get("errors", []) if error.get("audio_file")
            ]
        }
        return jsonify({
            "success": True,
            "message": "Fichiers audio de correction récupérés avec succès",
            "record_id": record.id,
            "audio_files": audio_files
        })
    except Exception as e:
        return jsonify({
            "error": f"Erreur lors de la récupération des fichiers audio: {str(e)}",
            "record_id": record.id
        }), 500

if __name__ == "__main__":
    app.run(debug=True,host='127.0.0.1',port=5005)