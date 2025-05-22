# app.py
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import json
import numpy as np

from audio_processor import ArabicAudioProcessor


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'ogg', 'mp3', 'm4a'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost:3306/Agentiai'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
processor = ArabicAudioProcessor()

class Recorder(db.Model):
    __tablename__ = 'recorder'
    id = db.Column(db.Integer, primary_key=True)
    id_eleve = db.Column(db.Integer, nullable=False)
    idTexte = db.Column(db.Integer, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    # Supprimé: quality_analysis = db.Column(db.Text, nullable=True)
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
    """Convertit les types NumPy en types Python natifs pour la sérialisation JSON"""
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

@app.route("/evaluer")
def evaluer_page():
    return render_template("evaluer.html")

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
            
            # Créer l'enregistrement dans la base de données
            record = Recorder(id_eleve=id_eleve, idTexte=idTexte, file_path=filepath)
            db.session.add(record)
            db.session.commit()
            
            # Traiter l'audio avec analyse de qualité
            result = processor.process_audio(record.file_path)
            
            # Debug: afficher le type et le contenu de result
            print(f"DEBUG - Type de result: {type(result)}")
            print(f"DEBUG - Contenu de result: {result}")
            
            # Gérer le cas où l'ancienne version retourne juste une string (transcription)
            if isinstance(result, str):
                # Ancienne version - juste la transcription
                record.transcription = result
                db.session.commit()
                
                return jsonify({
                    "success": True,
                    "message": "Fichier enregistré et transcrit avec succès",
                    "record_id": record.id,
                    "transcription": result
                })
            
            # Nouvelle version - dictionnaire avec analyse de qualité
            elif isinstance(result, dict):
                quality_analysis = convert_numpy_types(result.get("quality_analysis", {}))
                
                if result.get("success", False):
                    # Si l'audio est de bonne qualité, sauvegarder la transcription
                    record.transcription = result.get("transcription", "")
                    db.session.commit()
                    
                    return jsonify({
                        "success": True,
                        "message": "Fichier enregistré et transcrit avec succès",
                        "record_id": record.id,
                        "transcription": result.get("transcription", ""),
                        "quality_analysis": quality_analysis
                    })
                else:
                    # Si l'audio n'est pas de bonne qualité, retourner les erreurs
                    db.session.commit()
                    
                    return jsonify({
                        "success": False,
                        "message": "Qualité audio insuffisante",
                        "record_id": record.id,
                        "quality_analysis": quality_analysis,
                        "errors": quality_analysis.get("errors", []),
                        "warnings": quality_analysis.get("warnings", []),
                        "error_details": result.get("error", "Erreur inconnue")
                    }), 422
            
            else:
                # Type inattendu
                return jsonify({
                    "error": f"Erreur interne: type de résultat inattendu. Type: {type(result)}, Valeur: {result}"
                }), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Format de fichier non autorisé"}), 400

@app.route("/analyze_quality/<int:record_id>", methods=["GET"])
def analyze_audio_quality(record_id):
    """Endpoint pour analyser la qualité d'un audio déjà uploadé"""
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    try:
        quality_result = processor.quality_analyzer.analyze_audio_quality(record.file_path)
        
        # Convertir les types NumPy avant la sérialisation JSON
        quality_result = convert_numpy_types(quality_result)
        
        # Ne plus sauvegarder l'analyse dans la base de données
        # Retourner seulement le résultat
        return jsonify({
            "record_id": record.id,
            "quality_analysis": quality_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/retry_transcription/<int:record_id>", methods=["POST"])
def retry_transcription(record_id):
    """Endpoint pour forcer la transcription même si la qualité n'est pas parfaite"""
    record = Recorder.query.get(record_id)
    if not record:
        return jsonify({"error": f"Enregistrement avec id={record_id} non trouvé"}), 404
    
    try:
        # Forcer la transcription sans vérification de qualité
        transcription = processor.transcribe_audio(record.file_path)
        record.transcription = transcription
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Transcription forcée réussie",
            "record_id": record.id,
            "transcription": transcription
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
            "error": "Aucune transcription disponible pour cet enregistrement. Veuillez d'abord améliorer la qualité audio.",
            "record_id": record.id
        }), 400
    
    texte = Texte.query.filter_by(idTexte=record.idTexte).first()
    if not texte:
        return jsonify({"error": f"Texte original avec idTexte={record.idTexte} non trouvé"}), 404
    
    score = evaluer_lecture_diacritisee(record.transcription, texte.texteContent)
    
    return jsonify({
        "record_id": record.id,
        "idTexte": record.idTexte,
        "score_lecture": score,
        "transcription": record.transcription,
        "texte_original": texte.texteContent
    })

if __name__ == "__main__":
    app.run(debug=True)