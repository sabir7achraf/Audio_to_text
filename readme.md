# Arabic Reading Evaluation API

This Flask application provides an API for evaluating Arabic reading proficiency using audio input. It transcribes audio, analyzes reading quality, corrects pronunciation using Azure Speech Services, and evaluates reading performance with a Gemini-based language model. The application stores audio files, transcriptions, and correction results in a MySQL database.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

## Features
- Upload and transcribe Arabic audio files (`.wav`, `.ogg`, `.mp3`, `.m4a`).
- Analyze audio quality (e.g., noise, clarity).
- Correct pronunciation using Azure Speech Services, including diacritics and audio feedback.
- Evaluate reading proficiency using a Gemini-based language model.
- Store results in a MySQL database.
- Serve audio correction files (e.g., feedback audio) statically.

## Requirements
- **Python**: 3.8 or higher
- **MySQL**: 8.0 or higher
- **Dependencies** (listed in `requirements.txt`):
  - `flask`
  - `flask-sqlalchemy`
  - `pymysql`
  - `werkzeug`
  - `numpy`
  - `azure-cognitiveservices-speech`
  - `python-dotenv`
- **API Keys**:
  - Azure Speech Service subscription key and region.
  - Gemini API key for reading evaluation.
- **Folders**:
  - `uploads`: Stores uploaded audio files.
  - `audio_corrections`: Stores pronunciation correction audio files.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```
   AZURE_SPEECH_KEY=your-azure-speech-key
   AZURE_REGION=eastus
   Api_gemini=your-gemini-api-key
   ```
   Replace placeholders with actual keys. Alternatively, update `appo.py` with hardcoded keys (not recommended for production).

4. **Set Up MySQL Database**:
   - Create a database named `agentiai`:
     ```sql
     CREATE DATABASE agentiai;
     ```
   - Ensure MySQL credentials in `appo.py` (`mysql+pymysql://root:@localhost:3306/agentiai`) are correct.
   - The application automatically creates tables (`recorder`, `texte`) on startup via `db.create_all()`.

5. **Run the Application**:
   ```bash
   python appo.py
   ```
   The server runs on `http://localhost:5000` in debug mode.

## Database Schema

- **recorder**:
  - `id`: Integer, primary key.
  - `id_eleve`: Integer, student ID (required).
  - `idTexte`: Integer, text ID (required).
  - `file_path`: String(255), path to uploaded audio file (required).
  - `transcription`: Text, transcribed text (nullable).
  - `pronunciation_corrections`: Text, JSON of pronunciation corrections (nullable).
  - `date_enregistrement`: DateTime, recording timestamp (default: UTC now).

- **texte**:
  - `idTexte`: Integer, primary key.
  - `texteContent`: Text, original text content (required).

**Note**: If the `pronunciation_corrections` column is missing, apply a migration:
```sql
ALTER TABLE recorder ADD COLUMN pronunciation_corrections TEXT;
```

## API Endpoints

### `GET /`
- **Description**: Serves the main UI (`index.html`) for uploading audio files and viewing results.
- **Response**: HTML page.
- **Example**:
  ```bash
  curl http://localhost:5000/
  ```

### `POST /upload`
- **Description**: Uploads an audio file, transcribes it, analyzes quality, and applies pronunciation correction.
- **Request**:
  - Form-data:
    - `file`: Audio file (`.wav`, `.ogg`, `.mp3`, `.m4a`).
    - `id_eleve`: Integer, student ID (default: 1).
    - `idTexte`: Integer, text ID (default: 1).
- **Response**:
  - Success (200):
    ```json
    {
      "success": true,
      "record_id": 1,
      "message": "Fichier enregistré et transcrit avec succès",
      "transcription": "نص النسخ هنا",
      "quality_analysis": {},
      "pronunciation_corrections": {
        "total_errors": 2,
        "errors": [...],
        "corrected_text": "نص مصحح",
        "corrected_text_audio": "audio_corrections/corrected_reading_complete.wav",
        "feedback_audio": "audio_corrections/pronunciation_feedback.wav"
      }
    }
    ```
  - Error (400, 404, 500): `{ "error": "message" }`
- **Example**:
  ```bash
  curl -X POST -F "file=@sample.wav" -F "id_eleve=1" -F "idTexte=1" http://localhost:5000/upload
  ```

### `GET /analyze_quality/<record_id>`
- **Description**: Analyzes the audio quality of a recorded file.
- **Parameters**:
  - `record_id`: Integer, record ID.
- **Response**:
  - Success (200):
    ```json
    {
      "record_id": 1,
      "quality_analysis": {
        "noise_level": 0.1,
        "clarity": 0.9
      }
    }
    ```
  - Error (404, 500): `{ "error": "message" }`
- **Example**:
  ```bash
  curl http://localhost:5000/analyze_quality/1
  ```

### `POST /retry_transcription/<record_id>`
- **Description**: Re-transcribes an audio file and re-applies pronunciation correction.
- **Parameters**:
  - `record_id`: Integer, record ID.
- **Response**:
  - Success (200):
    ```json
    {
      "success": true,
      "message": "Transcription forcée réussie",
      "record_id": 1,
      "transcription": "نص النسخ الجديد",
      "pronunciation_corrections": {...}
    }
    ```
  - Error (404, 500): `{ "error": "message" }`
- **Example**:
  ```bash
  curl -X POST http://localhost:5000/retry_transcription/1
  ```

### `GET /evaluer_lecture_diacritisee/<record_id>`
- **Description**: Evaluates reading with diacritics-based pronunciation correction.
- **Parameters**:
  - `record_id`: Integer, record ID.
- **Response**:
  - Success (200):
    ```json
    {
      "record_id": 1,
      "idTexte": 1,
      "score_lecture": 85.0,
      "transcription": "نص النسخ",
      "texte_original": "النص الأصلي",
      "pronunciation_corrections": {...}
    }
    ```
  - Error (400, 404): `{ "error": "message" }`
- **Example**:
  ```bash
  curl http://localhost:5000/evaluer_lecture_diacritisee/1
  ```

### `POST /evaluate_reading/<record_id>`
- **Description**: Evaluates reading proficiency using the Gemini model and applies pronunciation correction.
- **Parameters**:
  - `record_id`: Integer, record ID.
- **Response**:
  - Success (200):
    ```json
    {
      "success": true,
      "message": "تم تقييم القراءة بنجاح",
      "data": {
        "record_id": 1,
        "student_id": 1,
        "text_id": 1,
        "evaluation": {
          "overall_score": 90,
          "level": "Advanced",
          "scores": {
            "pronunciation": 85,
            "fluency": 90,
            "accuracy": 88,
            "comprehension": 92
          },
          "feedback": "قراءة جيدة مع نطق واضح",
          "strengths": ["الطلاقة"],
          "areas_to_improve": ["النطق"],
          "suggestions": ["مارس النطق"],
          "detailed_feedback": {...}
        },
        "texts": {
          "original": "النص الأصلي",
          "transcribed": "نص النسخ"
        },
        "pronunciation_corrections": {...},
        "timestamp": "2025-05-28T22:35:00Z"
      }
    }
    ```
  - Error (400, 404, 503): `{ "error": "message" }`
- **Example**:
  ```bash
  curl -X POST http://localhost:5000/evaluate_reading/1
  ```

### `POST /test_evaluate_reading/<record_id>`
- **Description**: Tests reading evaluation with a custom original text.
- **Parameters**:
  - `record_id`: Integer, record ID.
- **Request**:
  - JSON body:
    ```json
    {
      "original_text": "النص الأصلي للاختبار"
    }
    ```
- **Response**: Similar to `/evaluate_reading/<record_id>`.
- **Example**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"original_text": "النص الأصلي"}' http://localhost:5000/test_evaluate_reading/1
  ```

### `POST /evaluate_reading_quick`
- **Description**: Evaluates reading without storing data, using provided transcription and original text.
- **Request**:
  - JSON body:
    ```json
    {
      "transcription": "نص النسخ",
      "original_text": "النص الأصلي"
    }
    ```
- **Response**:
  - Success (200):
    ```json
    {
      "success": true,
      "evaluation": {
        "overall_score": 90,
        "level": "Advanced",
        "scores": {...},
        "feedback": "قراءة جيدة",
        "strengths": [],
        "areas_to_improve": [],
        "suggestions": []
      },
      "pronunciation_corrections": {...}
    }
    ```
  - Error (400, 503): `{ "error": "message" }`
- **Example**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"transcription": "نص النسخ", "original_text": "النص الأصلي"}' http://localhost:5000/evaluate_reading_quick
  ```

### `GET /get_audio_feedback/<record_id>`
- **Description**: Retrieves audio feedback files for pronunciation corrections.
- **Parameters**:
  - `record_id`: Integer, record ID.
- **Response**:
  - Success (200):
    ```json
    {
      "success": true,
      "message": "Fichiers audio de correction récupérés avec succès",
      "record_id": 1,
      "audio_files": {
        "corrected_text_audio": "audio_corrections/corrected_reading_complete.wav",
        "feedback_audio": "audio_corrections/pronunciation_feedback.wav",
        "individual_corrections": [
          "audio_corrections/correction_1_word.wav",
          ...
        ]
      }
    }
    ```
  - Error (400, 404, 503): `{ "error": "message" }`
- **Example**:
  ```bash
  curl http://localhost:5000/get_audio_feedback/1
  ```

## Usage Examples

### Uploading an Audio File
1. Use the UI at `http://localhost:5000/` to upload an audio file.
2. Or via `curl`:
   ```bash
   curl -X POST -F "file=@sample.wav" -F "id_eleve=1" -F "idTexte=1" http://localhost:5000/upload
   ```
3. Note the `record_id` from the response.

### Evaluating Reading
```bash
curl -X POST http://localhost:5000/evaluate_reading/<record_id>
```

### Retrieving Audio Feedback
```bash
curl http://localhost:5000/get_audio_feedback/<record_id>
```
Access audio files at `http://localhost:5000/<audio_file_path>` (e.g., `http://localhost:5000/audio_corrections/pronunciation_feedback.wav`).

## File Structure
```
.
├── appo.py                  # Main Flask application
├── index.html              # UI for uploading and viewing results
├── .env                    # Environment variables (API keys)
├── uploads/                # Uploaded audio files
├── audio_corrections/      # Pronunciation correction audio files
├── audio_processor.py      # Audio processing logic
├── evaluator.py            # Reading evaluation logic
├── AzurePronunciationCorrector.py  # Pronunciation correction logic
└── requirements.txt        # Python dependencies
```

## Troubleshooting
- **404 for Audio Files**:
  - Ensure `audio_corrections` folder exists and contains the files.
  - Verify `AUDIO_CORRECTIONS_FOLDER` path in `appo.py`.
  - Add a static file serving route:
    ```python
    from flask import send_from_directory
    @app.route('/audio_corrections/<path:filename>')
    def serve_audio(filename):
        return send_from_directory(app.config['AUDIO_CORRECTIONS_FOLDER'], filename)
    ```

- **Database Errors**:
  - Check MySQL connection string.
  - Apply migrations for new columns (e.g., `pronunciation_corrections`).

- **API Key Issues**:
  - Ensure `.env` file is correctly formatted.
  - Verify Azure and Gemini API keys are valid.

- **Logs**:
  - Enable debug logging in `appo.py` to trace issues:
    ```python
    print(f"Debug: {variable}")
    ```

For further assistance, contact the project maintainer.
