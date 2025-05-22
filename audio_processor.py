import torch
import librosa
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from AudioQualityAnalyzer import AudioQualityAnalyzer

class ArabicAudioProcessor:
    def __init__(self):
        self.asr_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
        self.quality_analyzer = AudioQualityAnalyzer()

    def process_audio(self, audio_path):
        """
        Traite l'audio avec vérification de qualité avant transcription
        """
        # Étape 1: Analyser la qualité audio
        quality_result = self.quality_analyzer.analyze_audio_quality(audio_path)
        
        if not quality_result["valid"]:
            return {
                "success": False,
                "transcription": None,
                "quality_analysis": quality_result
            }
        
        # Étape 2: Si l'audio est valide, procéder à la transcription
        try:
            transcription = self.transcribe_audio(audio_path)
            return {
                "success": True,
                "transcription": transcription,
                "quality_analysis": quality_result
            }
        except Exception as e:
            return {
                "success": False,
                "transcription": None,
                "error": f"Erreur lors de la transcription: {str(e)}",
                "quality_analysis": quality_result
            }

    def transcribe_audio(self, audio_path, sample_rate=16000):
        """Transcrit l'audio en texte"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        
        speech_array, _ = librosa.load(audio_path, sr=sample_rate)
        inputs = self.asr_processor(speech_array, sampling_rate=sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.asr_model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.asr_processor.batch_decode(predicted_ids)[0]
        
        return transcription