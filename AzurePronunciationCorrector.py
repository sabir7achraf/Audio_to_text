import azure.cognitiveservices.speech as speechsdk
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import os

@dataclass
class PronunciationError:
    """Classe pour représenter une erreur de prononciation"""
    word_original: str
    word_transcribed: str
    word_with_diacritics: str
    pronunciation_score: float
    error_type: str  # 'missing_diacritics', 'wrong_pronunciation', 'omitted_word'
    position: int
    audio_feedback_path: Optional[str] = None

class AzurePronunciationCorrector:
    """Correcteur de prononciation utilisant Azure Speech Services"""
    
    def __init__(self, subscription_key: str, region: str, language: str = "ar-SA"):
        """
        Initialise le correcteur de prononciation Azure
        
        Args:
            subscription_key: Clé d'abonnement Azure Speech
            region: Région Azure (ex: "eastus", "westeurope")
            language: Code de langue (ar-SA pour l'arabe saoudien)
        """
        self.subscription_key = subscription_key
        self.region = region
        self.language = language
        
        # Configuration Azure Speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, 
            region=region
        )
        self.speech_config.speech_recognition_language = language
        self.speech_config.speech_synthesis_language = language
        
        # Configuration pour la synthèse vocale avec voix masculine/féminine
        self.speech_config.speech_synthesis_voice_name = "ar-SA-HamedNeural"  # Voix masculine
        # Alternative: "ar-SA-ZariyahNeural" pour voix féminine
        
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        
        # Dictionnaire des diacritiques arabes courantes
        self.arabic_diacritics = {
            'َ': 'fatha',      # Fat-ha
            'ِ': 'kasra',      # Kasra  
            'ُ': 'damma',      # Damma
            'ْ': 'sukun',      # Sukun
            'ّ': 'shadda',     # Shadda
            'ً': 'tanwin_fath', # Tanwin Fath
            'ٍ': 'tanwin_kasr', # Tanwin Kasr
            'ٌ': 'tanwin_damm', # Tanwin Damm
            'ـ': 'tatweel'     # Tatweel
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def add_diacritics_to_text(self, text: str) -> str:
        """
        Ajoute les diacritiques au texte arabe en utilisant des règles linguistiques
        Cette méthode peut être améliorée avec une API de vocalisation
        """
        # Pour une implémentation complète, vous devriez utiliser une API de vocalisation
        # comme Mishkal ou Harakat, ici nous utilisons des règles basiques
        
        # Règles basiques pour les articles et prépositions courantes
        diacritics_rules = {
            r'\bال([بتثجحخدذرزسشصضطظعغفقكلمنهوي])': r'الْ\1',  # Article défini
            r'\bفي\b': 'فِي',
            r'\bمن\b': 'مِنْ', 
            r'\bإلى\b': 'إِلَى',
            r'\bعلى\b': 'عَلَى',
            r'\bهذا\b': 'هَذَا',
            r'\bهذه\b': 'هَذِهِ',
            r'\bذلك\b': 'ذَلِكَ',
            r'\bالله\b': 'اللَّهُ',
            r'\bمحمد\b': 'مُحَمَّدٌ',
        }
        
        vocalized_text = text
        for pattern, replacement in diacritics_rules.items():
            vocalized_text = re.sub(pattern, replacement, vocalized_text)
            
        return vocalized_text

    def compare_words(self, original_word: str, transcribed_word: str) -> Tuple[bool, float]:
        """
        Compare deux mots arabes et retourne si ils correspondent et un score de similarité
        """
        # Enlever les diacritiques pour la comparaison de base
        def remove_diacritics(text):
            arabic_diacritics = 'َُِْٰٱّ'
            return ''.join([char for char in text if char not in arabic_diacritics])
        
        original_clean = remove_diacritics(original_word)
        transcribed_clean = remove_diacritics(transcribed_word)
        
        # Calcul de similarité simple (peut être amélioré avec Levenshtein)
        if original_clean == transcribed_clean:
            return True, 1.0
        
        # Calcul de similarité approximative
        shorter = min(len(original_clean), len(transcribed_clean))
        longer = max(len(original_clean), len(transcribed_clean))
        
        if longer == 0:
            return True, 1.0
            
        matches = sum(1 for i in range(shorter) if i < len(original_clean) and i < len(transcribed_clean) and original_clean[i] == transcribed_clean[i])
        similarity = matches / longer
        
        return similarity > 0.7, similarity

    def identify_pronunciation_errors(self, original_text: str, transcribed_text: str) -> List[PronunciationError]:
        """
        Identifie les erreurs de prononciation en comparant le texte original et la transcription
        """
        errors = []
        
        # Ajouter les diacritiques au texte original
        original_with_diacritics = self.add_diacritics_to_text(original_text)
        
        # Diviser en mots
        original_words = original_text.split()
        transcribed_words = transcribed_text.split()
        original_diac_words = original_with_diacritics.split()
        
        # Comparaison mot par mot
        max_len = max(len(original_words), len(transcribed_words))
        
        for i in range(max_len):
            original_word = original_words[i] if i < len(original_words) else ""
            transcribed_word = transcribed_words[i] if i < len(transcribed_words) else ""
            diac_word = original_diac_words[i] if i < len(original_diac_words) else original_word
            
            if not original_word and transcribed_word:
                # Mot ajouté
                continue
            elif original_word and not transcribed_word:
                # Mot omis
                errors.append(PronunciationError(
                    word_original=original_word,
                    word_transcribed="",
                    word_with_diacritics=diac_word,
                    pronunciation_score=0.0,
                    error_type="omitted_word",
                    position=i
                ))
            elif original_word and transcribed_word:
                # Comparer les mots
                is_similar, score = self.compare_words(original_word, transcribed_word)
                
                if not is_similar or score < 0.8:
                    error_type = "wrong_pronunciation" if score > 0.3 else "missing_diacritics"
                    errors.append(PronunciationError(
                        word_original=original_word,
                        word_transcribed=transcribed_word,
                        word_with_diacritics=diac_word,
                        pronunciation_score=score,
                        error_type=error_type,
                        position=i
                    ))
        
        return errors

    def generate_audio_feedback(self, word_with_diacritics: str, output_path: str) -> bool:
     try:
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )


        ssml_text = f"""
         <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
          <voice name="{self.speech_config.speech_synthesis_voice_name}">
          <prosody rate="slow" pitch="medium">
            {word_with_diacritics}
            </prosody>
          </voice>
          </speak>
        """


        result = synthesizer.speak_ssml_async(ssml_text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            self.logger.info(f"Audio généré avec succès: {output_path}")
            return True
        else:
            self.logger.error(f"Erreur de synthèse: {result.reason}")
            return False
     except Exception as e:
        self.logger.error(f"Exception dans la synthèse: {str(e)}")
        return False


    def generate_corrected_text_audio(self, original_text: str, output_path: str, speed: str = "medium") -> bool:
        """
        Génère un fichier audio avec la lecture complète du texte corrigé
        
        Args:
            original_text: Le texte original à corriger et lire
            output_path: Chemin du fichier audio de sortie
            speed: Vitesse de lecture ("slow", "medium", "fast")
        
        Returns:
            bool: True si la génération a réussi, False sinon
        """
        try:
            # Ajouter les diacritiques au texte complet
            corrected_text = self.add_diacritics_to_text(original_text)
            
            # Configuration pour sauvegarder dans un fichier
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # Créer le texte SSML pour une meilleure prononciation avec pauses
            ssml_text = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                    <prosody rate="{speed}" pitch="medium" volume="loud">
                        {corrected_text}
                    </prosody>
                </voice>
            </speak>
            """
            
            
            # Synthétiser la parole
            result = synthesizer.speak_ssml_async(ssml_text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                self.logger.info(f"Audio de lecture complète généré avec succès: {output_path}")
                return True
            else:
                self.logger.error(f"Erreur lors de la génération de l'audio complet: {result.reason}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de l'audio complet: {str(e)}")
            return False

    def generate_comprehensive_feedback_audio(self, errors: List[PronunciationError], output_path: str) -> bool:
        """
        Génère un feedback audio complet pour toutes les corrections
        """
        try:
            if not errors:
                feedback_text = "أحسنت! لا توجد أخطاء في النطق. استمر في التدريب."
            else:
                feedback_text = "الأخطاء في النطق هي: "
                
                for error in errors[:5]:  # Limiter à 5 erreurs pour éviter un audio trop long
                    if error.error_type == "omitted_word":
                        feedback_text += f"كلمة مفقودة: {error.word_with_diacritics}. "
                    elif error.error_type == "wrong_pronunciation":
                        feedback_text += f"النطق الصحيح لكلمة {error.word_original} هو {error.word_with_diacritics}. "
                    else:
                        feedback_text += f"انتبه للحركات في كلمة {error.word_with_diacritics}. "
                
                feedback_text += "حاول مرة أخرى مع التركيز على الحركات."
            
            # Configuration pour sauvegarder dans un fichier
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # Créer le texte SSML pour le feedback
            ssml_text = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                    <prosody rate="medium" pitch="medium">
                        {feedback_text}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = synthesizer.speak_ssml_async(ssml_text).get()
            
            return result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du feedback audio: {str(e)}")
            return False

    def correct_pronunciation(self, original_text: str, transcribed_text: str, audio_output_dir: str = "audio_corrections") -> Dict:
        """
        Fonction principale pour corriger la prononciation
        
        Returns:
            Dict contenant les erreurs identifiées et les chemins vers les fichiers audio de correction
        """
        # Créer le dossier de sortie audio s'il n'existe pas
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # Identifier les erreurs
        errors = self.identify_pronunciation_errors(original_text, transcribed_text)
        
        correction_results = {
            "total_errors": len(errors),
            "errors": [],
            "audio_files": [],
            "corrected_text": self.add_diacritics_to_text(original_text),
            "corrected_text_audio": None,
            "feedback_audio": None,
            "summary": {
                "omitted_word": 0,
                "wrong_pronunciation": 0,
                "missing_diacritics": 0
            }
        }
        
        # Traiter chaque erreur
        for i, error in enumerate(errors):
            # Nom du fichier audio pour cette correction
            audio_filename = f"correction_{i+1}_{error.word_original.replace(' ', '_')}.wav"
            audio_path = os.path.join(audio_output_dir, audio_filename)
            
            # Générer l'audio de correction
            if self.generate_audio_feedback(error.word_with_diacritics, audio_path):
                error.audio_feedback_path = audio_path
                correction_results["audio_files"].append(audio_path)
            
            # Ajouter l'erreur aux résultats
            correction_results["errors"].append({
                "position": error.position,
                "original_word": error.word_original,
                "transcribed_word": error.word_transcribed,
                "correct_pronunciation": error.word_with_diacritics,
                "pronunciation_score": error.pronunciation_score,
                "error_type": error.error_type,
                "audio_file": error.audio_feedback_path
            })
            
            # Mettre à jour le résumé
            correction_results["summary"][error.error_type] += 1
        
        # Générer l'audio de lecture complète corrigée
        corrected_text_audio_path = os.path.join(audio_output_dir, "corrected_reading_complete.wav")
        if self.generate_corrected_text_audio(original_text, corrected_text_audio_path):
            correction_results["corrected_text_audio"] = corrected_text_audio_path
            self.logger.info(f"Audio de lecture complète généré: {corrected_text_audio_path}")
        
        # Générer le feedback audio général
        feedback_audio_path = os.path.join(audio_output_dir, "pronunciation_feedback.wav")
        if self.generate_comprehensive_feedback_audio(errors, feedback_audio_path):
            correction_results["feedback_audio"] = feedback_audio_path
            self.logger.info(f"Feedback audio généré: {feedback_audio_path}")
        
        return correction_results

    def generate_learning_sequence_audio(self, original_text: str, output_dir: str = "learning_sequence") -> Dict:
        """
        Génère une séquence d'apprentissage complète avec différentes vitesses de lecture
        
        Returns:
            Dict contenant les chemins vers les différents fichiers audio générés
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sequence_files = {}
        
        # 1. Lecture très lente pour l'apprentissage
        slow_path = os.path.join(output_dir, "01_lecture_lente.wav")
        if self.generate_corrected_text_audio(original_text, slow_path, "slow"):
            sequence_files["slow_reading"] = slow_path
        
        # 2. Lecture normale
        normal_path = os.path.join(output_dir, "02_lecture_normale.wav")
        if self.generate_corrected_text_audio(original_text, normal_path, "medium"):
            sequence_files["normal_reading"] = normal_path
        
        # 3. Lecture rapide pour la fluidité
        fast_path = os.path.join(output_dir, "03_lecture_rapide.wav")
        if self.generate_corrected_text_audio(original_text, fast_path, "fast"):
            sequence_files["fast_reading"] = fast_path
        
        # 4. Instructions d'apprentissage
        instructions_path = os.path.join(output_dir, "00_instructions.wav")
        instructions_text = """
        مرحباً بك في جلسة تعلم النطق. 
        أولاً، استمع للقراءة البطيئة وركز على الحركات.
        ثانياً، استمع للقراءة العادية وحاول المتابعة.
        ثالثاً، استمع للقراءة السريعة لتحسين الطلاقة.
        كرر هذه العملية حتى تتقن النطق الصحيح.
        """
        
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=instructions_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            result = synthesizer.speak_text_async(instructions_text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                sequence_files["instructions"] = instructions_path
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des instructions: {str(e)}")
        
        return sequence_files

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration Azure (remplacez par vos vraies clés)
    AZURE_SPEECH_KEY = "F2d84URhnc9EhA8UG2QCbAAEWzGIeVnbgmQpyWC21OlXytPzbPxYJQQJ99BEACYeBjFXJ3w3AAAYACOGbTan"
    AZURE_REGION = "eastus"
    
    # Initialiser le correcteur
    corrector = AzurePronunciationCorrector(AZURE_SPEECH_KEY, AZURE_REGION)
    
    # Exemple de texte
    original_text = "مَحْمُودٌ وَالِدُ زَيْدٍ وهُوَ يَعْمَلُ فِي شَرِيكَةٍ فِي الْعَاصِمَةِ يَعْمَلُ سَبْعَ سَاعَاتٍ فِي الْيَوْمِ وهُوَ يُحِبُّ عَمَلَهُ فاطِمَةُ وَالِدَةُ زَيْدٍ وَهِيَ  وهِيَ طَبِيبَةٌ فِي مُسْتَشْفَى خَاصٍّ الْمُسْتَشْفَى قَرِيبٌ مِنَ الْبَيْتِ وَهِيَ تَعْمَلُ صَبَاحًا فَقَطْ"
    transcribed_text = "مَحْمُودًا والِدَةُ زَيدًا وَهُوَ يَعْمَل فِي شَرِيكَةٍ فِي ألْعاصِمَةَ يَعْمَلِ سبعُ سَاعاةً فِي الْيَومِ وَهُوَ يُحِبَّ عمَلَهُ فاطِمَةٌ والِدَةَ زَيْدٍ واهِيََ طَبِيبَةُ فِي مُستَشْفَى خَاصّا الْمُستَشْفَى قَرِيبٌ مينَ الْبَيتَ وهِيَ تَعْمَلُ صَبَاحٌ فَقَط"
    
    # Corriger la prononciation
    results = corrector.correct_pronunciation(original_text, transcribed_text)
    
    print("=== Résultats de correction ===")
    print(f"Nombre total d'erreurs: {results['total_errors']}")
    print(f"Fichiers audio générés: {len(results['audio_files'])}")
    print(f"Texte corrigé: {results['corrected_text']}")
    print(f"Audio de lecture complète: {results['corrected_text_audio']}")
    print(f"Audio de feedback: {results['feedback_audio']}")
    
    print("\n=== Détails des erreurs ===")
    for error in results['errors']:
        print(f"- Position {error['position']}: '{error['original_word']}' -> '{error['correct_pronunciation']}'")
        print(f"  Type d'erreur: {error['error_type']}")
        print(f"  Fichier audio: {error['audio_file']}")
    
    # Générer une séquence d'apprentissage complète
    print("\n=== Génération de la séquence d'apprentissage ===")
    learning_files = corrector.generate_learning_sequence_audio(original_text)
    
    for key, path in learning_files.items():
        print(f"{key}: {path}")