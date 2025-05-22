import difflib
import unicodedata

def evaluer_lecture_diacritisee(transcription, texte_original):
    if not transcription or not texte_original:
        return 0.0
    
    transcription_words = transcription.split()
    texte_words = texte_original.split()

    seq_matcher = difflib.SequenceMatcher(None, transcription_words, texte_words)
    score = seq_matcher.ratio()
    
    return round(score * 100, 2)

