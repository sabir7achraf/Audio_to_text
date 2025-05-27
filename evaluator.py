 # reading_evaluator.py
import google.generativeai as genai
import json
import difflib
import re
from typing import Dict, List, Any
import os
from dataclasses import dataclass
from enum import Enum

class ReadingLevel(Enum):
    EXCELLENT = "ممتاز"
    VERY_GOOD = "جيد جداً"
    GOOD = "جيد"
    ACCEPTABLE = "مقبول"
    NEEDS_IMPROVEMENT = "يحتاج تحسين"
    POOR = "ضعيف"

@dataclass
class ReadingEvaluation:
    overall_score: float
    level: ReadingLevel
    pronunciation_score: float
    fluency_score: float
    accuracy_score: float
    comprehension_score: float
    feedback: str
    detailed_feedback: Dict[str, Any]
    suggestions: List[str]
    strengths: List[str]
    areas_to_improve: List[str]

class ArabicReadingEvaluator:
    def __init__(self, api_key: str = None):
        """
        Initialize the Arabic Reading Evaluator
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from environment variable
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it directly.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier model
        
        # Arabic text processing patterns
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')  # Arabic diacritics and tatweel
        self.arabic_letters_pattern = re.compile(r'[\u0621-\u063A\u0641-\u064A]')  # Arabic letters
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text for comparison"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize some common variations
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')
        text = text.replace('ى', 'ي')
        
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics from text"""
        return self.diacritics_pattern.sub('', text)
    
    def calculate_accuracy_score(self, transcription: str, original_text: str) -> Dict[str, Any]:
        """Calculate detailed accuracy metrics"""
        # Normalize both texts
        norm_transcription = self.normalize_arabic_text(transcription)
        norm_original = self.normalize_arabic_text(original_text)
        
        # Remove diacritics for comparison
        clean_transcription = self.remove_diacritics(norm_transcription)
        clean_original = self.remove_diacritics(norm_original)
        
        # Word-level comparison
        trans_words = clean_transcription.split()
        orig_words = clean_original.split()
        
        # Use sequence matcher for similarity
        seq_matcher = difflib.SequenceMatcher(None, trans_words, orig_words)
        word_similarity = seq_matcher.ratio()
        
        # Character-level comparison
        char_matcher = difflib.SequenceMatcher(None, clean_transcription, clean_original)
        char_similarity = char_matcher.ratio()
        
        # Calculate word accuracy
        correct_words = 0
        total_words = len(orig_words)
        
        for i, orig_word in enumerate(orig_words):
            if i < len(trans_words):
                word_sim = difflib.SequenceMatcher(None, orig_word, trans_words[i]).ratio()
                if word_sim >= 0.8:  # Consider 80% similarity as correct
                    correct_words += 1
        
        word_accuracy = (correct_words / total_words * 100) if total_words > 0 else 0
        
        return {
            "word_accuracy": word_accuracy,
            "word_similarity": word_similarity * 100,
            "character_similarity": char_similarity * 100,
            "total_words_original": total_words,
            "total_words_transcribed": len(trans_words),
            "correct_words": correct_words,
            "missing_words": max(0, total_words - len(trans_words)),
            "extra_words": max(0, len(trans_words) - total_words)
        }
    
    def create_evaluation_prompt(self, transcription: str, original_text: str, accuracy_metrics: Dict) -> str:
        """Create a comprehensive prompt for LLM evaluation"""
        
        prompt = f"""
أنت خبير في تقييم قراءة النصوص العربية للطلاب. قم بتقييم قراءة الطالب بناءً على النص الأصلي والنص المنطوق.

النص الأصلي:
{original_text}

النص المنطوق (النسخة الصوتية):
{transcription}

معايير التقييم المطلوبة:
1. النطق والوضوح (Pronunciation): مدى وضوح نطق الكلمات والحروف
2. الطلاقة (Fluency): سلاسة القراءة وسرعتها المناسبة
3. الدقة (Accuracy): مطابقة النص المقروء للنص الأصلي
4. الفهم (Comprehension): فهم المعنى العام للنص

البيانات الإحصائية:
- دقة الكلمات: {accuracy_metrics['word_accuracy']:.1f}%
- تشابه الكلمات: {accuracy_metrics['word_similarity']:.1f}%
- تشابه الأحرف: {accuracy_metrics['character_similarity']:.1f}%
- إجمالي الكلمات الأصلية: {accuracy_metrics['total_words_original']}
- إجمالي الكلمات المنطوقة: {accuracy_metrics['total_words_transcribed']}
- الكلمات الصحيحة: {accuracy_metrics['correct_words']}
- الكلمات المفقودة: {accuracy_metrics['missing_words']}
- الكلمات الزائدة: {accuracy_metrics['extra_words']}

المطلوب منك:
1. تقييم شامل للقراءة مع درجة من 100 لكل معيار
2. تحديد نقاط القوة في القراءة
3. تحديد المجالات التي تحتاج إلى تحسين
4. تقديم اقتراحات محددة للتحسين
5. تقديم تغذية راجعة مشجعة ومفيدة

يرجى الرد بصيغة JSON بالشكل التالي:
{{
    "pronunciation_score": درجة من 100,
    "fluency_score": درجة من 100,
    "accuracy_score": درجة من 100,
    "comprehension_score": درجة من 100,
    "overall_score": المتوسط العام,
    "level": "مستوى القراءة (ممتاز/جيد جداً/جيد/مقبول/يحتاج تحسين/ضعيف)",
    "strengths": ["نقطة قوة 1", "نقطة قوة 2", "..."],
    "areas_to_improve": ["مجال التحسين 1", "مجال التحسين 2", "..."],
    "suggestions": ["اقتراح 1", "اقتراح 2", "..."],
    "detailed_analysis": {{
        "pronunciation_notes": "ملاحظات حول النطق",
        "fluency_notes": "ملاحظات حول الطلاقة",
        "accuracy_notes": "ملاحظات حول الدقة",
        "comprehension_notes": "ملاحظات حول الفهم"
    }},
    "encouragement": "رسالة تشجيعية للطالب",
    "specific_mistakes": ["خطأ محدد 1", "خطأ محدد 2", "..."],
    "improvement_priority": "أولوية التحسين الرئيسية"
}}

تأكد من أن التقييم عادل ومشجع ومفيد للطالب.
"""
        return prompt
    
    def determine_reading_level(self, overall_score: float) -> ReadingLevel:
        """Determine reading level based on overall score"""
        if overall_score >= 90:
            return ReadingLevel.EXCELLENT
        elif overall_score >= 80:
            return ReadingLevel.VERY_GOOD
        elif overall_score >= 70:
            return ReadingLevel.GOOD
        elif overall_score >= 60:
            return ReadingLevel.ACCEPTABLE
        elif overall_score >= 50:
            return ReadingLevel.NEEDS_IMPROVEMENT
        else:
            return ReadingLevel.POOR
    
    def evaluate_reading(self, transcription: str, original_text: str) -> ReadingEvaluation:
        """
        Evaluate Arabic reading using LLM
        
        Args:
            transcription: The transcribed audio text
            original_text: The original text that should be read
            
        Returns:
            ReadingEvaluation object with detailed assessment
        """
        try:
            # Calculate accuracy metrics first
            accuracy_metrics = self.calculate_accuracy_score(transcription, original_text)
            
            # Create evaluation prompt
            prompt = self.create_evaluation_prompt(transcription, original_text, accuracy_metrics)
            
            # Generate evaluation using Gemini
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON in the response (sometimes LLM adds extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                llm_evaluation = json.loads(json_text)
            else:
                raise ValueError("Could not extract JSON from LLM response")
            
            # Create comprehensive feedback
            comprehensive_feedback = self.create_comprehensive_feedback(
                llm_evaluation, accuracy_metrics, transcription, original_text
            )
            
            # Determine reading level
            overall_score = llm_evaluation.get('overall_score', 0)
            reading_level = self.determine_reading_level(overall_score)
            
            # Create evaluation object
            evaluation = ReadingEvaluation(
                overall_score=overall_score,
                level=reading_level,
                pronunciation_score=llm_evaluation.get('pronunciation_score', 0),
                fluency_score=llm_evaluation.get('fluency_score', 0),
                accuracy_score=llm_evaluation.get('accuracy_score', 0),
                comprehension_score=llm_evaluation.get('comprehension_score', 0),
                feedback=comprehensive_feedback,
                detailed_feedback={
                    'llm_analysis': llm_evaluation,
                    'accuracy_metrics': accuracy_metrics,
                    'reading_statistics': self.calculate_reading_statistics(transcription, original_text)
                },
                suggestions=llm_evaluation.get('suggestions', []),
                strengths=llm_evaluation.get('strengths', []),
                areas_to_improve=llm_evaluation.get('areas_to_improve', [])
            )
            
            return evaluation
            
        except Exception as e:
            # Fallback evaluation if LLM fails
            return self.create_fallback_evaluation(transcription, original_text, str(e))
    
    def create_comprehensive_feedback(self, llm_eval: Dict, accuracy_metrics: Dict, 
                                   transcription: str, original_text: str) -> str:
        """Create comprehensive feedback combining LLM analysis and metrics"""
        
        feedback_parts = []
        
        # Header with overall assessment
        overall_score = llm_eval.get('overall_score', 0)
        level = llm_eval.get('level', 'غير محدد')
        
        if overall_score >= 80:
            emoji = "🌟"
            tone = "ممتاز"
        elif overall_score >= 70:
            emoji = "👏"
            tone = "جيد جداً"
        elif overall_score >= 60:
            emoji = "👍"
            tone = "جيد"
        else:
            emoji = "💪"
            tone = "يحتاج إلى مزيد من التحسين"
        
        feedback_parts.append(f"{emoji} تقييم القراءة: {level} - {overall_score:.1f}/100")
        feedback_parts.append(f"التقدير العام: {tone}")
        
        # Scores breakdown
        feedback_parts.append("\n📊 تفصيل الدرجات:")
        feedback_parts.append(f"• النطق والوضوح: {llm_eval.get('pronunciation_score', 0):.1f}/100")
        feedback_parts.append(f"• الطلاقة: {llm_eval.get('fluency_score', 0):.1f}/100")
        feedback_parts.append(f"• الدقة: {llm_eval.get('accuracy_score', 0):.1f}/100")
        feedback_parts.append(f"• الفهم: {llm_eval.get('comprehension_score', 0):.1f}/100")
        
        # Statistical analysis
        feedback_parts.append(f"\n📈 التحليل الإحصائي:")
        feedback_parts.append(f"• دقة الكلمات: {accuracy_metrics['word_accuracy']:.1f}%")
        feedback_parts.append(f"• الكلمات الصحيحة: {accuracy_metrics['correct_words']} من {accuracy_metrics['total_words_original']}")
        
        if accuracy_metrics['missing_words'] > 0:
            feedback_parts.append(f"• كلمات مفقودة: {accuracy_metrics['missing_words']}")
        if accuracy_metrics['extra_words'] > 0:
            feedback_parts.append(f"• كلمات زائدة: {accuracy_metrics['extra_words']}")
        
        # Strengths
        strengths = llm_eval.get('strengths', [])
        if strengths:
            feedback_parts.append(f"\n✅ نقاط القوة:")
            for strength in strengths:
                feedback_parts.append(f"• {strength}")
        
        # Areas to improve
        areas = llm_eval.get('areas_to_improve', [])
        if areas:
            feedback_parts.append(f"\n🎯 مجالات التحسين:")
            for area in areas:
                feedback_parts.append(f"• {area}")
        
        # Specific suggestions
        suggestions = llm_eval.get('suggestions', [])
        if suggestions:
            feedback_parts.append(f"\n💡 اقتراحات للتحسين:")
            for suggestion in suggestions:
                feedback_parts.append(f"• {suggestion}")
        
        # Detailed analysis if available
        detailed = llm_eval.get('detailed_analysis', {})
        if detailed:
            feedback_parts.append(f"\n🔍 تحليل مفصل:")
            for key, value in detailed.items():
                if value and value.strip():
                    area_name = {
                        'pronunciation_notes': 'النطق',
                        'fluency_notes': 'الطلاقة',
                        'accuracy_notes': 'الدقة',
                        'comprehension_notes': 'الفهم'
                    }.get(key, key)
                    feedback_parts.append(f"• {area_name}: {value}")
        
        # Encouragement
        encouragement = llm_eval.get('encouragement', '')
        if encouragement:
            feedback_parts.append(f"\n🌟 {encouragement}")
        
        # Priority improvement
        priority = llm_eval.get('improvement_priority', '')
        if priority:
            feedback_parts.append(f"\n🎯 أولوية التحسين: {priority}")
        
        return "\n".join(feedback_parts)
    
    def calculate_reading_statistics(self, transcription: str, original_text: str) -> Dict[str, Any]:
        """Calculate additional reading statistics"""
        
        # Clean texts
        clean_trans = self.normalize_arabic_text(transcription)
        clean_orig = self.normalize_arabic_text(original_text)
        
        # Word counts
        trans_words = clean_trans.split()
        orig_words = clean_orig.split()
        
        # Character counts (Arabic letters only)
        trans_chars = len(self.arabic_letters_pattern.findall(clean_trans))
        orig_chars = len(self.arabic_letters_pattern.findall(clean_orig))
        
        # Calculate reading speed (assuming 1 word per second as baseline)
        estimated_reading_time = len(orig_words) * 1.0  # seconds
        
        return {
            "original_word_count": len(orig_words),
            "transcribed_word_count": len(trans_words),
            "original_char_count": orig_chars,
            "transcribed_char_count": trans_chars,
            "word_completion_rate": (len(trans_words) / len(orig_words) * 100) if orig_words else 0,
            "char_completion_rate": (trans_chars / orig_chars * 100) if orig_chars else 0,
            "estimated_reading_time": estimated_reading_time,
            "words_per_minute": (len(trans_words) / estimated_reading_time * 60) if estimated_reading_time > 0 else 0
        }
    
    def create_fallback_evaluation(self, transcription: str, original_text: str, error_msg: str) -> ReadingEvaluation:
        """Create a basic evaluation when LLM fails"""
        
        accuracy_metrics = self.calculate_accuracy_score(transcription, original_text)
        basic_score = accuracy_metrics['word_accuracy']
        
        # Simple level determination
        if basic_score >= 90:
            level = ReadingLevel.EXCELLENT
        elif basic_score >= 70:
            level = ReadingLevel.GOOD
        elif basic_score >= 50:
            level = ReadingLevel.ACCEPTABLE
        else:
            level = ReadingLevel.NEEDS_IMPROVEMENT
        
        fallback_feedback = f"""
⚠️ تقييم أساسي (حدث خطأ في التقييم المتقدم)

📊 النتيجة الأساسية: {basic_score:.1f}/100
📈 مستوى القراءة: {level.value}

📈 التحليل الإحصائي:
• دقة الكلمات: {accuracy_metrics['word_accuracy']:.1f}%
• الكلمات الصحيحة: {accuracy_metrics['correct_words']} من {accuracy_metrics['total_words_original']}
• الكلمات المفقودة: {accuracy_metrics['missing_words']}
• الكلمات الزائدة: {accuracy_metrics['extra_words']}

💡 نصائح عامة:
• تدرب على القراءة بصوت عالٍ
• اقرأ ببطء ووضوح
• راجع النص قبل القراءة
• تدرب على النطق الصحيح

تفاصيل الخطأ: {error_msg}
"""
        
        return ReadingEvaluation(
            overall_score=basic_score,
            level=level,
            pronunciation_score=basic_score,
            fluency_score=basic_score,
            accuracy_score=basic_score,
            comprehension_score=basic_score,
            feedback=fallback_feedback,
            detailed_feedback={
                'accuracy_metrics': accuracy_metrics,
                'error': error_msg
            },
            suggestions=["تدرب على القراءة بانتظام", "اقرأ ببطء ووضوح", "راجع النص قبل القراءة"],
            strengths=["محاولة جيدة للقراءة"],
            areas_to_improve=["دقة النطق", "طلاقة القراءة"]
        )
