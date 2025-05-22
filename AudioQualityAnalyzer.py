import numpy as np
import librosa
import os
from scipy import signal
from scipy.stats import kurtosis

class AudioQualityAnalyzer:
    def __init__(self):
        self.min_duration = 2.0  # Durée minimale en secondes
        self.max_duration = 300.0  # Durée maximale en secondes
        self.min_snr = 10.0  # Rapport signal/bruit minimum en dB
        self.silence_threshold = 0.01  # Seuil de silence
        self.max_silence_ratio = 0.7  # Ratio maximum de silence
        
    def analyze_audio_quality(self, audio_path, sample_rate=16000):
        """
        Analyse la qualité de l'audio et retourne un dictionnaire avec les résultats en arabe
        """
        if not os.path.exists(audio_path):
            return {
                "valid": False,
                "error": "ملف الصوت غير موجود",
                "errors": ["ملف الصوت غير موجود"],
                "warnings": [],
                "details": {},
                "student_feedback": "عذراً، لم يتم العثور على ملف الصوت. تأكد من رفع الملف بشكل صحيح."
            }
        
        try:
            # Charger l'audio
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Analyses de qualité
            quality_checks = {
                "duration": self._check_duration(audio, sr),
                "silence": self._check_silence_ratio(audio),
                "noise": self._check_noise_level(audio),
                "clipping": self._check_clipping(audio),
                "volume": self._check_volume_level(audio)
            }
            
            # Déterminer si l'audio est valide
            all_valid = all(check["valid"] for check in quality_checks.values())
            
            # Messages d'erreur prioritaires
            errors = []
            warnings = []
            
            for check_name, check_result in quality_checks.items():
                if not check_result["valid"]:
                    if check_result.get("severity") == "error":
                        errors.append(check_result["message"])
                    else:
                        warnings.append(check_result["message"])
                elif check_result.get("severity") == "warning":
                    warnings.append(check_result["message"])
            
            # Générer le feedback pour l'étudiant
            student_feedback = self._generate_student_feedback(quality_checks, all_valid and len(errors) == 0)
            
            return {
                "valid": all_valid and len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "details": quality_checks,
                "audio_info": {
                    "duration": len(audio) / sr,
                    "sample_rate": sr,
                    "channels": 1,
                    "max_amplitude": float(np.max(np.abs(audio)))
                },
                "student_feedback": student_feedback
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"خطأ في تحليل الصوت: {str(e)}",
                "errors": [f"خطأ في تحليل الصوت: {str(e)}"],
                "warnings": [],
                "details": {},
                "student_feedback": f"حدث خطأ أثناء تحليل ملف الصوت. يرجى المحاولة مرة أخرى أو التأكد من صحة الملف.\nتفاصيل الخطأ: {str(e)}"
            }
    
    def _generate_student_feedback(self, quality_checks, is_valid):
        """
        Génère un feedback complet en arabe pour l'étudiant
        """
        feedback_parts = []
        
        if is_valid:
            feedback_parts.append("🎉 ممتاز! جودة التسجيل الصوتي جيدة جداً.")
            
            # Ajouter des conseils même si tout va bien
            good_aspects = []
            for check_name, result in quality_checks.items():
                if result["valid"] and result.get("severity") != "warning":
                    if check_name == "duration":
                        good_aspects.append(f"• مدة التسجيل مناسبة ({result['value']:.1f} ثانية)")
                    elif check_name == "volume":
                        good_aspects.append(f"• مستوى الصوت جيد")
                    elif check_name == "noise":
                        good_aspects.append(f"• جودة الصوت واضحة (نسبة الإشارة للضوضاء: {result['value']:.1f} ديسيبل)")
                    elif check_name == "silence":
                        good_aspects.append(f"• نسبة الصمت مقبولة ({result['value']*100:.1f}%)")
                    elif check_name == "clipping":
                        good_aspects.append("• لا توجد مشاكل في التشبع الصوتي")
            
            if good_aspects:
                feedback_parts.append("\n✅ الجوانب الإيجابية:")
                feedback_parts.extend(good_aspects)
        else:
            feedback_parts.append("⚠️ يحتاج التسجيل الصوتي إلى تحسين. إليك النصائح لحل المشاكل:")
        
        # Analyser chaque problème et donner des conseils spécifiques
        for check_name, result in quality_checks.items():
            if not result["valid"] or result.get("severity") == "warning":
                feedback_parts.append(f"\n🔧 {self._get_problem_solution(check_name, result)}")
        
        # Ajouter des conseils généraux
        if not is_valid:
            feedback_parts.append(self._get_general_tips())
        
        return "\n".join(feedback_parts)
    
    def _get_problem_solution(self, check_name, result):
        """
        Retourne des solutions spécifiques pour chaque type de problème
        """
        if check_name == "duration":
            if result["value"] < self.min_duration:
                return f"""مشكلة: التسجيل قصير جداً ({result['value']:.1f} ثانية)
الحل:
• تحدث لمدة أطول (على الأقل {self.min_duration} ثانية)
• تأكد من اكتمال إجابتك قبل إنهاء التسجيل
• راجع السؤال وأعط إجابة أكثر تفصيلاً"""
            else:
                return f"""مشكلة: التسجيل طويل جداً ({result['value']:.1f} ثانية)
الحل:
• اختصر إجابتك (الحد الأقصى {self.max_duration} ثانية)
• ركز على النقاط الأساسية
• تجنب التكرار غير المفيد"""
        
        elif check_name == "silence":
            if result.get("severity") == "error":
                return f"""مشكلة: كثرة الصمت في التسجيل ({result['value']*100:.1f}%)
الحل:
• تحدث بصوت أعلى وأوضح
• تأكد من قرب الميكروفون من فمك
• تجنب التوقف الطويل أثناء الكلام
• تدرب على إجابتك قبل التسجيل"""
            else:
                return f"""تحسين مقترح: يوجد صمت كثير ({result['value']*100:.1f}%)
النصيحة:
• حاول التحدث بشكل أكثر استمرارية
• قلل من فترات التوقف
• فكر في إجابتك قبل بدء التسجيل"""
        
        elif check_name == "noise":
            if result.get("severity") == "error":
                return f"""مشكلة: ضوضاء عالية في الخلفية (نسبة الإشارة للضوضاء: {result['value']:.1f} ديسيبل)
الحل:
• سجل في مكان هادئ
• أغلق النوافذ والأبواب
• أطفئ المكيف والمروحة إن أمكن
• ابتعد عن مصادر الضوضاء (تلفاز، راديو، إلخ)
• استخدم سماعة رأس بميكروفون إن توفرت"""
            else:
                return f"""تحسين مقترح: يوجد بعض الضوضاء (نسبة الإشارة للضوضاء: {result['value']:.1f} ديسيبل)
النصيحة:
• حاول تقليل الضوضاء المحيطة
• سجل في وقت أكثر هدوءاً"""
        
        elif check_name == "clipping":
            if result.get("severity") == "error":
                return f"""مشكلة: تشبع في الصوت ({result['value']*100:.2f}%)
الحل:
• اخفض مستوى الميكروفون
• ابتعد قليلاً عن الميكروفون
• تحدث بصوت أهدأ قليلاً
• تجنب الصراخ أو رفع الصوت كثيراً"""
            else:
                return f"""تحسين مقترح: تشبع طفيف في الصوت ({result['value']*100:.2f}%)
النصيحة:
• انتبه لمستوى الصوت
• تجنب التحدث بصوت عالٍ جداً"""
        
        elif check_name == "volume":
            if result["value"] < 0.01:
                return f"""مشكلة: الصوت منخفض جداً (مستوى RMS: {result['value']:.3f})
الحل:
• تحدث بصوت أعلى
• اقترب من الميكروفون
• تحقق من إعدادات الميكروفون
• تأكد من عمل الميكروفون بشكل صحيح"""
            elif result["value"] > 0.5:
                return f"""مشكلة: الصوت عالٍ جداً (مستوى RMS: {result['value']:.3f})
الحل:
• تحدث بصوت أهدأ
• ابتعد عن الميكروفون
• اخفض مستوى الميكروفون في إعدادات الجهاز"""
            else:
                return f"""تحسين مقترح: الصوت منخفض قليلاً (مستوى RMS: {result['value']:.3f})
النصيحة:
• حاول التحدث بصوت أوضح قليلاً
• تأكد من استقرار المسافة من الميكروفون"""
        
        return "مشكلة غير محددة"
    
    def _get_general_tips(self):
        """
        نصائح عامة لتحسين جودة التسجيل
        """
        return """\n
💡 نصائح عامة لتحسين التسجيل:
• اختر مكاناً هادئاً للتسجيل
• تأكد من استقرار اتصال الإنترنت
• استخدم سماعة رأس بميكروفون إن أمكن
• فكر في إجابتك قبل بدء التسجيل
• تحدث بوضوح ولا تتعجل
• اجعل الجهاز على مسافة مناسبة (15-20 سم من فمك)
• تجنب الحركة أثناء التسجيل لتجنب الضوضاء

🔄 بعد إصلاح المشاكل، أعد تسجيل الإجابة للحصول على أفضل نتيجة."""
    
    def _check_duration(self, audio, sample_rate):
        """Vérifie la durée de l'audio"""
        duration = len(audio) / sample_rate
        
        if duration < self.min_duration:
            return {
                "valid": False,
                "severity": "error",
                "message": f"لتسجيل قصير جداً ({duration:.1f} ثانية). الحد الأدنى المطلوب: {self.min_duration} ثانية",
                "value": duration
            }
        elif duration > self.max_duration:
            return {
                "valid": False,
                "severity": "error", 
                "message": f"التسجيل طويل جداً ({duration:.1f} ثانية). الحد الأقصى المسموح: {self.max_duration} ثانية",
                "value": duration
            }
        else:
            return {
                "valid": True,
                "message": f"مدة التسجيل مناسبة ({duration:.1f} ثانية)",
                "value": duration
            }
    
    def _check_silence_ratio(self, audio):
        """Vérifie le ratio de silence dans l'audio"""
        # Calculer l'énergie RMS par frame
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Compter les frames silencieuses
        silent_frames = np.sum(rms < self.silence_threshold)
        total_frames = len(rms)
        silence_ratio = silent_frames / total_frames if total_frames > 0 else 1.0
        
        if silence_ratio > self.max_silence_ratio:
            return {
                "valid": False,
                "severity": "error",
                "message": f"كثرة الصمت في التسجيل ({silence_ratio*100:.1f}%). يرجى التحدث بصوت أعلى وأوضح.",
                "value": silence_ratio
            }
        elif silence_ratio > 0.4:
            return {
                "valid": True,
                "severity": "warning",
                "message": f"يوجد صمت كثير في التسجيل ({silence_ratio*100:.1f}%). حاول التحدث بشكل أكثر استمرارية.",
                "value": silence_ratio
            }
        else:
            return {
                "valid": True,
                "message": f"نسبة الصمت مقبولة ({silence_ratio*100:.1f}%)",
                "value": silence_ratio
            }
    
    def _check_noise_level(self, audio):
        """Vérifie le niveau de bruit dans l'audio"""
        # Estimer le bruit de fond (premières et dernières 0.5 secondes)
        sr = 16000  # Assumé
        noise_duration = int(0.5 * sr)
        
        if len(audio) > 2 * noise_duration:
            noise_start = audio[:noise_duration]
            noise_end = audio[-noise_duration:]
            noise_estimate = np.concatenate([noise_start, noise_end])
        else:
            # Si l'audio est trop court, utiliser tout l'audio pour estimer le bruit
            noise_estimate = audio
        
        # Calculer l'énergie du signal et du bruit
        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise_estimate**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        if snr < self.min_snr:
            return {
                "valid": False,
                "severity": "error",
                "message": f"ضوضاء عالية في الخلفية (نسبة الإشارة للضوضاء: {snr:.1f} ديسيبل). يرجى التسجيل في بيئة أكثر هدوءاً.",
                "value": snr
            }
        elif snr < 15:
            return {
                "valid": True,
                "severity": "warning",
                "message": f"يوجد ضوضاء في الخلفية (نسبة الإشارة للضوضاء: {snr:.1f} ديسيبل). حاول تقليل الضوضاء المحيطة.",
                "value": snr
            }
        else:
            return {
                "valid": True,
                "message": f"مستوى الضوضاء مقبول (نسبة الإشارة للضوضاء: {snr:.1f} ديسيبل)",
                "value": snr
            }
    
    def _check_clipping(self, audio):
        """Vérifie la saturation/clipping de l'audio"""
        # Détecter les échantillons saturés
        threshold = 0.95
        clipped_samples = np.sum(np.abs(audio) > threshold)
        clipping_ratio = clipped_samples / len(audio)
        
        if clipping_ratio > 0.01:  # Plus de 1% d'échantillons saturés
            return {
                "valid": False,
                "severity": "error",
                "message": f"تشبع في الصوت ({clipping_ratio*100:.2f}%). يرجى تخفيض مستوى الميكروفون.",
                "value": clipping_ratio
            }
        elif clipping_ratio > 0.001:  # Plus de 0.1% d'échantillons saturés
            return {
                "valid": True,
                "severity": "warning",
                "message": f"تشبع طفيف في الصوت ({clipping_ratio*100:.2f}%). انتبه لمستوى الصوت.",
                "value": clipping_ratio
            }
        else:
            return {
                "valid": True,
                "message": "لا يوجد تشبع في الصوت",
                "value": clipping_ratio
            }
    
    def _check_volume_level(self, audio):
        """Vérifie le niveau de volume de l'audio"""
        rms = np.sqrt(np.mean(audio**2))
        max_amplitude = np.max(np.abs(audio))
        
        # Seuils de volume
        min_rms = 0.01
        max_rms = 0.5
        
        if rms < min_rms:
            return {
                "valid": False,
                "severity": "error",
                "message": f"الصوت منخفض جداً (مستوى RMS: {rms:.3f}). يرجى التحدث بصوت أعلى أو الاقتراب من الميكروفون.",
                "value": rms
            }
        elif rms > max_rms:
            return {
                "valid": False,
                "severity": "error",
                "message": f"الصوت عالٍ جداً (مستوى RMS: {rms:.3f}). يرجى التحدث بصوت أهدأ أو الابتعاد عن الميكروفون.",
                "value": rms
            }
        elif rms < 0.02:
            return {
                "valid": True,
                "severity": "warning",
                "message": f"الصوت منخفض قليلاً (مستوى RMS: {rms:.3f}). حاول التحدث بصوت أوضح قليلاً.",
                "value": rms
            }
        else:
            return {
                "valid": True,
                "message": f"مستوى الصوت مناسب (مستوى RMS: {rms:.3f})",
                "value": rms
            }
