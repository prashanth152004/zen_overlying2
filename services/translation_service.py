import time
import requests

# Delimiter used to batch multiple segments in one translation call.
# Google Translate preserves paragraph breaks perfectly, allowing context transfer.
_BATCH_DELIMITER = "\n\n"
# Number of consecutive segments to batch for context continuity
_BATCH_SIZE = 5

# Languages where deep-translator (Google) supports the target
# (Google Translate supports kn = Kannada)
_DEEP_TRANSLATOR_SUPPORTED = {"en", "hi", "kn"}


class TranslationEngine:
    def __init__(self, model="deep_translator", api_key=None):
        self.model = model
        self.api_key = api_key

    # ─────────────────────────────────────────────────────────────
    # SARVAM AI TRANSLATION
    # ─────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_chars: int = 480) -> list:
        """Split text into chunks ≤ max_chars preferring sentence/word boundaries."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        sentences = text.replace("। ", "।\n").replace(". ", ".\n").split("\n")
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                for word in sentence.split():
                    if len(current_chunk) + len(word) + 1 > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        current_chunk = f"{current_chunk} {word}".strip()
            elif len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}".strip()
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _translate_with_sarvam(self, text: str, target_lang: str, speaker_gender: str = "Male") -> str:
        if not self.api_key:
            print("[TranslationEngine] No Sarvam API Key — falling back to deep-translator.")
            return self._translate_with_deep_translator(text, target_lang)

        url = "https://api.sarvam.ai/translate"
        lang_map = {
            "hi": "hi-IN", "en": "en-IN", "kn": "kn-IN",
            "ta": "ta-IN", "te": "te-IN", "ml": "ml-IN",
            "bn": "bn-IN", "gu": "gu-IN", "mr": "mr-IN",
            "pa": "pa-IN", "od": "or-IN",
        }
        sarvam_target_lang = lang_map.get(target_lang, f"{target_lang}-IN")
        # Ensure we always get full native script (Devanagari/Kannada) rather than code-mixed.
        translation_mode = "formal"

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }
        chunks = self._chunk_text(text, max_chars=480)
        translated_chunks = []

        for chunk in chunks:
            payload = {
                "input": chunk,
                "source_language_code": "en-IN",
                "target_language_code": sarvam_target_lang,
                "speaker_gender": speaker_gender,
                "mode": translation_mode,
                "model": "mayura:v1",
                "enable_preprocessing": True,
            }
            translated_chunk = chunk
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=15)
                    if response.status_code == 200:
                        result = response.json()
                        translated_chunk = result.get("translated_text", chunk)
                        print(f"[TranslationEngine] Sarvam ✓ ({len(chunk)}c): '{chunk[:30]}' → '{translated_chunk[:30]}'")
                        break
                    elif response.status_code == 429:
                        wait = 2 ** attempt
                        print(f"[TranslationEngine] Sarvam rate limit (attempt {attempt+1}/3). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"[TranslationEngine] Sarvam API Error {response.status_code}: {response.text[:200]}")
                        break
                except requests.exceptions.Timeout:
                    print(f"[TranslationEngine] Sarvam timeout (attempt {attempt+1}/3).")
                    time.sleep(2 ** attempt)
                except Exception as e:
                    print(f"[TranslationEngine] Sarvam exception: {e}")
                    break
            translated_chunks.append(translated_chunk)

        return " ".join(translated_chunks)

    # ─────────────────────────────────────────────────────────────
    # DEEP TRANSLATOR — single segment
    # ─────────────────────────────────────────────────────────────

    def _translate_with_deep_translator(self, text: str, target_lang: str) -> str:
        from deep_translator import GoogleTranslator
        # Google Translate uses 'kn' for Kannada, 'hi' for Hindi, 'en' for English
        translator = GoogleTranslator(source="en", target=target_lang)
        try:
            translated = translator.translate(text)
            time.sleep(0.1)
            return translated if translated else text
        except Exception as e:
            print(f"[TranslationEngine] deep-translator error: {e}")
            return text

    def _translate_batch_with_deep_translator(self, texts: list, target_lang: str) -> list:
        """
        The 'Context-Batch' Algorithm.
        Translates a batch of consecutive sentences as a single numbered list.
        This fundamentally tricks Google Translate into understanding conversational 
        grammar across sentences, astronomically improving Hindi/Kannada accuracy.
        """
        from deep_translator import GoogleTranslator
        import re
        
        # Wrap the sentences in an indestructible numbered format
        payload_lines = []
        for j, t in enumerate(texts):
            payload_lines.append(f"{j+1}. {t}")
            
        joined_context = "\n".join(payload_lines)
        translator = GoogleTranslator(source="en", target=target_lang)
        
        try:
            translated_joined = translator.translate(joined_context)
            time.sleep(0.15)
            if not translated_joined:
                raise ValueError("Empty response")
                
            # Safely unpack the translated paragraph back into a clean array
            parts = []
            for line in translated_joined.split('\n'):
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                # Strip the leading numerals (e.g. "1. " or Hindi "१. " or Kannada "೧. ")
                # Python's \d natively matches Unicode Indic numeric characters perfectly!
                clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                parts.append(clean_line)
            
            if len(parts) == len(texts):
                print(f"[TranslationEngine] Batch Context Translation SUCCESS ({len(texts)} sentences)")
                return parts
                
            print(f"[TranslationEngine] Context Batch split mismatch ({len(parts)} vs {len(texts)}). Array corrupted by Translator. Falling back.")
            return [self._translate_with_deep_translator(t, target_lang) for t in texts]
            
        except Exception as e:
            print(f"[TranslationEngine] Context Batch error: {e}. Falling back per-segment.")
            return [self._translate_with_deep_translator(t, target_lang) for t in texts]

    # ─────────────────────────────────────────────────────────────
    # POST-PROCESSING
    # ─────────────────────────────────────────────────────────────

    def _clean_translated_text(self, text: str) -> str:
        return text.replace("\n", " ").replace("  ", " ").strip()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def translate_transcript(self, transcript: list, source="en", target="en", speaker_gender: str = "Male") -> list:
        """
        Translate transcript segments from source → target.
        Pass-through if source == target (no translation needed).
        Supports: en, hi, kn.

        Args:
            speaker_gender: 'Male' or 'Female' — used by Sarvam AI for gender-appropriate phrasing.
        """
        if source == target:
            print(f"[TranslationEngine] Pass-through — source == target == '{target}' ({len(transcript)} segs).")
            return transcript

        print(f"[TranslationEngine] Translating {len(transcript)} segments "
              f"'{source}' → '{target}' using '{self.model}' (batch_size={_BATCH_SIZE}, gender={speaker_gender})...")

        translated_transcript = []

        # ── Deep Translator: batched for context ─────────────────────────
        if self.model != "sarvam_ai":
            i = 0
            while i < len(transcript):
                batch = transcript[i: i + _BATCH_SIZE]
                batch_texts = [seg["text"] for seg in batch]
                print(f"[TranslationEngine] Batch {i}–{i+len(batch)-1} ({len(batch)} segs) → '{target}'")
                translated_texts = self._translate_batch_with_deep_translator(batch_texts, target)
                for j, seg in enumerate(batch):
                    translated_transcript.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker_id": seg.get("speaker_id", "SPEAKER_01"),
                        "speaker_gender": seg.get("speaker_gender", speaker_gender),
                        "text": self._clean_translated_text(translated_texts[j])
                    })
                i += _BATCH_SIZE

        # ── Sarvam AI: per-segment with gender ────────────────────────────
        else:
            for segment in transcript:
                seg_gender = segment.get("speaker_gender", speaker_gender)
                translated_text = self._translate_with_sarvam(segment["text"], target, seg_gender)
                translated_transcript.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker_id": segment.get("speaker_id", "SPEAKER_01"),
                    "speaker_gender": seg_gender,
                    "text": self._clean_translated_text(translated_text)
                })

        return translated_transcript
