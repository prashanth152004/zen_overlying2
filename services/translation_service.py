import time
import requests

# Delimiter used to batch multiple segments in one translation call
_BATCH_DELIMITER = " ||| "
# Number of consecutive segments to batch for context continuity
_BATCH_SIZE = 5


class TranslationEngine:
    def __init__(self, model="deep_translator", api_key=None):
        self.model = model
        self.api_key = api_key

    # ─────────────────────────────────────────────────────────────
    # SARVAM AI TRANSLATION
    # ─────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, max_chars: int = 480) -> list:
        """
        Split text into chunks of at most max_chars, preferring sentence/word boundaries.
        Sarvam AI has a ~500 char per-request limit.
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        # Try to split on sentence boundaries first
        sentences = text.replace("। ", "।\n").replace(". ", ".\n").split("\n")
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # If a single sentence exceeds the limit, split it further by words
            if len(sentence) > max_chars:
                words = sentence.split(" ")
                for word in words:
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

    def _translate_with_sarvam(self, text: str, target_lang: str) -> str:
        if not self.api_key:
            print("[TranslationEngine] No Sarvam API Key — falling back to deep-translator.")
            return self._translate_with_deep_translator(text, target_lang)

        url = "https://api.sarvam.ai/translate"

        # Map internal lang codes to Sarvam BCP-47 codes
        lang_map = {
            "hi": "hi-IN",
            "en": "en-IN",
            "kn": "kn-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "ml": "ml-IN",
            "bn": "bn-IN",
            "gu": "gu-IN",
            "mr": "mr-IN",
            "pa": "pa-IN",
            "od": "or-IN",
        }
        sarvam_target_lang = lang_map.get(target_lang, f"{target_lang}-IN")

        # Use code-mixed mode for Hindi (better naturalness), formal for others
        translation_mode = "code-mixed" if target_lang == "hi" else "formal"

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }

        # Chunk text to stay within Sarvam's 500-char per-request limit
        chunks = self._chunk_text(text, max_chars=480)
        translated_chunks = []

        for chunk in chunks:
            payload = {
                "input": chunk,
                "source_language_code": "en-IN",
                "target_language_code": sarvam_target_lang,
                "speaker_gender": "Male",
                "mode": translation_mode,
                "model": "mayura:v1",
                "enable_preprocessing": True,
            }

            # Retry up to 3 times with exponential backoff
            translated_chunk = chunk  # fallback: keep original
            for attempt in range(3):
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=15)
                    if response.status_code == 200:
                        result = response.json()
                        translated_chunk = result.get("translated_text", chunk)
                        print(f"[TranslationEngine] Sarvam ✓ chunk ({len(chunk)}c): '{chunk[:30]}' → '{translated_chunk[:30]}'")
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
    # DEEP TRANSLATOR  — single segment
    # ─────────────────────────────────────────────────────────────

    def _translate_with_deep_translator(self, text: str, target_lang: str) -> str:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target=target_lang)
        try:
            translated_text = translator.translate(text)
            time.sleep(0.1)
            return translated_text if translated_text else text
        except Exception as e:
            print(f"[TranslationEngine] deep-translator error: {e}")
            return text

    def _translate_batch_with_deep_translator(self, texts: list, target_lang: str) -> list:
        """
        Translate a batch of consecutive segments joined by a delimiter so the
        AI sees sentence context across segment boundaries, producing more fluent output.
        Falls back to per-segment translation if batch parsing fails.
        """
        from deep_translator import GoogleTranslator
        joined = _BATCH_DELIMITER.join(texts)
        translator = GoogleTranslator(source="en", target=target_lang)
        try:
            translated_joined = translator.translate(joined)
            time.sleep(0.15)
            if not translated_joined:
                raise ValueError("Empty response from Google Translate")

            # Split result back by delimiter
            parts = translated_joined.split("|||")
            # Clean whitespace
            parts = [p.strip() for p in parts]

            if len(parts) == len(texts):
                return parts

            # Mismatch: fall back to per-segment so we never lose a segment
            print(f"[TranslationEngine] Batch split mismatch ({len(parts)} vs {len(texts)}). "
                  f"Falling back to per-segment for this batch.")
            return [self._translate_with_deep_translator(t, target_lang) for t in texts]

        except Exception as e:
            print(f"[TranslationEngine] Batch translation error: {e}. Falling back per-segment.")
            return [self._translate_with_deep_translator(t, target_lang) for t in texts]

    # ─────────────────────────────────────────────────────────────
    # POST-PROCESSING
    # ─────────────────────────────────────────────────────────────

    def _clean_translated_text(self, text: str) -> str:
        """Remove stray delimiters, fix double spaces, strip leading/trailing whitespace."""
        text = text.replace("|||", "").replace("  ", " ").strip()
        return text

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def translate_transcript(self, transcript: list, source="en", target="en") -> list:
        """Translate text segments. If target is 'en', pass-through (Whisper already translated)."""
        if target == "en":
            print(f"[TranslationEngine] Pass-through ({len(transcript)} segments, already in English).")
            return transcript

        print(f"[TranslationEngine] Translating {len(transcript)} segments → '{target}' "
              f"using '{self.model}' (batch_size={_BATCH_SIZE})...")

        translated_transcript = []

        # ── Deep Translator: batch for context continuity ──────────────────────
        if self.model != "sarvam_ai":
            # Process in batches of _BATCH_SIZE
            i = 0
            while i < len(transcript):
                batch = transcript[i : i + _BATCH_SIZE]
                batch_texts = [seg["text"] for seg in batch]

                print(f"[TranslationEngine] Batch translating segments {i}–{i+len(batch)-1} "
                      f"({len(batch)} segs) → '{target}'")
                translated_texts = self._translate_batch_with_deep_translator(batch_texts, target)

                for j, seg in enumerate(batch):
                    clean_text = self._clean_translated_text(translated_texts[j])
                    translated_transcript.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker_id": seg.get("speaker_id", "SPEAKER_01"),
                        "text": clean_text
                    })
                i += _BATCH_SIZE

        # ── Sarvam AI: per-segment (its chunking already handles context) ──────
        else:
            for segment in transcript:
                original_text = segment["text"]
                translated_text = self._translate_with_sarvam(original_text, target)
                clean_text = self._clean_translated_text(translated_text)
                translated_transcript.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker_id": segment.get("speaker_id", "SPEAKER_01"),
                    "text": clean_text
                })

        return translated_transcript
