import time
import random
import requests

# Delimiter used to batch multiple segments in one translation call.
# Google Translate preserves paragraph breaks perfectly, allowing context transfer.
_BATCH_DELIMITER = "\n\n"
# Aggressive Context Mode: Batching 12 consecutive sentences translates virtually an entire scene natively
_BATCH_SIZE = 12

# Languages where deep-translator (Google) supports the target
# (Google Translate supports kn = Kannada)
_DEEP_TRANSLATOR_SUPPORTED = {"en", "hi", "kn"}

_SARVAM_LANG_MAP = {
    "hi": "hi-IN", "en": "en-IN", "kn": "kn-IN",
    "ta": "ta-IN", "te": "te-IN", "ml": "ml-IN",
    "bn": "bn-IN", "gu": "gu-IN", "mr": "mr-IN",
    "pa": "pa-IN", "od": "or-IN",
}


class TranslationEngine:
    def __init__(self, model="deep_translator", api_key=None):
        self.model = model
        self.api_key = api_key

    # ─────────────────────────────────────────────────────────────
    # LITERAL PROTECTION ENGINE
    # ─────────────────────────────────────────────────────────────

    def _protect_literals(self, text: str) -> tuple:
        """
        Shields numbers, URLs, and website domains from being rewritten by the AI.
        Replaces them with indestructible placeholders: __LIT0__, __LIT1__...
        Returns (protected_text, protected_dict)
        """
        import re
        protected = {}
        counter = [0]

        def make_key():
            key = f"__LIT{counter[0]}__"
            counter[0] += 1
            return key

        result = text

        # 1. Protect full URLs — e.g. https://passportseva.gov.in
        def protect_url(m):
            key = make_key()
            protected[key] = m.group(0)
            return key
        result = re.sub(r'https?://\S+', protect_url, result)

        # 2. Protect website domains — e.g. passportseva.com, parivansewa.com
        result = re.sub(r'\b[a-zA-Z0-9.-]+\.(?:com|in|gov|org|net|co|io)\b', protect_url, result)

        # 3. Protect standalone numbers, decimals, percentages — e.g. 18, 4, 100%
        result = re.sub(r'\b\d+(?:[.,]\d+)*\s*%?\b', protect_url, result)

        return result, protected

    def _restore_literals(self, text: str, protected: dict) -> str:
        """Restores all protected literals back into the translated text exactly as they were."""
        result = text
        for key, value in protected.items():
            result = result.replace(key, value)
        return result

    # ─────────────────────────────────────────────────────────────
    # TEXT CHUNKER
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

    # ─────────────────────────────────────────────────────────────
    # SARVAM AI TRANSLATION — single chunk
    # ─────────────────────────────────────────────────────────────

    def _translate_with_sarvam(self, text: str, source_lang: str, target_lang: str,
                                speaker_gender: str = "Male") -> str:
        """Translate a single text chunk with Sarvam AI — with literal number/URL protection."""
        if not self.api_key:
            print("[TranslationEngine] No Sarvam API Key — falling back to deep-translator.")
            return self._translate_with_deep_translator(text, target_lang)

        # ── LITERAL PROTECTION: shield numbers/URLs/domains before sending to AI ──
        protected_text, protected = self._protect_literals(text)

        url = "https://api.sarvam.ai/translate"
        sarvam_source = _SARVAM_LANG_MAP.get(source_lang, f"{source_lang}-IN")
        sarvam_target = _SARVAM_LANG_MAP.get(target_lang, f"{target_lang}-IN")

        # Strict Formal Mode: Indian regional languages use intense 'Code-Mixing' (Kanglish/Hinglish)
        # in 'modern-colloquial' mode. To guarantee 100% pure Kannada/Hindi vocabulary for subtitles
        # and dubbing, we MUST lock the translation mode to 'formal' regardless of sentence length.
        translation_mode = "formal"

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.api_key
        }
        chunks = self._chunk_text(protected_text, max_chars=480)
        translated_chunks = []

        for chunk in chunks:
            payload = {
                "input": chunk,
                "source_language_code": sarvam_source,
                "target_language_code": sarvam_target,
                "speaker_gender": speaker_gender,
                "mode": translation_mode,
                "model": "sarvam-translate:v1",
                "enable_preprocessing": False,  # CRITICAL: prevents numbers/names from being phonetically rewritten
            }
            translated_chunk = chunk
            for attempt in range(4):
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=20)
                    if response.status_code == 200:
                        result = response.json()
                        translated_chunk = result.get("translated_text", chunk)
                        print(f"[TranslationEngine] Sarvam ✓ [{translation_mode}] "
                              f"({len(chunk)}c): '{chunk[:30]}' → '{translated_chunk[:30]}'")
                        break
                    elif response.status_code == 429:
                        # Exponential backoff with jitter to prevent thundering-herd
                        wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                        print(f"[TranslationEngine] Sarvam rate limit (attempt {attempt+1}/4). "
                              f"Waiting {wait:.1f}s...")
                        time.sleep(wait)
                    else:
                        print(f"[TranslationEngine] Sarvam API Error {response.status_code}: "
                              f"{response.text[:200]}")
                        break
                except requests.exceptions.Timeout:
                    wait = (2 ** attempt) + random.uniform(0.2, 0.8)
                    print(f"[TranslationEngine] Sarvam timeout (attempt {attempt+1}/4). "
                          f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                except Exception as e:
                    print(f"[TranslationEngine] Sarvam exception: {e}")
                    break
            translated_chunks.append(translated_chunk)

        raw_result = " ".join(translated_chunks)
        # ── RESTORE all protected numbers, URLs, domains back exactly ──
        return self._restore_literals(raw_result, protected)

    # ─────────────────────────────────────────────────────────────
    # SARVAM AI — Context-Batch mode
    # ─────────────────────────────────────────────────────────────

    def _translate_batch_with_sarvam(self, texts: list, source_lang: str, target_lang: str,
                                     speaker_genders: list = None) -> list:
        """
        Context-Batch mode for Sarvam AI.
        Uses indestructible alpha-anchor tags [S0], [S1]... as delimiters.
        These tags can NEVER collide with numbers/words inside the translated sentences.
        Falls back to per-segment if batch parse fails.
        """
        import re

        if not self.api_key:
            return [self._translate_with_deep_translator(t, target_lang) for t in texts]

        if speaker_genders is None:
            speaker_genders = ["Male"] * len(texts)

        # CRITICAL FIX: Use alpha-anchor tags [S0], [S1]... NOT numbers (1., 2.)
        # Numeric delimiters collide with actual numbers inside sentences (e.g. "18 years old")
        # Alpha-anchor tags are 100% collision-proof — no translated language uses [S0] syntax.
        tag_sep = "|||"  # unique separator that no language ever generates naturally
        payload_lines = [f"[S{j}] {t}" for j, t in enumerate(texts)]
        joined_context = f"\n".join(payload_lines)

        # Use majority gender in the batch
        male_count = sum(1 for g in speaker_genders if str(g).lower() == "male")
        batch_gender = "Male" if male_count >= len(speaker_genders) / 2 else "Female"

        try:
            translated_joined = self._translate_with_sarvam(
                joined_context, source_lang, target_lang, batch_gender
            )

            # Extract segments by splitting on [S<n>] anchor tags
            # Regex looks ONLY for [S0], [S1]... and strips nothing else
            raw_parts = re.split(r'\[S\d+\]\s*', translated_joined)
            parts = [p.strip() for p in raw_parts if p.strip()]

            if len(parts) == len(texts):
                print(f"[TranslationEngine] Sarvam Batch Context SUCCESS ({len(texts)} segs)")
                return parts

            print(f"[TranslationEngine] Sarvam Batch mismatch ({len(parts)} vs {len(texts)}). "
                  f"Falling back per-segment.")
            return [
                self._translate_with_sarvam(t, source_lang, target_lang, g)
                for t, g in zip(texts, speaker_genders)
            ]

        except Exception as e:
            print(f"[TranslationEngine] Sarvam Batch error: {e}. Falling back per-segment.")
            return [
                self._translate_with_sarvam(t, source_lang, target_lang, g)
                for t, g in zip(texts, speaker_genders)
            ]

    # ─────────────────────────────────────────────────────────────
    # DEEP TRANSLATOR — single segment
    # ─────────────────────────────────────────────────────────────

    def _translate_with_deep_translator(self, text: str, target_lang: str) -> str:
        from deep_translator import GoogleTranslator
        # ── LITERAL PROTECTION ──
        protected_text, protected = self._protect_literals(text)
        translator = GoogleTranslator(source="auto", target=target_lang)
        try:
            translated = translator.translate(protected_text)
            time.sleep(0.1)
            result = translated if translated else protected_text
            # Restore numbers/URLs exactly as original
            return self._restore_literals(result, protected)
        except Exception as e:
            print(f"[TranslationEngine] deep-translator error: {e}")
            return text  # on complete failure, return original text unchanged

    def _translate_batch_with_deep_translator(self, texts: list, target_lang: str) -> list:
        """
        Context-Batch translation using Google Translate.
        Uses indestructible [S0], [S1]... alpha-anchor tags as delimiters.
        These NEVER collide with actual numbers/words inside sentences.
        """
        from deep_translator import GoogleTranslator
        import re

        # CRITICAL FIX: Alpha-anchor tags instead of numeric prefixes
        # Old: "1. If you are 18 years old" → regex could strip the "18"
        # New: "[S0] If you are 18 years old" → regex only strips [S0], preserves 18
        payload_lines = [f"[S{j}] {t}" for j, t in enumerate(texts)]
        joined_context = "\n".join(payload_lines)
        translator = GoogleTranslator(source="auto", target=target_lang)

        try:
            translated_joined = translator.translate(joined_context)
            time.sleep(0.15)
            if not translated_joined:
                raise ValueError("Empty response")

            # Extract segments by splitting on [S<n>] anchor tags
            raw_parts = re.split(r'\[S\d+\]\s*', translated_joined)
            parts = [p.strip() for p in raw_parts if p.strip()]

            if len(parts) == len(texts):
                print(f"[TranslationEngine] Batch Context Translation SUCCESS ({len(texts)} sentences)")
                return parts

            print(f"[TranslationEngine] Context Batch split mismatch ({len(parts)} vs {len(texts)}). "
                  f"Array corrupted by Translator. Falling back per-segment.")
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

    def translate_transcript(self, transcript: list, source="en", target="en",
                             speaker_gender: str = "Male") -> list:
        """
        Translate transcript segments from source → target.
        Pass-through if source == target (no translation needed).
        Supports: en, hi, kn.

        Args:
            speaker_gender: 'Male' or 'Female' — used by Sarvam AI for gender-appropriate phrasing.
        """
        if source == target:
            print(f"[TranslationEngine] Pass-through — source == target == '{target}' "
                  f"({len(transcript)} segs).")
            return transcript

        print(f"[TranslationEngine] Translating {len(transcript)} segments "
              f"'{source}' → '{target}' using '{self.model}' "
              f"(batch_size={_BATCH_SIZE}, gender={speaker_gender})...")

        translated_transcript = []

        # ── Deep Translator: context-batched numbered list ────────────────
        if self.model != "sarvam_ai":
            i = 0
            while i < len(transcript):
                batch = transcript[i: i + _BATCH_SIZE]
                batch_texts = [seg["text"] for seg in batch]
                print(f"[TranslationEngine] Batch {i}–{i+len(batch)-1} "
                      f"({len(batch)} segs) → '{target}'")
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

        # ── Sarvam AI: Context-Batch with gender + dynamic source language ─
        else:
            i = 0
            while i < len(transcript):
                batch = transcript[i: i + _BATCH_SIZE]
                batch_texts = [seg["text"] for seg in batch]
                batch_genders = [seg.get("speaker_gender", speaker_gender) for seg in batch]

                print(f"[TranslationEngine] Sarvam Batch {i}–{i+len(batch)-1} "
                      f"({len(batch)} segs) '{source}' → '{target}'")

                translated_texts = self._translate_batch_with_sarvam(
                    batch_texts, source_lang=source, target_lang=target,
                    speaker_genders=batch_genders
                )

                for j, seg in enumerate(batch):
                    translated_transcript.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker_id": seg.get("speaker_id", "SPEAKER_01"),
                        "speaker_gender": batch_genders[j],
                        "text": self._clean_translated_text(translated_texts[j])
                    })
                i += _BATCH_SIZE

        return translated_transcript
