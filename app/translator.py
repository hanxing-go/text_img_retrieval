from __future__ import annotations
from pathlib import Path
from typing import Optional


def _contains_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            return True
    return False


class OfflineTranslator:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        model_path: Optional[str] = None,
        only_when_cjk: bool = True,
    ) -> None:
        try:
            import argostranslate.package as argos_package
            import argostranslate.translate as argos_translate
        except Exception as exc:  # pragma: no cover - import error
            raise RuntimeError(
                "Argos Translate is not installed. Add 'argostranslate' to requirements and install a model."
            ) from exc

        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise RuntimeError(f"Translation model not found: {model_path}")
            argos_package.install_from_path(str(path))

        translation = argos_translate.get_translation_from_codes(source_lang, target_lang)
        if translation is None:
            raise RuntimeError(
                "Translation model not installed for the given language pair. "
                "Install a local Argos model or set TIR_TRANSLATE_MODEL_PATH."
            )

        self._translation = translation
        self._only_when_cjk = only_when_cjk

    def translate_text(self, text: str) -> str:
        if not text:
            return text
        return self._translation.translate(text)

    def translate_if_needed(self, text: str) -> str:
        if not text:
            return text
        if self._only_when_cjk and not _contains_cjk(text):
            return text
        return self.translate_text(text)
