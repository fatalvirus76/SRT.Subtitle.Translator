# english_to_swedish_subtitle_translator_v10.py
# ------------------------------------------------------------
# SRT Translator GUI för PyQt6 + Ollama/OpenAI-kompatibla API:er.
#
# v10:
# - Profiler borttagna.
# - Stöd för upp till 20 Ollama Cloud API-nycklar.
# - Vid HTTP 429 på en Cloud-nyckel testas nästa nyckel direkt.
# - Om alla Cloud-nycklar är rate-limitade väntar programmet och försöker igen.
# - Hakparenteser översätts fortfarande, t.ex. [PANTING] -> [FLÄMTAR].
# ------------------------------------------------------------

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from threading import Lock

import requests
from PyQt6.QtCore import (
    QSettings,
    QStandardPaths,
    QThread,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyleFactory,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


# ==========================================================
# CONSTANTS
# ==========================================================

APP_NAME = "SRTTranslator"

DEFAULT_SERVER_URL = "http://localhost:11434/api"
DEFAULT_CLOUD_URL = "https://ollama.com/api"
DEFAULT_OPENAI_COMPAT_URL = "http://localhost:8000/v1"
DEFAULT_PROVIDER = "local"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_RATE_LIMIT_RPM = 0
DEFAULT_MIN_DELAY_SEC = 0.0

MAX_CLOUD_API_KEYS = 20
MAX_BACKOFF_SEC = 120
RETRY_COUNT = 2
RETRY_DELAY = 5

BATCH_SIZE = 8
CONTEXT_TRANSLATED_BLOCKS = 3

SETTINGS_FILE = "settings.ini"

LANGUAGES = [
    ("Arabic", "ar"),
    ("Chinese (Simplified)", "zh-CN"),
    ("Chinese (Traditional)", "zh-TW"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("German", "de"),
    ("Greek", "el"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Norwegian", "no"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Spanish", "es"),
    ("Swedish", "sv"),
    ("Thai", "th"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
    ("Vietnamese", "vi"),
]
LANGUAGES.sort(key=lambda item: item[0].casefold())

STYLES = [
    "Formal",
    "Natural (recommended)",
    "Simple clear",
]
STYLES.sort(key=str.casefold)

THEMES = {
    "Light": {
        "window_bg": "#F0F0F0",
        "text_color": "#000000",
        "base_bg": "#FFFFFF",
        "button_bg": "#E0E0E0",
        "button_hover": "#D0D0D0",
        "progress_bg": "#E0E0E0",
        "progress_chunk": "#4CAF50",
        "log_bg": "#FFFFFF",
        "log_text": "#000000",
        "border": "#CCCCCC",
    },
    "Dark": {
        "window_bg": "#2D2D2D",
        "text_color": "#E0E0E0",
        "base_bg": "#3C3C3C",
        "button_bg": "#505050",
        "button_hover": "#606060",
        "progress_bg": "#404040",
        "progress_chunk": "#4CAF50",
        "log_bg": "#252525",
        "log_text": "#E0E0E0",
        "border": "#555555",
    },
    "Dracula": {
        "window_bg": "#282A36",
        "text_color": "#F8F8F2",
        "base_bg": "#44475A",
        "button_bg": "#6272A4",
        "button_hover": "#50FA7B",
        "progress_bg": "#44475A",
        "progress_chunk": "#FF79C6",
        "log_bg": "#282A36",
        "log_text": "#F8F8F2",
        "border": "#6272A4",
    },
    "Nord": {
        "window_bg": "#2E3440",
        "text_color": "#D8DEE9",
        "base_bg": "#3B4252",
        "button_bg": "#4C566A",
        "button_hover": "#5E81AC",
        "progress_bg": "#4C566A",
        "progress_chunk": "#A3BE8C",
        "log_bg": "#2E3440",
        "log_text": "#D8DEE9",
        "border": "#4C566A",
    },
}


# ==========================================================
# HELPERS
# ==========================================================

def get_config_dir():
    config_base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )

    if not config_base:
        config_base = os.path.join(os.path.expanduser("~"), ".config")

    config_dir = os.path.join(config_base, APP_NAME)
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def migrate_old_config(config_dir):
    old_script_dir = os.path.dirname(os.path.abspath(__file__))
    old_path = os.path.join(old_script_dir, SETTINGS_FILE)
    new_path = os.path.join(config_dir, SETTINGS_FILE)

    if os.path.exists(old_path) and not os.path.exists(new_path):
        shutil.copy2(old_path, new_path)


def normalize_server_url(url):
    return (url or "").strip().rstrip("/")


def chat_url(server_url):
    return f"{normalize_server_url(server_url)}/chat"


def tags_url(server_url):
    return f"{normalize_server_url(server_url)}/tags"


def openai_chat_completions_url(server_url):
    return f"{normalize_server_url(server_url)}/chat/completions"


def openai_models_url(server_url):
    return f"{normalize_server_url(server_url)}/models"


def language_name_from_code(code):
    for name, lang_code in LANGUAGES:
        if lang_code == code:
            return name

    return code


def normalize_api_keys(value, max_keys=MAX_CLOUD_API_KEYS):
    if value is None:
        return []

    raw_items = []

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        if text.startswith("["):
            try:
                loaded = json.loads(text)
                raw_items = loaded if isinstance(loaded, list) else [text]
            except Exception:
                raw_items = re.split(r"[\r\n,;]+", text)
        else:
            raw_items = re.split(r"[\r\n,;]+", text)

    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [str(value)]

    keys = []
    seen = set()

    for item in raw_items:
        key = str(item).strip()

        if not key or key in seen:
            continue

        keys.append(key)
        seen.add(key)

        if len(keys) >= max_keys:
            break

    return keys


# ==========================================================
# WORKER THREAD
# ==========================================================

class TranslationWorker(QThread):
    progress = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    processing_file = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        files,
        target_lang,
        style,
        model,
        advanced_prompt,
        server_url,
        provider="local",
        request_headers=None,
        cloud_api_keys=None,
        overwrite_existing=False,
        rate_limit_rpm=DEFAULT_RATE_LIMIT_RPM,
        min_delay_sec=DEFAULT_MIN_DELAY_SEC,
        parent=None,
    ):
        super().__init__(parent)

        self.files = files
        self.target_lang = target_lang
        self.style = style
        self.model = model
        self.advanced_prompt = advanced_prompt
        self.server_url = server_url
        self.provider = provider
        self.request_headers = request_headers or {}
        self.cloud_api_keys = normalize_api_keys(cloud_api_keys)
        self._cloud_key_index = 0
        self.overwrite_existing = bool(overwrite_existing)

        if provider == "openai_compat":
            self.chat_endpoint = openai_chat_completions_url(server_url)
        else:
            self.chat_endpoint = chat_url(server_url)

        self.rate_limit_rpm = int(rate_limit_rpm) if rate_limit_rpm else 0
        self.min_delay_sec = float(min_delay_sec) if min_delay_sec else 0.0

        self._rl_lock = Lock()
        self._recent_requests = []
        self._last_request_ts = 0.0

        self._cancel_lock = Lock()
        self._is_cancelled = False

    def cancel(self):
        with self._cancel_lock:
            self._is_cancelled = True

    def is_cancelled(self):
        with self._cancel_lock:
            return self._is_cancelled

    def _rate_limit_wait(self):
        if self.rate_limit_rpm <= 0 and self.min_delay_sec <= 0:
            return

        with self._rl_lock:
            now = time.monotonic()

            if self.min_delay_sec > 0 and self._last_request_ts > 0:
                wait = self.min_delay_sec - (now - self._last_request_ts)

                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()

            if self.rate_limit_rpm > 0:
                window = 60.0
                self._recent_requests = [
                    ts for ts in self._recent_requests
                    if now - ts <= window
                ]

                if len(self._recent_requests) >= self.rate_limit_rpm:
                    wait = self._recent_requests[0] + window - now

                    if wait > 0:
                        time.sleep(wait)
                        now = time.monotonic()
                        self._recent_requests = [
                            ts for ts in self._recent_requests
                            if now - ts <= window
                        ]

            now = time.monotonic()
            self._recent_requests.append(now)
            self._last_request_ts = now

    def run(self):
        try:
            total_files = len(self.files)

            for idx, file_path in enumerate(self.files):
                if self.is_cancelled():
                    self.log_message.emit("Översättning avbruten av användaren.")
                    break

                base_name = os.path.basename(file_path)
                self.processing_file.emit(
                    f"Bearbetar {idx + 1}/{total_files}: {base_name}"
                )
                self.log_message.emit(f"Bearbetar fil: {file_path}")

                try:
                    self.process_file(file_path)
                    self.progress.emit(idx + 1, total_files)
                except Exception as exc:
                    self.log_message.emit(
                        f"Kunde inte bearbeta {file_path}: {exc}"
                    )

            self.progress.emit(total_files, total_files)

            if self.is_cancelled():
                self.processing_file.emit("Avbruten")
            else:
                self.log_message.emit("Översättningen är klar!")
                self.processing_file.emit("Klar!")

        except Exception as exc:
            self.error.emit(f"Kritiskt fel: {exc}")
        finally:
            self.finished.emit()

    def process_file(self, file_path):
        path = Path(file_path)

        if "subs" in [part.lower() for part in path.parts]:
            self.log_message.emit(
                f"Hoppar över {path.name} - ligger i en 'Subs'-mapp."
            )
            return

        stem = path.stem
        base_stem = re.sub(
            r"\.[a-z]{2,3}(-[a-zA-Z]{2,3})?$",
            "",
            stem,
            flags=re.IGNORECASE,
        )

        if any(tag in path.name.lower() for tag in [".sv.", ".swe."]):
            self.log_message.emit(
                f"Hoppar över {path.name} - det är redan en svensk undertext."
            )
            return

        if not self.overwrite_existing:
            sv_out = path.parent / f"{base_stem}.sv{path.suffix}"
            swe_out = path.parent / f"{base_stem}.swe{path.suffix}"

            if sv_out.exists() or swe_out.exists():
                self.log_message.emit(
                    f"Hoppar över {path.name} - svensk undertext finns redan."
                )
                return

        with open(file_path, "r", encoding="utf-8-sig") as file:
            content = file.read()

        blocks = self.parse_srt(content)

        if not blocks:
            self.log_message.emit(f"Inga giltiga block hittades i {file_path}")
            return

        target_path = self.get_target_path(file_path)
        translated_blocks = []
        context_history = []

        total_blocks = len(blocks)
        total_batches = (total_blocks + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_number, start in enumerate(
            range(0, total_blocks, BATCH_SIZE),
            1,
        ):
            if self.is_cancelled():
                return

            batch = blocks[start:start + BATCH_SIZE]
            end = start + len(batch)

            self.log_message.emit(
                f"Översätter batch {batch_number}/{total_batches} "
                f"(block {start + 1}-{end} av {total_blocks})"
            )

            translated_batch = self.translate_batch(batch, context_history)

            if translated_batch is None:
                self.log_message.emit(
                    "Batch misslyckades. Försöker översätta blocken ett och ett."
                )
                translated_batch = self.translate_batch_fallback(
                    batch,
                    context_history,
                )

            for original_block, translated_text in zip(batch, translated_batch):
                index, timecode, original_text = original_block

                if translated_text:
                    final_text = translated_text.strip()
                else:
                    final_text = original_text.strip()
                    self.log_message.emit(
                        f"Behåller originaltext för block {index} "
                        "p.g.a. misslyckad översättning."
                    )

                translated_blocks.append((index, timecode, final_text))
                context_history.append(final_text)

                if len(context_history) > CONTEXT_TRANSLATED_BLOCKS:
                    context_history = context_history[-CONTEXT_TRANSLATED_BLOCKS:]

        self.save_srt(target_path, translated_blocks)
        self.log_message.emit(f"Sparade översatt fil: {target_path}")

    def get_target_path(self, file_path):
        path = Path(file_path)
        stem = re.sub(
            r"\.[a-z]{2,3}(-[a-zA-Z]{2,3})?$",
            "",
            path.stem,
            flags=re.IGNORECASE,
        )
        return path.parent / f"{stem}.{self.target_lang}{path.suffix}"

    def parse_srt(self, content):
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        blocks = []

        raw_blocks = re.split(r"\n{2,}", content)
        time_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*"
            r"\d{2}:\d{2}:\d{2},\d{3}.*)"
        )

        for raw_block in raw_blocks:
            lines = raw_block.strip().split("\n")

            if len(lines) < 3:
                continue

            index = lines[0].strip()
            timecode = lines[1].strip()
            text = "\n".join(lines[2:]).strip()

            if not index.isdigit():
                continue

            if not time_pattern.match(timecode):
                continue

            blocks.append((index, timecode, text))

        return blocks

    def protect_content(self, text, prefix):
        replacements = {}
        protected_text = text

        patterns = [
            r"\d{2}:\d{2}:\d{2},\d{3}",
            r"</?[^>]+>",
            r"\{\\[^}]+\}",
        ]

        counter = 0

        for pattern in patterns:
            matches = list(dict.fromkeys(re.findall(pattern, protected_text)))

            for match in matches:
                placeholder = f"<PH_{prefix}_{counter}>"
                protected_text = protected_text.replace(match, placeholder)
                replacements[placeholder] = match
                counter += 1

        return protected_text, replacements

    def restore_content(self, text, replacements):
        restored_text = text or ""

        for placeholder, original in replacements.items():
            restored_text = restored_text.replace(placeholder, original)

        return restored_text

    def generate_system_prompt(self):
        target_name = language_name_from_code(self.target_lang)

        style_map = {
            "Natural (recommended)": "natural, idiomatic and subtitle-friendly",
            "Formal": "formal and polished",
            "Simple clear": "simple, clear and concise",
        }
        style_desc = style_map.get(self.style, "natural and subtitle-friendly")

        prompt = (
            f"You are a professional subtitle translator.\n"
            f"Translate English subtitles into {target_name}.\n\n"
            f"Style: {style_desc}.\n\n"
            "STRICT RULES:\n"
            "1. Translate ONLY the text inside each item.\n"
            "2. Keep every item marker exactly unchanged:\n"
            "   <<<ITEM_1>>> and <<<END_ITEM_1>>> etc.\n"
            "3. Return the same number of items as the input.\n"
            "4. Keep placeholders exactly unchanged, for example "
            "<PH_B1_0>, <PH_B2_3>.\n"
            "5. Do not translate placeholders.\n"
            "6. Do not add explanations, notes, comments, markdown or code blocks.\n"
            "7. Output only the translated batch in the same marker format.\n"
            "8. Keep subtitles concise and natural.\n"
            "9. Translate descriptive sound/action text inside square brackets.\n"
            "   Keep the square brackets themselves unchanged.\n"
            "   Examples:\n"
            "   [PANTING] -> [FLÄMTAR]\n"
            "   [LAUGHS] -> [SKRATTAR]\n"
            "   [SIGHS] -> [SUCKAR]\n"
            "   [MUSIC PLAYING] -> [MUSIK SPELAS]\n"
            "   [DOOR OPENS] -> [DÖRREN ÖPPNAS]\n"
        )

        if self.advanced_prompt.strip():
            try:
                prompt = self.advanced_prompt.format(
                    target_lang=target_name,
                    target_code=self.target_lang,
                    style=style_desc,
                )
            except Exception:
                prompt = self.advanced_prompt

            prompt += (
                "\n\nAdditional mandatory formatting rules:\n"
                "Return the same <<<ITEM_n>>> blocks and <<<END_ITEM_n>>> markers. "
                "Do not remove or rename markers. Do not use markdown. "
                "Translate descriptive sound/action text inside square brackets, "
                "for example [PANTING] -> [FLÄMTAR], "
                "[LAUGHS] -> [SKRATTAR], "
                "but keep the square brackets themselves unchanged."
            )

        return prompt

    def build_batch_user_text(self, protected_items, context_history):
        parts = []

        if context_history:
            parts.append(
                "Previous translated subtitle context. Use only for meaning, "
                "names, pronouns and tone. Do not repeat it:\n"
            )

            for idx, line in enumerate(context_history, 1):
                parts.append(f"CONTEXT_{idx}: {line}")

            parts.append("\nBatch to translate:")

        for item_id, text in protected_items:
            parts.append(f"<<<ITEM_{item_id}>>>")
            parts.append(text)
            parts.append(f"<<<END_ITEM_{item_id}>>>")

        return "\n".join(parts)

    def translate_batch(self, batch, context_history):
        protected_items = []
        replacements_by_item = {}

        for item_pos, block in enumerate(batch, 1):
            _, _, text = block
            prefix = f"B{item_pos}"
            protected_text, replacements = self.protect_content(text, prefix)
            protected_items.append((item_pos, protected_text))
            replacements_by_item[item_pos] = replacements

        system_prompt = self.generate_system_prompt()
        user_text = self.build_batch_user_text(protected_items, context_history)

        raw_output = self.translate_text(system_prompt, user_text)

        if not raw_output:
            return None

        cleaned_output = self.clean_model_output(raw_output)
        parsed = self.parse_batch_response(cleaned_output, len(batch))

        if parsed is None:
            self.log_message.emit(
                "Kunde inte tolka batch-svaret perfekt. "
                "Försöker fallback-parser."
            )
            parsed = self.parse_batch_response_fallback(
                cleaned_output,
                len(batch),
            )

        if parsed is None or len(parsed) != len(batch):
            return None

        restored = []

        for item_pos, translated_text in enumerate(parsed, 1):
            restored_text = self.restore_content(
                translated_text,
                replacements_by_item.get(item_pos, {}),
            )
            restored.append(self.postprocess_translation(restored_text))

        return restored

    def translate_batch_fallback(self, batch, context_history):
        results = []

        for item_pos, block in enumerate(batch, 1):
            if self.is_cancelled():
                return results

            _, _, text = block
            protected_text, replacements = self.protect_content(
                text,
                f"FB{item_pos}",
            )

            target_name = language_name_from_code(self.target_lang)
            system_prompt = (
                f"Translate English subtitle text into {target_name}.\n"
                "Return only the translated text.\n"
                "Keep placeholders like <PH_FB1_0> exactly unchanged.\n"
                "Do not use markdown. Do not add explanations.\n"
                "Translate descriptive sound/action text inside square brackets.\n"
                "Keep the square brackets themselves unchanged.\n"
                "Examples:\n"
                "[PANTING] -> [FLÄMTAR]\n"
                "[LAUGHS] -> [SKRATTAR]\n"
                "[SIGHS] -> [SUCKAR]\n"
                "[MUSIC PLAYING] -> [MUSIK SPELAS]\n"
                "[DOOR OPENS] -> [DÖRREN ÖPPNAS]"
            )

            if context_history:
                user_text = (
                    "Previous translated context, do not repeat:\n"
                    + "\n".join(context_history[-CONTEXT_TRANSLATED_BLOCKS:])
                    + "\n\nText to translate:\n"
                    + protected_text
                )
            else:
                user_text = protected_text

            translated = self.translate_text(system_prompt, user_text)
            translated = self.clean_model_output(translated or "")
            translated = self.restore_content(translated, replacements)
            translated = self.postprocess_translation(translated)

            results.append(translated or text)

            if translated:
                context_history.append(translated)

                if len(context_history) > CONTEXT_TRANSLATED_BLOCKS:
                    del context_history[:-CONTEXT_TRANSLATED_BLOCKS]

        return results

    def parse_batch_response(self, text, expected_count):
        results = []

        for item_id in range(1, expected_count + 1):
            pattern = (
                rf"<<<ITEM_{item_id}>>>\s*"
                rf"([\s\S]*?)"
                rf"\s*<<<END_ITEM_{item_id}>>>"
            )
            match = re.search(pattern, text)

            if not match:
                return None

            results.append(match.group(1).strip())

        return results

    def parse_batch_response_fallback(self, text, expected_count):
        text = self.clean_model_output(text)

        item_pattern = re.compile(
            r"<<<ITEM_\d+>>>\s*([\s\S]*?)(?=<<<ITEM_\d+>>>|$)"
        )
        matches = item_pattern.findall(text)

        if matches and len(matches) == expected_count:
            cleaned = []

            for match in matches:
                match = re.sub(r"<<<END_ITEM_\d+>>>", "", match).strip()
                cleaned.append(match)

            return cleaned

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = [
            line for line in lines
            if not re.match(r"^<<<\/?ITEM", line)
            and not re.match(r"^<<<END_ITEM", line)
        ]

        if len(lines) == expected_count:
            return lines

        return None

    def clean_model_output(self, text):
        if not text:
            return ""

        text = text.strip()

        text = re.sub(
            r"^```(?:json|text|txt|srt|markdown|md)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

        text = re.sub(
            r"```(?:json|text|txt|srt|markdown|md)?\s*([\s\S]*?)```",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

        prefixes = [
            "Here is the translation:",
            "Here are the translations:",
            "Translation:",
            "Translated text:",
        ]

        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        return text.strip()

    def postprocess_translation(self, text):
        text = self.clean_model_output(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_response_text(self, data):
        if self.provider == "openai_compat":
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return None

        return data.get("message", {}).get("content")

    def _build_payload(self, system_prompt, text):
        if self.provider == "openai_compat":
            return {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": DEFAULT_TEMPERATURE,
                "stream": False,
            }

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {
                "temperature": DEFAULT_TEMPERATURE,
            },
        }

    def _headers_for_cloud_key(self, api_key):
        headers = dict(self.request_headers or {})

        if self.provider == "cloud" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def _retry_wait_seconds(self, response, attempt):
        retry_after = response.headers.get("Retry-After")

        try:
            wait_s = float(retry_after) if retry_after else None
        except Exception:
            wait_s = None

        if wait_s is None:
            wait_s = min(MAX_BACKOFF_SEC, RETRY_DELAY * (2 ** attempt))

        return wait_s

    def _post_chat_request(self, payload, headers):
        self._rate_limit_wait()

        return requests.post(
            self.chat_endpoint,
            json=payload,
            headers=headers,
            timeout=180,
        )

    def _translate_text_with_cloud_key_rotation(self, payload):
        total_keys = len(self.cloud_api_keys)

        if total_keys <= 0:
            return self._translate_text_single_key(payload)

        for attempt in range(RETRY_COUNT + 1):
            if self.is_cancelled():
                return None

            rate_limited_waits = []
            had_non_rate_limit_error = False
            start_index = self._cloud_key_index % total_keys

            for offset in range(total_keys):
                if self.is_cancelled():
                    return None

                key_index = (start_index + offset) % total_keys
                api_key = self.cloud_api_keys[key_index]
                headers = self._headers_for_cloud_key(api_key)

                try:
                    response = self._post_chat_request(payload, headers)

                    if response.status_code == 200:
                        self._cloud_key_index = key_index
                        data = response.json()
                        return self.clean_model_output(
                            self._extract_response_text(data) or ""
                        )

                    if response.status_code == 429:
                        wait_s = self._retry_wait_seconds(response, attempt)
                        rate_limited_waits.append(wait_s)
                        self._cloud_key_index = (key_index + 1) % total_keys

                        self.log_message.emit(
                            "Rate limit 429 på Ollama Cloud-nyckel "
                            f"{key_index + 1}/{total_keys}. "
                            "Testar nästa nyckel direkt..."
                        )
                        continue

                    had_non_rate_limit_error = True
                    self.log_message.emit(
                        "API-fel med Ollama Cloud-nyckel "
                        f"{key_index + 1}/{total_keys} "
                        f"({attempt + 1}/{RETRY_COUNT + 1}): "
                        f"{response.status_code} - {response.text[:500]}"
                    )
                    break

                except Exception as exc:
                    had_non_rate_limit_error = True
                    self.log_message.emit(
                        "Nätverksfel med Ollama Cloud-nyckel "
                        f"{key_index + 1}/{total_keys} "
                        f"({attempt + 1}/{RETRY_COUNT + 1}): {exc}"
                    )
                    break

            if attempt < RETRY_COUNT:
                if rate_limited_waits and not had_non_rate_limit_error:
                    wait_s = min(rate_limited_waits)
                    self.log_message.emit(
                        "Alla Ollama Cloud-nycklar verkar vara rate-limitade. "
                        f"Väntar {wait_s:.1f}s innan nytt varv..."
                    )
                    time.sleep(wait_s)
                else:
                    time.sleep(RETRY_DELAY)

        return None

    def _translate_text_single_key(self, payload):
        for attempt in range(RETRY_COUNT + 1):
            if self.is_cancelled():
                return None

            try:
                response = self._post_chat_request(
                    payload,
                    self.request_headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    return self.clean_model_output(
                        self._extract_response_text(data) or ""
                    )

                if response.status_code == 429:
                    wait_s = self._retry_wait_seconds(response, attempt)

                    self.log_message.emit(
                        f"Rate limit 429. Väntar {wait_s:.1f}s "
                        f"({attempt + 1}/{RETRY_COUNT + 1})..."
                    )
                    time.sleep(wait_s)
                    continue

                self.log_message.emit(
                    f"API-fel ({attempt + 1}/{RETRY_COUNT + 1}): "
                    f"{response.status_code} - {response.text[:500]}"
                )

            except Exception as exc:
                self.log_message.emit(
                    f"Nätverksfel ({attempt + 1}/{RETRY_COUNT + 1}): {exc}"
                )

            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)

        return None

    def translate_text(self, system_prompt, text):
        payload = self._build_payload(system_prompt, text)

        if self.provider == "cloud" and self.cloud_api_keys:
            return self._translate_text_with_cloud_key_rotation(payload)

        return self._translate_text_single_key(payload)

    def save_srt(self, path, blocks):
        with open(path, "w", encoding="utf-8") as file:
            for index, timecode, text in blocks:
                file.write(f"{index}\n{timecode}\n{text.strip()}\n\n")


# ==========================================================
# DIALOGS
# ==========================================================

class OllamaSettingsDialog(QDialog):
    def __init__(
        self,
        provider,
        current_server_url,
        cloud_url,
        cloud_api_keys,
        openai_compat_url,
        openai_api_key,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle("Anslutningsinställningar")
        self.setMinimumWidth(620)
        self.setMinimumHeight(620)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Lokal Ollama", "local")
        self.provider_combo.addItem("Ollama Cloud", "cloud")
        self.provider_combo.addItem(
            "OpenAI-kompatibel endpoint",
            "openai_compat",
        )

        index = self.provider_combo.findData(provider)

        if index >= 0:
            self.provider_combo.setCurrentIndex(index)

        self.url_edit = QLineEdit(current_server_url)
        self.cloud_url_edit = QLineEdit(cloud_url)
        self.openai_compat_url_edit = QLineEdit(openai_compat_url)

        self.openai_api_key_edit = QLineEdit(openai_api_key)
        self.openai_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

        form_layout.addRow("Anslutning:", self.provider_combo)
        form_layout.addRow("Lokal server URL:", self.url_edit)
        form_layout.addRow("Cloud base URL:", self.cloud_url_edit)
        form_layout.addRow(
            "OpenAI-kompatibel base URL:",
            self.openai_compat_url_edit,
        )
        form_layout.addRow(
            "API-nyckel OpenAI-kompatibel:",
            self.openai_api_key_edit,
        )

        layout.addLayout(form_layout)

        self.cloud_api_key_edits = []
        normalized_keys = normalize_api_keys(cloud_api_keys)

        cloud_keys_group = QGroupBox("Ollama Cloud API-nycklar, max 20")
        cloud_keys_layout = QFormLayout(cloud_keys_group)

        for idx in range(MAX_CLOUD_API_KEYS):
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.EchoMode.Password)
            edit.setPlaceholderText(
                f"API-nyckel {idx + 1} "
                "(lämna tom om den inte används)"
            )

            if idx < len(normalized_keys):
                edit.setText(normalized_keys[idx])

            self.cloud_api_key_edits.append(edit)
            cloud_keys_layout.addRow(f"Nyckel {idx + 1}:", edit)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cloud_keys_group)
        scroll.setMinimumHeight(280)
        layout.addWidget(scroll)

        info = QLabel(
            "Lokal Ollama använder t.ex. http://localhost:11434/api\n"
            "Ollama Cloud använder t.ex. https://ollama.com/api\n"
            "OpenAI-kompatibel endpoint använder t.ex. http://localhost:8000/v1\n\n"
            "Vid HTTP 429 på Ollama Cloud byter programmet automatiskt "
            "till nästa angivna Cloud-nyckel."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def get_values(self):
        cloud_api_keys = [
            edit.text().strip()
            for edit in self.cloud_api_key_edits
            if edit.text().strip()
        ]

        cloud_api_keys = normalize_api_keys(cloud_api_keys)

        return {
            "provider": self.provider_combo.currentData(),
            "server_url": self.url_edit.text().strip(),
            "cloud_url": self.cloud_url_edit.text().strip(),
            "cloud_api_keys": cloud_api_keys,
            "openai_compat_url": self.openai_compat_url_edit.text().strip(),
            "openai_api_key": self.openai_api_key_edit.text().strip(),
        }


class FavoriteModelsDialog(QDialog):
    def __init__(self, models, favorites, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Favoritmodeller")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Markera modeller du vill ha som favoriter. "
            "Favoriter visas överst i modellistan."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget()

        for name in models:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if name in set(favorites or [])
                else Qt.CheckState.Unchecked
            )
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        row = QHBoxLayout()

        select_all = QPushButton("Markera alla")
        select_none = QPushButton("Avmarkera alla")

        select_all.clicked.connect(self._check_all)
        select_none.clicked.connect(self._uncheck_all)

        row.addWidget(select_all)
        row.addWidget(select_none)
        row.addStretch()

        layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def _check_all(self):
        for row in range(self.list_widget.count()):
            self.list_widget.item(row).setCheckState(Qt.CheckState.Checked)

    def _uncheck_all(self):
        for row in range(self.list_widget.count()):
            self.list_widget.item(row).setCheckState(Qt.CheckState.Unchecked)

    def get_favorites(self):
        favorites = []

        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)

            if item.checkState() == Qt.CheckState.Checked:
                favorites.append(item.text())

        return favorites


class TranslationOptionsDialog(QDialog):
    def __init__(self, overwrite_existing, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Översättningsalternativ")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Styr hur programmet hanterar redan existerande översatta filer."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.overwrite_checkbox = QCheckBox(
            "Skriv över befintliga mål-filer, t.ex. .sv.srt"
        )
        self.overwrite_checkbox.setChecked(bool(overwrite_existing))
        layout.addWidget(self.overwrite_checkbox)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def get_overwrite_existing(self):
        return self.overwrite_checkbox.isChecked()


class RateLimitDialog(QDialog):
    def __init__(self, rate_limit_rpm, min_delay_sec, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Rate limit")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Använd detta om endpointen ger HTTP 429 / rate limit.\n"
            "Sätt 0 för att stänga av respektive begränsning.\n\n"
            "OBS: För Ollama Cloud med flera nycklar byter programmet ändå "
            "nyckel direkt vid 429."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(0, 600)
        self.rpm_spin.setValue(int(rate_limit_rpm))
        self.rpm_spin.setSuffix(" rpm")
        form.addRow("Max anrop/minut:", self.rpm_spin)

        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 60.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setSingleStep(0.25)
        self.delay_spin.setValue(float(min_delay_sec))
        self.delay_spin.setSuffix(" s")
        form.addRow("Min paus mellan anrop:", self.delay_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)


# ==========================================================
# MAIN WINDOW
# ==========================================================

class SRTTranslatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SRT Översättare")
        self.setMinimumSize(820, 720)

        config_dir = get_config_dir()
        self.settings_file = os.path.join(config_dir, SETTINGS_FILE)
        self.settings = QSettings(self.settings_file, QSettings.Format.IniFormat)

        self.provider = DEFAULT_PROVIDER
        self.server_url = DEFAULT_SERVER_URL
        self.cloud_url = DEFAULT_CLOUD_URL
        self.openai_compat_url = DEFAULT_OPENAI_COMPAT_URL
        self.ollama_api_keys = []
        self.openai_api_key = ""

        self.current_theme = "Dracula"
        self.selected_files = []
        self.available_models = []
        self.favorite_models = set()
        self.overwrite_existing = False
        self.rate_limit_rpm = DEFAULT_RATE_LIMIT_RPM
        self.min_delay_sec = DEFAULT_MIN_DELAY_SEC

        self.worker_thread = None

        self.init_ui()
        self.load_settings()
        self.apply_theme(self.current_theme)
        self.refresh_models()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.create_menu_bar()

        model_group = QGroupBox("Modellval")
        model_layout = QHBoxLayout()

        model_layout.addWidget(QLabel("Modell:"))

        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo, 3)

        self.refresh_btn = QPushButton("Uppdatera modeller")
        self.refresh_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(self.refresh_btn)

        self.model_status = QLabel("Inga modeller laddade")
        model_layout.addWidget(self.model_status)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        lang_group = QGroupBox("Översättningsinställningar")
        lang_layout = QVBoxLayout()

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Målspråk:"))

        self.lang_combo = QComboBox()

        for name, code in LANGUAGES:
            self.lang_combo.addItem(name, code)

        lang_row.addWidget(self.lang_combo, 1)

        lang_row.addWidget(QLabel("Stil:"))

        self.style_combo = QComboBox()
        self.style_combo.addItems(STYLES)
        lang_row.addWidget(self.style_combo, 1)

        lang_layout.addLayout(lang_row)

        lang_layout.addWidget(QLabel("Avancerad prompt:"))

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Valfritt. Använd {target_lang}, {target_code} och {style}."
        )
        self.prompt_edit.setMinimumHeight(100)
        lang_layout.addWidget(self.prompt_edit)

        lang_group.setLayout(lang_layout)
        main_layout.addWidget(lang_group)

        file_group = QGroupBox("Filval")
        file_layout = QVBoxLayout()

        self.file_label = QLabel("Ingen fil eller mapp vald")
        file_layout.addWidget(self.file_label)

        file_buttons = QHBoxLayout()

        self.file_btn = QPushButton("Välj fil (.srt)")
        self.file_btn.clicked.connect(self.select_file)
        file_buttons.addWidget(self.file_btn)

        self.files_btn = QPushButton("Välj flera filer (.srt)")
        self.files_btn.clicked.connect(self.select_files)
        file_buttons.addWidget(self.files_btn)

        self.folder_btn = QPushButton("Välj mapp")
        self.folder_btn.clicked.connect(self.select_folder)
        file_buttons.addWidget(self.folder_btn)

        self.folders_btn = QPushButton("Välj flera mappar")
        self.folders_btn.clicked.connect(self.select_folders)
        file_buttons.addWidget(self.folders_btn)

        file_layout.addLayout(file_buttons)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        action_group = QGroupBox("Åtgärder")
        action_layout = QHBoxLayout()

        self.start_btn = QPushButton("Starta översättning")
        self.start_btn.clicked.connect(self.start_translation)
        action_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Avbryt översättning")
        self.cancel_btn.clicked.connect(self.cancel_translation)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)

        action_group.setLayout(action_layout)
        main_layout.addWidget(action_group)

        progress_group = QGroupBox("Framsteg")
        progress_layout = QVBoxLayout()

        self.current_file_label = QLabel("Väntar på start...")
        progress_layout.addWidget(self.current_file_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFontFamily("Courier New")
        self.log_view.setMinimumHeight(180)
        progress_layout.addWidget(self.log_view)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)

        tools_group = QGroupBox("Verktyg")
        tools_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Rensa debug/logg")
        self.clear_btn.clicked.connect(self.clear_log)
        tools_layout.addWidget(self.clear_btn)

        self.open_btn = QPushButton("Öppna målmapp")
        self.open_btn.clicked.connect(self.open_output_folder)
        tools_layout.addWidget(self.open_btn)

        tools_layout.addStretch()
        tools_group.setLayout(tools_layout)
        main_layout.addWidget(tools_group)

        self.statusBar().showMessage("Redo")

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Inställningar")

        connection_action = QAction("Anslutningsinställningar...", self)
        connection_action.triggered.connect(self.open_ollama_settings)

        fav_models_action = QAction("Favoritmodeller...", self)
        fav_models_action.triggered.connect(self.open_favorite_models)

        translation_opts_action = QAction("Översättningsalternativ...", self)
        translation_opts_action.triggered.connect(self.open_translation_options)

        rate_limit_action = QAction("Rate limit...", self)
        rate_limit_action.triggered.connect(self.open_rate_limit_settings)

        save_settings_action = QAction("Spara inställningar", self)
        save_settings_action.triggered.connect(self.save_settings_from_ui)

        theme_menu = QMenu("Tema", self)
        self.theme_actions = {}

        for theme_name in sorted(THEMES.keys(), key=str.lower):
            action = QAction(theme_name, self, checkable=True)
            action.triggered.connect(
                lambda _, selected=theme_name: self.apply_theme(selected)
            )
            theme_menu.addAction(action)
            self.theme_actions[theme_name] = action

        settings_menu.addAction(connection_action)
        settings_menu.addAction(fav_models_action)
        settings_menu.addAction(translation_opts_action)
        settings_menu.addAction(rate_limit_action)
        settings_menu.addSeparator()
        settings_menu.addAction(save_settings_action)
        settings_menu.addMenu(theme_menu)

        tools_menu = menu_bar.addMenu("Verktyg")

        clear_log_action = QAction("Rensa debug/logg", self)
        clear_log_action.triggered.connect(self.clear_log)
        tools_menu.addAction(clear_log_action)

    def load_settings(self):
        self.provider = self.settings.value("provider", DEFAULT_PROVIDER)
        self.server_url = self.settings.value("server_url", DEFAULT_SERVER_URL)
        self.cloud_url = self.settings.value("cloud_url", DEFAULT_CLOUD_URL)
        self.openai_compat_url = self.settings.value(
            "openai_compat_url",
            DEFAULT_OPENAI_COMPAT_URL,
        )

        legacy_key = self.settings.value("ollama_api_key", "")
        raw_keys = self.settings.value("ollama_api_keys", "[]")
        self.ollama_api_keys = normalize_api_keys(raw_keys)

        if not self.ollama_api_keys:
            self.ollama_api_keys = normalize_api_keys(legacy_key)

        self.openai_api_key = self.settings.value("openai_api_key", "")
        self.current_theme = self.settings.value("theme", "Dracula")

        fav_raw = self.settings.value("favorite_models", "[]")

        try:
            fav_list = json.loads(fav_raw) if isinstance(fav_raw, str) else list(fav_raw)
        except Exception:
            fav_list = []

        self.favorite_models = {item for item in fav_list if isinstance(item, str)}

        self.overwrite_existing = self.settings.value(
            "overwrite_existing",
            False,
            type=bool,
        )
        self.rate_limit_rpm = self.settings.value(
            "rate_limit_rpm",
            DEFAULT_RATE_LIMIT_RPM,
            type=int,
        )
        self.min_delay_sec = self.settings.value(
            "min_delay_sec",
            DEFAULT_MIN_DELAY_SEC,
            type=float,
        )

        model = self.settings.value("model", "")

        if model:
            self.model_combo.addItem(model)
            self.model_combo.setCurrentText(model)

        target_language = self.settings.value("target_language", "sv")
        lang_index = self.lang_combo.findData(target_language)

        if lang_index >= 0:
            self.lang_combo.setCurrentIndex(lang_index)

        style = self.settings.value("style", "Natural (recommended)")
        style_index = self.style_combo.findText(style)

        if style_index >= 0:
            self.style_combo.setCurrentIndex(style_index)

        self.prompt_edit.setPlainText(
            self.settings.value("advanced_prompt", "")
        )

    def save_settings_from_ui(self):
        self.save_settings()
        self.statusBar().showMessage("Inställningar sparade.")

    def save_settings(self):
        self.ollama_api_keys = normalize_api_keys(self.ollama_api_keys)

        self.settings.setValue("provider", self.provider)
        self.settings.setValue("server_url", self.server_url)
        self.settings.setValue("cloud_url", self.cloud_url)
        self.settings.setValue("openai_compat_url", self.openai_compat_url)

        first_cloud_key = self.ollama_api_keys[0] if self.ollama_api_keys else ""
        self.settings.setValue("ollama_api_key", first_cloud_key)
        self.settings.setValue(
            "ollama_api_keys",
            json.dumps(self.ollama_api_keys, ensure_ascii=False),
        )

        self.settings.setValue("openai_api_key", self.openai_api_key)
        self.settings.setValue("theme", self.current_theme)
        self.settings.setValue(
            "favorite_models",
            json.dumps(sorted(self.favorite_models, key=str.lower)),
        )
        self.settings.setValue("overwrite_existing", bool(self.overwrite_existing))
        self.settings.setValue("rate_limit_rpm", int(self.rate_limit_rpm))
        self.settings.setValue("min_delay_sec", float(self.min_delay_sec))

        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("target_language", self.lang_combo.currentData())
        self.settings.setValue("style", self.style_combo.currentText())
        self.settings.setValue(
            "advanced_prompt",
            self.prompt_edit.toPlainText(),
        )

    def get_active_base_url(self):
        if self.provider == "cloud":
            return self.cloud_url

        if self.provider == "openai_compat":
            return self.openai_compat_url

        return self.server_url

    def get_cloud_api_keys(self):
        return normalize_api_keys(self.ollama_api_keys)

    def get_request_headers(self, cloud_api_key=None):
        headers = {}

        if self.provider == "cloud":
            key = cloud_api_key

            if not key:
                keys = self.get_cloud_api_keys()
                key = keys[0] if keys else ""

            if key:
                headers["Authorization"] = f"Bearer {key}"

        if self.provider == "openai_compat":
            headers["Content-Type"] = "application/json"

            if self.openai_api_key:
                headers["Authorization"] = f"Bearer {self.openai_api_key}"

        return headers

    def open_ollama_settings(self):
        dialog = OllamaSettingsDialog(
            self.provider,
            self.server_url,
            self.cloud_url,
            self.ollama_api_keys,
            self.openai_compat_url,
            self.openai_api_key,
            self,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.get_values()

        self.provider = values["provider"]
        self.server_url = values["server_url"] or DEFAULT_SERVER_URL
        self.cloud_url = values["cloud_url"] or DEFAULT_CLOUD_URL
        self.ollama_api_keys = normalize_api_keys(values["cloud_api_keys"])
        self.openai_compat_url = (
            values["openai_compat_url"] or DEFAULT_OPENAI_COMPAT_URL
        )
        self.openai_api_key = values["openai_api_key"]

        self.save_settings()
        self.refresh_models()
        self.statusBar().showMessage("Anslutningsinställningar uppdaterade.")

    def open_favorite_models(self):
        if not self.available_models:
            self.refresh_models()

        if not self.available_models:
            QMessageBox.information(
                self,
                "Inga modeller",
                "Inga modeller är laddade ännu.",
            )
            return

        dialog = FavoriteModelsDialog(
            self.available_models,
            self.favorite_models,
            self,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.favorite_models = set(dialog.get_favorites())
            self.save_settings()
            self.refresh_models()
            self.statusBar().showMessage("Favoritmodeller uppdaterade.")

    def open_translation_options(self):
        dialog = TranslationOptionsDialog(self.overwrite_existing, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.overwrite_existing = dialog.get_overwrite_existing()
            self.save_settings()
            state = "på" if self.overwrite_existing else "av"
            self.statusBar().showMessage(f"Skriv över befintliga filer: {state}")

    def open_rate_limit_settings(self):
        dialog = RateLimitDialog(
            self.rate_limit_rpm,
            self.min_delay_sec,
            self,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.rate_limit_rpm = int(dialog.rpm_spin.value())
            self.min_delay_sec = float(dialog.delay_spin.value())
            self.save_settings()

            self.log_message(
                f"Rate limit uppdaterad: {self.rate_limit_rpm} rpm, "
                f"{self.min_delay_sec:.2f}s paus."
            )

    def refresh_models(self):
        current = self.model_combo.currentText()

        self.model_combo.clear()
        self.model_status.setText("Laddar modeller...")
        self.refresh_btn.setEnabled(False)

        try:
            base_url = self.get_active_base_url()
            response = None

            if self.provider == "openai_compat":
                response = requests.get(
                    openai_models_url(base_url),
                    headers=self.get_request_headers(),
                    timeout=10,
                )

            elif self.provider == "cloud":
                cloud_keys = self.get_cloud_api_keys()

                if not cloud_keys:
                    self.available_models = []
                    self.model_status.setText("Ingen Ollama Cloud API-nyckel angiven")
                    return

                last_response = None

                for key_index, api_key in enumerate(cloud_keys):
                    candidate = requests.get(
                        tags_url(base_url),
                        headers=self.get_request_headers(api_key),
                        timeout=10,
                    )
                    last_response = candidate

                    if candidate.status_code == 200:
                        response = candidate

                        if key_index > 0:
                            self.statusBar().showMessage(
                                "Modellhämtning använde Cloud-nyckel "
                                f"{key_index + 1}/{len(cloud_keys)}"
                            )

                        break

                    if candidate.status_code == 429:
                        continue

                    response = candidate
                    break

                if response is None:
                    response = last_response

            else:
                response = requests.get(
                    tags_url(base_url),
                    headers=self.get_request_headers(),
                    timeout=10,
                )

            if response is None:
                self.available_models = []
                self.model_status.setText("Kunde inte hämta modeller")
                return

            if response.status_code != 200:
                self.available_models = []
                self.model_status.setText(
                    f"Fel: {response.status_code} - {response.text[:120]}"
                )
                return

            data = response.json()

            if self.provider == "openai_compat":
                models = [item.get("id", "") for item in data.get("data", [])]
            else:
                models = [item.get("name", "") for item in data.get("models", [])]

            models = sorted([model for model in models if model], key=str.lower)
            self.available_models = models

            favorites = [model for model in models if model in self.favorite_models]
            rest = [model for model in models if model not in self.favorite_models]

            if favorites:
                self.model_combo.addItems(favorites)

                if rest:
                    self.model_combo.insertSeparator(self.model_combo.count())

            if rest:
                self.model_combo.addItems(rest)

            if current and current in models:
                index = self.model_combo.findText(current)

                if index >= 0:
                    self.model_combo.setCurrentIndex(index)

            provider_name = {
                "local": "Lokalt",
                "cloud": "Cloud",
                "openai_compat": "OpenAI-kompatibel",
            }.get(self.provider, self.provider)

            cloud_info = ""

            if self.provider == "cloud":
                cloud_info = f", {len(self.get_cloud_api_keys())} nycklar"

            self.model_status.setText(
                f"{len(models)} modeller laddade ({provider_name}"
                f"{cloud_info}, {len(favorites)} favoriter)"
            )

        except Exception as exc:
            self.available_models = []
            self.model_status.setText(f"Fel: {exc}")
        finally:
            self.refresh_btn.setEnabled(True)

    def update_file_label(self):
        count = len(self.selected_files)

        if count == 0:
            self.file_label.setText("Ingen fil eller mapp vald")
        elif count == 1:
            self.file_label.setText(f"Fil: {self.selected_files[0]}")
        else:
            self.file_label.setText(f"{count} SRT-filer valda")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Välj SRT-fil",
            "",
            "Undertextfiler (*.srt)",
        )

        if file_path:
            self.selected_files = [file_path]
            self.update_file_label()

    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Välj en eller flera SRT-filer",
            "",
            "Undertextfiler (*.srt)",
        )

        if file_paths:
            self.selected_files = sorted(set(file_paths))
            self.update_file_label()

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Välj mapp med SRT-filer",
        )

        if folder_path:
            self.selected_files = self.find_srt_files(folder_path)
            self.update_file_label()

    def select_folders(self):
        dialog = QFileDialog(self, "Välj en eller flera mappar")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        for view in dialog.findChildren((QListView, QTreeView)):
            view.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )

        if dialog.exec():
            folders = dialog.selectedFiles()
            all_srt = []

            for folder in folders:
                all_srt.extend(self.find_srt_files(folder))

            self.selected_files = sorted(set(all_srt))
            self.file_label.setText(
                f"Mappar: {len(folders)} | SRT-filer: {len(self.selected_files)}"
            )

    def find_srt_files(self, folder_path):
        srt_files = []

        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(".srt"):
                    srt_files.append(os.path.join(root, filename))

        return srt_files

    def start_translation(self):
        if not self.selected_files:
            QMessageBox.warning(
                self,
                "Inga filer",
                "Välj en fil, flera filer eller mapp först.",
            )
            return

        if not self.model_combo.currentText():
            QMessageBox.warning(self, "Ingen modell", "Välj en modell först.")
            return

        cloud_api_keys = self.get_cloud_api_keys()

        if self.provider == "cloud" and not cloud_api_keys:
            QMessageBox.warning(
                self,
                "API-nyckel saknas",
                "Ange minst en API-nyckel för Ollama Cloud först.",
            )
            return

        self.save_settings()

        target_lang = self.lang_combo.currentData()
        style = self.style_combo.currentText()
        model = self.model_combo.currentText()
        advanced_prompt = self.prompt_edit.toPlainText()

        base_url = self.get_active_base_url()

        headers = self.get_request_headers(
            cloud_api_keys[0]
            if self.provider == "cloud" and cloud_api_keys
            else None
        )

        provider_name = {
            "local": "Lokal Ollama",
            "cloud": "Ollama Cloud",
            "openai_compat": "OpenAI-kompatibel endpoint",
        }.get(self.provider, self.provider)

        self.worker_thread = TranslationWorker(
            files=self.selected_files,
            target_lang=target_lang,
            style=style,
            model=model,
            advanced_prompt=advanced_prompt,
            server_url=base_url,
            provider=self.provider,
            request_headers=headers,
            cloud_api_keys=cloud_api_keys,
            overwrite_existing=self.overwrite_existing,
            rate_limit_rpm=self.rate_limit_rpm,
            min_delay_sec=self.min_delay_sec,
        )

        self.worker_thread.progress.connect(self.update_progress)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.processing_file.connect(self.update_current_file)
        self.worker_thread.finished.connect(self.translation_finished)
        self.worker_thread.error.connect(self.show_error)

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        self.log_message("Startar översättning...")
        self.log_message(f"Anslutning: {provider_name}")
        self.log_message(f"Bas-URL: {base_url}")

        if self.provider == "cloud":
            self.log_message(
                f"Ollama Cloud API-nycklar aktiva: {len(cloud_api_keys)}"
            )
            self.log_message(
                "Vid 429 rate limit byter programmet automatiskt "
                "till nästa nyckel."
            )

        self.log_message(f"Batchstorlek: {BATCH_SIZE}")
        self.log_message(f"Kontextblock: {CONTEXT_TRANSLATED_BLOCKS}")
        self.log_message("Text inom [hakparenteser] översätts.")

        self.current_file_label.setText("Startar...")
        self.worker_thread.start()

    def cancel_translation(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.cancel_btn.setEnabled(False)
            self.log_message("Avbryter...")

    def translation_finished(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.worker_thread = None

    def update_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        else:
            self.progress_bar.setValue(0)

    def update_current_file(self, filename):
        self.current_file_label.setText(filename)

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def show_error(self, message):
        QMessageBox.critical(self, "Fel", message)
        self.translation_finished()

    def clear_log(self):
        self.log_view.clear()
        self.log_message("Debug/logg rensad.")

    def open_output_folder(self):
        if not self.selected_files:
            QMessageBox.information(self, "Ingen mapp", "Ingen fil eller mapp vald.")
            return

        output_dir = os.path.dirname(self.selected_files[0])
        QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    def apply_theme(self, theme_name):
        if theme_name not in THEMES:
            return

        self.current_theme = theme_name
        theme = THEMES[theme_name]

        stylesheet = f"""
            QMainWindow, QDialog {{
                background-color: {theme['window_bg']};
                color: {theme['text_color']};
            }}
            QGroupBox {{
                background-color: {theme['base_bg']};
                border: 1px solid {theme['border']};
                border-radius: 5px;
                margin-top: 1ex;
                color: {theme['text_color']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                background-color: {theme['base_bg']};
            }}
            QLabel, QCheckBox, QSpinBox, QDoubleSpinBox {{
                color: {theme['text_color']};
            }}
            QLineEdit, QTextEdit, QComboBox, QListWidget {{
                background-color: {theme['log_bg']};
                color: {theme['log_text']};
                border: 1px solid {theme['border']};
                border-radius: 3px;
                padding: 2px;
            }}
            QPushButton {{
                background-color: {theme['button_bg']};
                color: {theme['text_color']};
                border: 1px solid {theme['border']};
                padding: 5px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {theme['button_hover']};
            }}
            QProgressBar {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                background-color: {theme['progress_bg']};
                text-align: center;
                color: {theme['text_color']};
            }}
            QProgressBar::chunk {{
                background-color: {theme['progress_chunk']};
                width: 10px;
                margin: 0.5px;
            }}
            QMenuBar {{
                background-color: {theme['base_bg']};
                color: {theme['text_color']};
            }}
            QMenuBar::item {{
                background: transparent;
                color: {theme['text_color']};
            }}
            QMenuBar::item:selected {{
                background: {theme['button_hover']};
            }}
            QMenu {{
                background-color: {theme['base_bg']};
                color: {theme['text_color']};
                border: 1px solid {theme['border']};
            }}
            QMenu::item:selected {{
                background-color: {theme['button_hover']};
            }}
        """

        self.setStyleSheet(stylesheet)

        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.cancel_btn.setStyleSheet("background-color: #F44336; color: white;")

        for name, action in self.theme_actions.items():
            action.setChecked(name == theme_name)

        self.save_settings()


# ==========================================================
# APPLICATION START
# ==========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle(QStyleFactory.create("Fusion"))

    config_dir = get_config_dir()
    migrate_old_config(config_dir)

    window = SRTTranslatorWindow()
    window.show()

    sys.exit(app.exec())
