# 🌐 SRT Translator (English → Swedish)

A PyQt6-based GUI application that translates English subtitle files (.srt) to Swedish using Ollama or OpenAI-compatible APIs. Supports cloud API key rotation, rate limiting, custom styles, themes, and batch processing.

---

## 🔧 Features

- Translates .SRT files from English to Swedish
- Works with **Ollama** (local), **Ollama Cloud**, or **OpenAI-compatible endpoints**
- Supports up to 20 cloud API keys with automatic failover on HTTP 429
- Handles syntax like `[PANTING]` → `[FLÄMTAR]`
- Customizable translation style: *Formal*, *Natural* (recommended), or *Simple clear*
- Dark/Light/Dracula/Nord themes included
- Batch processing of multiple files/folders
- Configurable rate limiting (RPM and minimum delay)
- Saves translated subtitles as `.sv.srt` or `.swe.srt`

---

## 🛠️ Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

> **requirements.txt** contains:
> - `PyQt6`
> - `requests`

Also ensure you have:
- Python 3.8+
- An active Ollama server OR an API key from Ollama Cloud / OpenAI-compatible provider

---

## ▶️ How to Run

1. Save the script as `english_to_swedish_subtitle_translator_v7.py`
2. Make sure you're in the same directory as the file
3. Launch the app:

```bash
python english_to_swedish_subtitle_translator_v7.py
```

The first run will create a config folder (`~/.config/SRTTranslator` on Linux/macOS, or equivalent elsewhere).

---

## ⚙️ Configuration Options

### Connection Settings (via menu → Inställningar → Anslutningsinställningar):

| Field              | Example Value                     | Description                              |
|--------------------|-----------------------------------|------------------------------------------|
| Provider           | Local Ollama / Cloud / OpenAI     | Select your backend                      |
| Server URL         | `http://localhost:11434/api`      | For local Ollama                         |
| Cloud Base URL     | `https://ollama.com/api`          | For Ollama Cloud                         |
| OpenAI Compatible  | `http://localhost:8000/v1`        | For other LLM servers                    |
| API Keys           | Comma-separated or list format    | For cloud providers                      |

You can also set favorite models, output overwrite behavior, rate limits, etc., through the UI menus.

---

## 💡 Tips & Notes

- Use natural style for best results unless formal tone is required.
- If getting “rate limited”, increase min-delay RPM setting or wait before retrying.
- The tool preserves placeholders like `<PH_B1_0>` exactly — don’t modify them!
- Output filenames follow pattern: `{original_filename}.sv.srt` or `{original_filename}.swe.srt`

---

## 📁 File Structure Example

```
project-root/
├── Subtitle_translator_v7.py
├── requirements.txt
└── README.md
```

Place your `.srt` files anywhere — even nested folders are scanned when selecting directories.

---

## 🖥️ Screenshots (Optional)

Add screenshot images here if available:

![Screenshot showing main window](images/screenshot.png)

*(Replace path with actual image location in repo)*

---

## 🤝 Contributions Welcome!

Feel free to submit pull requests for bug fixes, new features, translations, or documentation improvements.

Let’s make subtitle translation easier! 🎬✨

---

## ℹ️ About This Tool

Built by [Your Name], open-source project under MIT License. Designed for translators, editors, and fans who want quick, accurate Swedish subtitles without manual typing.

Powered by modern large language models running locally or in the cloud.

---

## 🇸🇪 Svenska Version (Swedish Translation)

> **Detta program översätter engelska undertexter till svenska med hjälp av AI-modelle.**  
> Stöder Ollama lokalt, molntjänster samt OpenAI-kompatibla endpoints. Har automatisk omställning vid rate-limiting, flera nycklar, teman och anpassbara stilval. Installera所需 paket körs med `pip install -r requirements.txt`. Kör sedan Python-skriptet och följ menyn för att konfigurera inställningar.

---

✅ Ready to commit! Just replace any placeholders (like `[Your Name]`) and add screenshots/docs as needed. Would you like me to generate a version with icons, badges (e.g., PyPI status, license badge), or support for additional languages/styles next? I’m happy to extend it further!
``` 

This README includes bilingual content structured clearly for international contributors while keeping technical details precise. Let me know if you’d like enhancements such as Docker instructions, CLI options, or CI/CD integration notes!
# SRT.Subtitle.Translator
