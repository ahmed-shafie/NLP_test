# Banking NLU Assistant — Mobile (Flutter)

A cross-platform **Android + iOS** client for the Banking NLU brain. It talks to
the existing FastAPI backend over HTTP and supports a full **multi-turn money
transfer dialogue by text or voice**, in English and Arabic.

## Features

- **Chat UI** — type a message or **tap the mic** to speak.
- **Voice turns** — records audio, uploads it to `POST /conversation/voice`, shows
  the transcript, and **auto-plays the synthesized spoken reply**.
- **Text turns** — `POST /conversation/text`.
- **Live state** — amount / currency / recipient slots and the dialogue status
  (collecting → confirming → completed/cancelled) update each turn.
- **Multi-turn sessions** — the `session_id` is reused across turns; "New
  conversation" resets it.
- **Bilingual EN/AR** with right-to-left rendering for Arabic.
- **Settings** — configure the backend base URL, an optional API key (sent as
  `X-API-Key` when the backend has auth enabled), and the language; includes a
  "Test connection" button that hits `/health`.

## Project layout

```
lib/
  main.dart                  # app entry, theme
  models/conversation.dart   # response + slot models
  services/
    settings_service.dart    # persisted base URL / API key / language
    api_client.dart          # /conversation/text + /conversation/voice
  screens/
    chat_screen.dart         # the assistant UI (mic, bubbles, slots, status)
    settings_screen.dart     # backend configuration
```

## Configure the backend URL

The app defaults to `http://10.0.2.2:8000` — the Android emulator's alias for the
host machine's `localhost`. Change it in **Settings**:

- **Android emulator:** `http://10.0.2.2:8000`
- **iOS simulator:** `http://localhost:8000`
- **Physical device:** `http://<your-computer-LAN-IP>:8000` (or a deployed URL)

Run the backend first:

```bash
# from the repo root
pip install -r requirements.txt -r requirements-voice.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run / build

```bash
cd mobile
flutter pub get
flutter run                 # on a connected device/emulator

flutter build apk --release # Android APK  -> build/app/outputs/flutter-apk/
flutter build ios --release # iOS (requires macOS + Xcode)
```

## Permissions

- **Android:** `INTERNET`, `RECORD_AUDIO`, and `usesCleartextTraffic` (for plain
  `http://` during development) are declared in `AndroidManifest.xml`.
- **iOS:** `NSMicrophoneUsageDescription` and an ATS exception are set in
  `Info.plist`.

> For production, terminate TLS in front of the backend and serve it over
> `https://`, then remove the cleartext/ATS exceptions.
