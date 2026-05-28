Pillars of Text to Speech

---

**Pillar 1 — TTS engine setup.** You'll use XTTS v2 by Coqui, the best open-source voice-cloning TTS model. It runs on CPU (slow) or GPU (fast) and supports fine-tuning on a user's voice with just a few minutes of audio.---

**Pillar 2 — Voice training pipeline.** Training adapts the base XTTS model to a specific user's voice characteristics. Here's the complete flow from raw recordings to a trained model:---

**Pillar 3 & 4 — Voice profile and model weight management.** Every user gets a folder of their own. Here's the exact directory structure and database schema you'll maintain:---

**Pillar 5 — Audio generation and local setup.** Here's the inference flow (text → audio), and the exact local machine setup using Docker to keep everything isolated:Here's a summary of the full build plan to get started:

**Project folder structure:**

```
tts-project/
├── backend/         # FastAPI app
│   ├── main.py      # API routes
│   ├── trainer.py   # Fine-tune logic
│   ├── inference.py # TTS generation
│   ├── models.py    # SQLite models
│   └── preprocess.py
├── frontend/        # React + Vite
├── voices/          # Per-user audio + model weights
├── docker-compose.yml
└── requirements.txt
```

**Key Python packages:**

```
TTS==0.22.0       # Coqui XTTS v2
torch             # ML backbone
torchaudio        # Audio processing
fastapi           # API server
uvicorn           # ASGI server
librosa           # Audio preprocessing
soundfile         # .wav I/O
sqlalchemy        # DB ORM
```

**Recommended local build order:**

1. Set up FastAPI skeleton with SQLite/postgres(We will use postgress database, create the set up steps as well)
2. Write the audio preprocessing pipeline (`librosa`)
3. Get a basic XTTS inference working (no fine-tuning yet)
4. Add the fine-tuning job with the trainer API
5. Wire in the React frontend with `MediaRecorder`
6. Dockerize everything
