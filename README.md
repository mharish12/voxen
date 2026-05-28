# Voice Training + Voice-Cloned TTS

This project provides:
- FastAPI backend for voice profile CRUD, sample upload, synthesis, and training jobs.
- React frontend for creating profiles, recording/uploading samples, and generating speech.
- PostgreSQL persistence for users, voices, samples, jobs, models, and synthesis logs.
- Docker Compose stack for local end-to-end execution.

## Quick Start

1. Start containers:
   - `docker compose up --build`
2. Open frontend:
   - `http://localhost:3000`
3. Backend docs:
   - `http://localhost:8000/docs`

## API Milestones

Phase 1:
- `POST /api/users`
- `POST /api/voices`
- `GET /api/voices`
- `DELETE /api/voices/{voice_id}`
- `POST /api/voices/{voice_id}/samples`
- `POST /api/synthesize`

Phase 2:
- `POST /api/train`
- `GET /api/train/{job_id}`
- `GET /api/voices/{voice_id}/models`

## Data Layout

- `voices/{user_id}/{voice_id}/samples/`
- `voices/{user_id}/{voice_id}/processed/`
- `voices/{user_id}/{voice_id}/models/{model_version}/`
- `voices/{user_id}/{voice_id}/outputs/`

## Notes

- XTTS model loading can be heavy in constrained environments; when unavailable, backend writes a silent WAV fallback for API-level testing.
- Training worker currently uses a placeholder checkpoint artifact and metadata pipeline so the queue/model registry flow can be validated end-to-end before full XTTS fine-tuning is plugged in.
