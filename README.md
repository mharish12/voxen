# Voice Training + Voice-Cloned TTS

FastAPI backend and React frontend for personal voice profiles, guided training datasets, speaker-latent checkpoints, and XTTS-based synthesis.

## Quick Start

### Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

### Local development

1. PostgreSQL running (see `.env` / `backend/app/config.py`).
2. Backend (from `backend/`):

```bash
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. Frontend (from `frontend/`):

```bash
npm install && npm run dev
```

Requires **ffmpeg** for MP3 export and a working **XTTS v2** load on first request (warmup logs `xtts_warmup_success`).

## How to train and use your voice

1. **Create or select a user** — search by email or create a new account.
2. **Create a voice profile** — pick a name and language (e.g. `en`).
3. **Train My Voice** — in section 3, record or upload at least **8** guided sentences (12 recommended). Each upload includes an automatic transcript from the displayed sentence.
4. **Start Voice Training** — builds a real `speaker_latents.pth` checkpoint from your labeled samples (CPU-friendly; typically minutes, not hours).
5. **Synthesize** — leave **Auto** selected to use the promoted trained model. Use **Reference audio only** for zero-shot cloning without training.

### Tips for better quality

- Record 6–15 seconds per sentence in a quiet room, natural pace, no music.
- Complete all 12 training sentences when possible.
- After training, check the status banner for `trained` synthesis mode.

## Progress checklist (UI)

The app shows: user selected → voice created → training sentences ready → training completed → ready to synthesize.

## API overview

| Endpoint | Purpose |
|----------|---------|
| `POST /api/users` | Create user |
| `GET /api/users?email=` | Search users |
| `POST /api/voices` | Create voice profile |
| `GET /api/voices?user_id=` | List voices |
| `POST /api/voices/{id}/samples` | Upload sample (`file`, optional `transcript`) |
| `GET /api/voices/{id}/samples` | List samples |
| `GET /api/voices/{id}/training-readiness` | Dataset readiness |
| `POST /api/train` | Start training job |
| `GET /api/train/{job_id}` | Job status |
| `GET /api/voices/{id}/models` | List trained checkpoints |
| `POST /api/synthesize` | Generate MP3 (`model_version`, `use_reference_only`) |

Synthesis response header: `X-Synthesis-Mode: trained` or `reference`.

## Data layout

- `voices/{user_id}/{voice_id}/samples/`
- `voices/{user_id}/{voice_id}/processed/`
- `voices/{user_id}/{voice_id}/models/{model_version}/speaker_latents.pth`
- `voices/{user_id}/{voice_id}/outputs/`

## Training model

Training aggregates **XTTS speaker conditioning latents** from all transcript-labeled samples and saves them as `speaker_latents.pth`. Synthesis in trained mode calls `xtts.inference` with those latents instead of a single reference clip.

Full GPT fine-tuning of XTTS weights is not enabled by default (GPU-heavy); the latent checkpoint path is the supported CPU MVP.

## Validation checklist

- [ ] One good 10s reference sample → synthesize in reference mode → voice resembles speaker.
- [ ] 8+ training sentences with transcripts → `training-readiness` returns `ready: true`.
- [ ] `POST /api/train` → job `completed` → `speaker_latents.pth` on disk.
- [ ] Synthesize with Auto → logs / header `synthesis_mode=trained`.
- [ ] No trained model → reference mode still works with UI label.

## Notes

- If XTTS fails to load, synthesis returns **503** (no silent placeholder audio).
- Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for PyTorch 2.6+ checkpoint loading.
- Compatible stack: `TTS==0.22.x`, `transformers==4.41.2`.
