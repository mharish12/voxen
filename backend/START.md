# steps to start

```bash
# make sure local postgres is running on localhost:5432 with db tts
export TTS_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/tts

cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
