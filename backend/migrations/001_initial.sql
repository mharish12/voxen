CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS voice_profiles (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(120) NOT NULL,
    language VARCHAR(16) NOT NULL DEFAULT 'en',
    sample_count INTEGER NOT NULL DEFAULT 0,
    model_path TEXT,
    status VARCHAR(32) NOT NULL DEFAULT 'ready',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS voice_samples (
    id VARCHAR(36) PRIMARY KEY,
    voice_id VARCHAR(36) REFERENCES voice_profiles(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    transcript TEXT,
    duration_seconds FLOAT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TYPE training_status AS ENUM ('queued', 'running', 'failed', 'completed');

CREATE TABLE IF NOT EXISTS training_jobs (
    id VARCHAR(36) PRIMARY KEY,
    voice_id VARCHAR(36) REFERENCES voice_profiles(id) ON DELETE CASCADE,
    status training_status NOT NULL DEFAULT 'queued',
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    epochs INTEGER NOT NULL DEFAULT 0,
    loss FLOAT,
    error_msg TEXT,
    idempotency_key VARCHAR(128) UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trained_models (
    id VARCHAR(36) PRIMARY KEY,
    voice_id VARCHAR(36) REFERENCES voice_profiles(id) ON DELETE CASCADE,
    model_version VARCHAR(64) NOT NULL,
    model_path TEXT NOT NULL,
    base_model VARCHAR(128) NOT NULL DEFAULT 'xtts_v2',
    hyperparameters TEXT,
    training_duration_seconds INTEGER,
    quality_score FLOAT,
    is_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS synthesis_requests (
    id VARCHAR(36) PRIMARY KEY,
    voice_id VARCHAR(36) REFERENCES voice_profiles(id) ON DELETE SET NULL,
    user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    input_text TEXT NOT NULL,
    language VARCHAR(16) NOT NULL DEFAULT 'en',
    output_path TEXT NOT NULL,
    model_version VARCHAR(64),
    latency_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_voice_profiles_user_id ON voice_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_samples_voice_id ON voice_samples(voice_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_voice_id ON training_jobs(voice_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_voice_id ON trained_models(voice_id);
CREATE INDEX IF NOT EXISTS idx_synthesis_requests_voice_id ON synthesis_requests(voice_id);
