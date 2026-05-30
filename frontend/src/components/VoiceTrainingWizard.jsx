import { useCallback, useEffect, useRef, useState } from "react";
import { MIN_TRAINING_SENTENCES, TARGET_TRAINING_SENTENCES, TRAINING_SENTENCES } from "../trainingSentences";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";
const MIN_RECORD_SECONDS = 3;

function apiErrorMessage(errBody, fallback) {
  const detail = errBody?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  return fallback;
}

function audioBufferToWavBlob(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = audioBuffer.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  let offset = 0;
  writeString(offset, "RIFF");
  offset += 4;
  view.setUint32(offset, 36 + dataLength, true);
  offset += 4;
  writeString(offset, "WAVE");
  offset += 4;
  writeString(offset, "fmt ");
  offset += 4;
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint16(offset, numChannels, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, sampleRate * blockAlign, true);
  offset += 4;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, bitDepth, true);
  offset += 2;
  writeString(offset, "data");
  offset += 4;
  view.setUint32(offset, dataLength, true);
  offset += 4;

  const channelData = Array.from({ length: numChannels }, (_, ch) => audioBuffer.getChannelData(ch));
  for (let i = 0; i < audioBuffer.length; i += 1) {
    for (let ch = 0; ch < numChannels; ch += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[ch][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export default function VoiceTrainingWizard({ voiceId, onStatus, onTrainingComplete }) {
  const [readiness, setReadiness] = useState(null);
  const [samples, setSamples] = useState([]);
  const [recordingIndex, setRecordingIndex] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const recordTimerRef = useRef(null);

  const refresh = useCallback(async () => {
    if (!voiceId) return;
    const [rRes, sRes] = await Promise.all([
      fetch(`${API_BASE}/voices/${voiceId}/training-readiness`),
      fetch(`${API_BASE}/voices/${voiceId}/samples`),
    ]);
    if (rRes.ok) setReadiness(await rRes.json());
    if (sRes.ok) setSamples(await sRes.json());
  }, [voiceId]);

  useEffect(() => {
    refresh().catch((err) => onStatus(err.message));
  }, [voiceId, refresh, onStatus]);

  useEffect(() => {
    if (!job?.id || job.status === "completed" || job.status === "failed") return undefined;
    const timer = setInterval(async () => {
      const res = await fetch(`${API_BASE}/train/${job.id}`);
      if (!res.ok) return;
      const data = await res.json();
      setJob(data);
      if (data.status === "completed") {
        onStatus("Voice training completed. You can synthesize with your trained profile.");
        onTrainingComplete?.();
        refresh();
      } else if (data.status === "failed") {
        onStatus(`Training failed: ${data.error_msg || "unknown error"}`);
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [job, onStatus, onTrainingComplete, refresh]);

  async function uploadSentence(sentenceIndex, blob, filename) {
    const form = new FormData();
    form.append("file", blob, filename);
    form.append("transcript", TRAINING_SENTENCES[sentenceIndex]);
    const res = await fetch(`${API_BASE}/voices/${voiceId}/samples`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(apiErrorMessage(err, "Failed to upload training sample"));
    }
    await refresh();
    onStatus(`Recorded sentence ${sentenceIndex + 1} of ${TRAINING_SENTENCES.length}.`);
  }

  async function startRecording(sentenceIndex) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const preferredTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
    const selectedType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type));
    const rec = selectedType ? new MediaRecorder(stream, { mimeType: selectedType }) : new MediaRecorder(stream);
    chunksRef.current = [];
    rec.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    rec.onstop = async () => {
      try {
        setBusy(true);
        const recordedBlob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        const arrayBuffer = await recordedBlob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioContext.decodeAudioData(arrayBuffer);
        if (decoded.duration < MIN_RECORD_SECONDS) {
          onStatus(
            `Recording too short (${decoded.duration.toFixed(1)}s). Hold Record for at least ${MIN_RECORD_SECONDS} seconds.`,
          );
          return;
        }
        const wavBlob = audioBufferToWavBlob(decoded);
        await audioContext.close();
        await uploadSentence(sentenceIndex, wavBlob, `training_${sentenceIndex + 1}.wav`);
      } catch (err) {
        onStatus(err.message || "Recording failed");
      } finally {
        setBusy(false);
        if (recordTimerRef.current) {
          clearInterval(recordTimerRef.current);
          recordTimerRef.current = null;
        }
        setRecordSeconds(0);
        stream.getTracks().forEach((track) => track.stop());
      }
    };
    rec.start();
    mediaRecorderRef.current = rec;
    setRecordingIndex(sentenceIndex);
    setRecordSeconds(0);
    recordTimerRef.current = setInterval(() => {
      setRecordSeconds((s) => s + 1);
    }, 1000);
  }

  function stopRecording() {
    if (mediaRecorderRef.current && recordingIndex !== null) {
      if (recordSeconds < MIN_RECORD_SECONDS) {
        onStatus(`Keep recording… at least ${MIN_RECORD_SECONDS}s (now ${recordSeconds}s).`);
        return;
      }
      mediaRecorderRef.current.stop();
      setRecordingIndex(null);
      if (recordTimerRef.current) {
        clearInterval(recordTimerRef.current);
        recordTimerRef.current = null;
      }
    }
  }

  async function startTraining() {
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voice_id: voiceId, epochs: 50, batch_size: 2, learning_rate: 0.000005 }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to start training");
      }
      const data = await res.json();
      setJob(data);
      onStatus("Training started. On CPU this may take several minutes.");
    } finally {
      setBusy(false);
    }
  }

  const transcriptSamples = samples.filter((s) => s.transcript && s.transcript.trim());
  const readyCount = readiness?.with_transcript_count ?? transcriptSamples.length;

  return (
    <section className="training-wizard">
      <h2>3) Train My Voice</h2>
      <p className="hint">
        Read each sentence clearly in a quiet room. Hold <strong>Record</strong> for at least 3 seconds (6–15s is
        ideal). Train at least {MIN_TRAINING_SENTENCES} sentences; {TARGET_TRAINING_SENTENCES} is recommended.
      </p>
      <p className="progress-bar">
        Training progress: {readyCount} / {TARGET_TRAINING_SENTENCES} sentences with transcripts
        {readiness?.ready ? " — ready to train" : ""}
      </p>

      <ol className="sentence-list">
        {TRAINING_SENTENCES.map((sentence, index) => {
          const done = transcriptSamples.some((s) => s.transcript?.trim() === sentence);
          return (
            <li key={sentence} className={done ? "sentence-done" : ""}>
              <p className="sentence-text">{sentence}</p>
              <div className="sentence-actions">
                {recordingIndex === index ? (
                  <>
                    <button type="button" onClick={stopRecording} disabled={busy}>
                      Stop ({recordSeconds}s)
                    </button>
                    {recordSeconds < MIN_RECORD_SECONDS && (
                      <span className="hint">min {MIN_RECORD_SECONDS}s</span>
                    )}
                  </>
                ) : (
                  <button type="button" onClick={() => startRecording(index).catch((e) => onStatus(e.message))} disabled={busy || !voiceId}>
                    Record
                  </button>
                )}
                <label className="upload-label">
                  Upload
                  <input
                    type="file"
                    accept=".wav,.mp3,audio/wav,audio/mpeg"
                    hidden
                    disabled={busy || !voiceId}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      setBusy(true);
                      uploadSentence(sentenceIndex, file, file.name)
                        .catch((err) => onStatus(err.message))
                        .finally(() => {
                          setBusy(false);
                          e.target.value = "";
                        });
                    }}
                  />
                </label>
                {done && <span className="badge">Recorded</span>}
              </div>
            </li>
          );
        })}
      </ol>

      <button
        type="button"
        className="primary"
        disabled={!voiceId || busy || readyCount < MIN_TRAINING_SENTENCES}
        onClick={() => startTraining().catch((e) => onStatus(e.message))}
      >
        Start Voice Training
      </button>
      {job && (
        <p className="job-status">
          Training job: {job.status}
          {job.error_msg ? ` — ${job.error_msg}` : ""}
        </p>
      )}
    </section>
  );
}
