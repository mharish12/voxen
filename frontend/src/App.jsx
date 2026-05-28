import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

export default function App() {
  const [email, setEmail] = useState("demo@example.com");
  const [user, setUser] = useState(null);
  const [matchedUsers, setMatchedUsers] = useState([]);
  const [voiceName, setVoiceName] = useState("My Voice");
  const [voices, setVoices] = useState([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [text, setText] = useState("Hello, this is a voice cloned speech test.");
  const [language, setLanguage] = useState("en");
  const [audioUrl, setAudioUrl] = useState("");
  const [status, setStatus] = useState("Idle");
  const [recording, setRecording] = useState(false);
  const [sampleFile, setSampleFile] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  function audioBufferToWavBlob(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const dataLength = audioBuffer.length * blockAlign;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    function writeString(offset, value) {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    }

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

  async function createUser() {
    const res = await fetch(`${API_BASE}/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password_hash: "dev-only-placeholder" }),
    });
    if (!res.ok && res.status !== 409) {
      throw new Error("Failed to create user");
    }
    if (res.status === 409) {
      setStatus("User already exists; use existing ID from backend.");
      return;
    }
    const data = await res.json();
    setUser(data);
    setMatchedUsers([]);
    setStatus(`User ready: ${data.id}`);
  }

  async function searchUsers() {
    if (!email.trim()) {
      setStatus("Enter an email to search.");
      return;
    }
    const res = await fetch(`${API_BASE}/users?email=${encodeURIComponent(email.trim())}`);
    if (!res.ok) {
      throw new Error("Failed to search users");
    }
    const data = await res.json();
    setMatchedUsers(data);
    if (!data.length) {
      setStatus("No existing user found. You can create a new user.");
      return;
    }
    setStatus(`Found ${data.length} user(s). Select one.`);
  }

  function selectUser(foundUser) {
    setUser(foundUser);
    setMatchedUsers([]);
    setStatus(`Using existing user: ${foundUser.id}`);
  }

  async function fetchVoices(userId) {
    const res = await fetch(`${API_BASE}/voices?user_id=${userId}`);
    if (!res.ok) {
      throw new Error("Failed to fetch voices");
    }
    const data = await res.json();
    setVoices(data);
    if (!selectedVoiceId && data.length) {
      setSelectedVoiceId(data[0].id);
    }
  }

  async function createVoice() {
    if (!user?.id) {
      setStatus("Create user first.");
      return;
    }
    const res = await fetch(`${API_BASE}/voices`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: user.id, name: voiceName, language }),
    });
    if (!res.ok) {
      throw new Error("Failed to create voice");
    }
    const data = await res.json();
    setSelectedVoiceId(data.id);
    await fetchVoices(user.id);
    setStatus("Voice profile created.");
  }

  async function uploadSample() {
    if (!selectedVoiceId || !sampleFile) {
      setStatus("Select a voice and provide a .wav or .mp3 sample.");
      return;
    }
    const form = new FormData();
    form.append("file", sampleFile);
    const res = await fetch(`${API_BASE}/voices/${selectedVoiceId}/samples`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to upload sample");
    }
    setStatus("Sample uploaded and processed.");
    if (user?.id) {
      await fetchVoices(user.id);
    }
  }

  async function synthesize() {
    if (!selectedVoiceId) {
      setStatus("Select a voice first.");
      return;
    }
    const res = await fetch(`${API_BASE}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: user?.id || null,
        voice_id: selectedVoiceId,
        text,
        language,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Synthesis failed");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    setAudioUrl(url);
    setStatus("Speech generated.");
  }

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const preferredTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
    const selectedType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type));
    const rec = selectedType ? new MediaRecorder(stream, { mimeType: selectedType }) : new MediaRecorder(stream);
    chunksRef.current = [];
    rec.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };
    rec.onstop = async () => {
      try {
        const recordedBlob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        const arrayBuffer = await recordedBlob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioContext.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWavBlob(decoded);
        await audioContext.close();
        const file = new File([wavBlob], "recorded.wav", { type: "audio/wav" });
        setSampleFile(file);
        setStatus("Recorded sample converted to WAV.");
      } catch (err) {
        setStatus("Recording conversion failed. Please upload a WAV or MP3 file manually.");
      } finally {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
    rec.start();
    mediaRecorderRef.current = rec;
    setRecording(true);
  }

  function stopRecording() {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  }

  useEffect(() => {
    if (user?.id) {
      fetchVoices(user.id).catch((err) => setStatus(err.message));
    }
  }, [user]);

  const selectedVoice = useMemo(
    () => voices.find((v) => v.id === selectedVoiceId),
    [voices, selectedVoiceId],
  );

  return (
    <div className="app">
      <h1>Voice Training + Voice Cloned TTS</h1>
      <p className="status">{status}</p>

      <section>
        <h2>1) User</h2>
        <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
        <button onClick={() => searchUsers().catch((err) => setStatus(err.message))}>Search User</button>
        <button onClick={() => createUser().catch((err) => setStatus(err.message))}>Create User</button>
        {user && <p>User ID: {user.id}</p>}
        {matchedUsers.length > 0 && (
          <select onChange={(e) => {
            const foundUser = matchedUsers.find((item) => item.id === e.target.value);
            if (foundUser) {
              selectUser(foundUser);
            }
          }}>
            <option value="">Select existing user</option>
            {matchedUsers.map((item) => (
              <option key={item.id} value={item.id}>
                {item.email} ({item.id})
              </option>
            ))}
          </select>
        )}
      </section>

      <section>
        <h2>2) Voice Profile</h2>
        <input value={voiceName} onChange={(e) => setVoiceName(e.target.value)} placeholder="Voice name" />
        <input value={language} onChange={(e) => setLanguage(e.target.value)} placeholder="Language" />
        <button onClick={() => createVoice().catch((err) => setStatus(err.message))}>Create Voice</button>
        <select value={selectedVoiceId} onChange={(e) => setSelectedVoiceId(e.target.value)}>
          <option value="">Select voice</option>
          {voices.map((voice) => (
            <option key={voice.id} value={voice.id}>
              {voice.name} ({voice.sample_count} samples)
            </option>
          ))}
        </select>
        {selectedVoice && <p>Selected voice status: {selectedVoice.status}</p>}
      </section>

      <section>
        <h2>3) Reference Audio</h2>
        <input
          type="file"
          accept=".wav,.mp3,audio/wav,audio/mpeg,audio/mp3"
          onChange={(e) => setSampleFile(e.target.files?.[0] || null)}
        />
        {!recording ? (
          <button onClick={() => startRecording().catch((err) => setStatus(err.message))}>Record</button>
        ) : (
          <button onClick={stopRecording}>Stop</button>
        )}
        <button onClick={() => uploadSample().catch((err) => setStatus(err.message))}>Upload Sample</button>
        {sampleFile && <p>Sample: {sampleFile.name}</p>}
      </section>

      <section>
        <h2>4) Synthesize</h2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />
        <button onClick={() => synthesize().catch((err) => setStatus(err.message))}>Generate Speech</button>
        {audioUrl && (
          <div>
            <audio controls src={audioUrl} />
            <a href={audioUrl} download="speech.wav">
              Download WAV
            </a>
          </div>
        )}
      </section>
    </div>
  );
}
