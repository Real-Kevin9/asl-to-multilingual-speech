import { useCallback, useEffect, useRef, useState } from "react";
import CameraPanel from "./CameraPanel.jsx";
import InfoPanel from "./InfoPanel.jsx";

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/sign-to-speech`;

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <strong className="stat__value">{value || "—"}</strong>
      {hint ? <span className="stat__hint">{hint}</span> : null}
    </div>
  );
}

export default function SignToSpeechDashboard() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const rafRef = useRef(null);
  const reconnectRef = useRef(null);
  const closingRef = useRef(false);
  const recordingRef = useRef(false);
  const detectingRef = useRef(false);
  const recognitionModeRef = useRef("alphabet");

  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [cameraError, setCameraError] = useState("");
  const [state, setState] = useState({
    current_label: "",
    confidence: 0,
    backend: "",
    assembled_gloss: "",
    holding_label: "",
    hold_progress: 0,
    bbox_pct: null,
  });
  const [models, setModels] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [nlpMode, setNlpMode] = useState("off");
  const [recognitionMode, setRecognitionMode] = useState("alphabet");

  useEffect(() => {
    recognitionModeRef.current = recognitionMode;
  }, [recognitionMode]);

  const connectRef = useRef(() => {});

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setCameraError("");
      ws.send(
        JSON.stringify({
          type: "configure",
          config: {
            backend: "auto",
            recognition_mode: recognitionModeRef.current,
            nlp: nlpMode,
            languages: ["en", "ne"],
          },
        }),
      );
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "ready") {
        setLoading(false);
        setModels(msg.models);
      } else if (msg.type === "state") {
        recordingRef.current = Boolean(msg.recording);
        setState((prev) => ({ ...prev, ...msg }));
        if (detectingRef.current) {
          detectingRef.current = false;
          setBusy(false);
        }
      } else if (msg.type === "result") {
        setResult(msg);
        setBusy(false);
      } else if (msg.type === "error") {
        setCameraError(msg.message || "WebSocket error");
        detectingRef.current = false;
        setBusy(false);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setLoading(true);
      detectingRef.current = false;
      setBusy(false);
      if (closingRef.current) {
        closingRef.current = false;
        return;
      }
      if (reconnectRef.current) return;
      reconnectRef.current = window.setTimeout(() => {
        reconnectRef.current = null;
        connectRef.current();
      }, 800);
    };

    ws.onerror = () => {
      setCameraError("Could not connect to the backend. Is the server running?");
    };

    return ws;
  }, [nlpMode, recognitionMode]);

  connectRef.current = connect;

  useEffect(() => {
    connect();
    return () => {
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
      closingRef.current = true;
      wsRef.current?.close();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [connect]);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setCameraError(err.message || "Camera permission denied");
      }
    }
    startCamera();
    return () => {
      const stream = videoRef.current?.srcObject;
      stream?.getTracks()?.forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let lastSent = 0;
    const tick = (ts) => {
      rafRef.current = requestAnimationFrame(tick);
      const video = videoRef.current;
      const ws = wsRef.current;
      if (!video || !ws || ws.readyState !== WebSocket.OPEN) return;
      if (video.readyState < 2) return;
      if (detectingRef.current || busy) return;
      if (recognitionModeRef.current === "wlasl" && !recordingRef.current) return;
      if (ts - lastSent < 120) return;
      lastSent = ts;

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", 0.85);
      ws.send(JSON.stringify({ type: "frame", image }));
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [connected, busy]);

  const sendAction = (type) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (type === "finish") {
      setBusy(true);
      setResult(null);
    }
    if (type === "start_sign") {
      recordingRef.current = true;
    }
    if (type === "cancel_sign") {
      recordingRef.current = false;
    }
    if (type === "commit_sign") {
      recordingRef.current = false;
      detectingRef.current = true;
      setBusy(true);
    }
    ws.send(JSON.stringify({ type }));
  };

  const playAudio = (lang) => {
    const info = result?.speech?.[lang];
    const url = result?.audio_urls?.[lang];
    if (!info) return;

    if (url) {
      const audio = new Audio(url);
      audio.play().catch(() => {
        if (lang === "en" && "speechSynthesis" in window && info.text) {
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(info.text);
          utterance.lang = "en-US";
          window.speechSynthesis.speak(utterance);
        }
      });
      return;
    }

    if (lang === "en" && "speechSynthesis" in window && info.text) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(info.text);
      utterance.lang = "en-US";
      window.speechSynthesis.speak(utterance);
    }
  };

  const statusText = connected
    ? loading
      ? "Loading models…"
      : busy
        ? "Detecting…"
        : "Live"
    : "Reconnecting…";

  return (
    <div className="dashboard">
      <aside className="dashboard__column dashboard__column--left">
        <InfoPanel title="Recognition" accent="cyan">
          <div className="toggle-row">
            <span className="stat__label">Input</span>
            <div className="mini-toggle" role="tablist" aria-label="Recognition mode">
              <button
                type="button"
                className={recognitionMode === "alphabet" ? "active" : ""}
                onClick={() => setRecognitionMode("alphabet")}
              >
                Alphabet
              </button>
              <button
                type="button"
                className={recognitionMode === "wlasl" ? "active" : ""}
                onClick={() => setRecognitionMode("wlasl")}
              >
                Words
              </button>
            </div>
          </div>
          <Stat
            label={recognitionMode === "wlasl" ? "Last word" : "Current sign"}
            value={state.current_label || "nothing"}
            hint={`${(state.confidence * 100).toFixed(0)}% confidence`}
          />
          <Stat label="Backend" value={state.backend || models?.recognition_backend} />
          <Stat
            label={recognitionMode === "wlasl" ? "Committed words" : "Committed letters"}
            value={(state.committed_labels || []).join(" ") || "—"}
          />
          {recognitionMode === "wlasl" && state.recording ? (
            <Stat
              label="Recording"
              value={`${Number(state.record_seconds || 0).toFixed(1)}s`}
              hint={`${state.record_frames || 0} frames`}
            />
          ) : null}
          {state.notice || state.error ? (
            <p className="muted">{state.error || state.notice}</p>
          ) : null}
        </InfoPanel>

        <InfoPanel title="Pipeline" accent="violet">
          <div className="toggle-row">
            <span className="stat__label">Webcam NLP</span>
            <div className="mini-toggle" role="tablist" aria-label="NLP mode">
              <button
                type="button"
                className={nlpMode === "off" ? "active" : ""}
                onClick={() => setNlpMode("off")}
              >
                Rules
              </button>
              <button
                type="button"
                className={nlpMode === "auto" ? "active" : ""}
                onClick={() => setNlpMode("auto")}
              >
                T5
              </button>
            </div>
          </div>
          <Stat label="NLP" value={models?.nlp_backend} />
          <Stat label="Emotion" value={models?.emotion_backend} />
          <Stat label="TTS" value={models?.tts_backend || "gtts_cached"} />
        </InfoPanel>
      </aside>

      <div className="dashboard__center">
        <CameraPanel
          videoRef={videoRef}
          bboxPct={state.bbox_pct}
          statusText={statusText}
          holdProgress={state.hold_progress || 0}
          holdingLabel={state.holding_label}
        />
        <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />

        <div className="action-row">
          <button type="button" className="ghost-btn" onClick={() => sendAction("clear")}>
            Clear
          </button>
          {recognitionMode === "wlasl" ? (
            <>
              {!state.recording ? (
                <button
                  type="button"
                  className="primary-btn"
                  disabled={busy || loading || !connected}
                  onClick={() => sendAction("start_sign")}
                >
                  Record sign
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="primary-btn"
                    disabled={busy}
                    onClick={() => sendAction("commit_sign")}
                  >
                    {busy ? "Detecting…" : "Stop & detect"}
                  </button>
                  <button
                    type="button"
                    className="ghost-btn"
                    onClick={() => sendAction("cancel_sign")}
                  >
                    Cancel
                  </button>
                </>
              )}
            </>
          ) : null}
          <button
            type="button"
            className="primary-btn"
            disabled={busy || !state.assembled_gloss}
            onClick={() => sendAction("finish")}
          >
            {busy ? "Processing…" : "Finish & speak"}
          </button>
        </div>
      </div>

      <aside className="dashboard__column dashboard__column--right">
        <InfoPanel title="Output" accent="green">
          <Stat label="Gloss" value={state.assembled_gloss || "—"} />
          <Stat label="English" value={result?.english || "—"} />
          <Stat
            label="Emotion"
            value={
              result
                ? `${result.emotion} (${Number(result.emotion_confidence || 0).toFixed(2)})`
                : "—"
            }
          />
          <Stat
            label="Latency"
            value={result?.timings ? `${Math.round(result.timings.total_ms)} ms` : "—"}
          />
        </InfoPanel>

        <InfoPanel title="Speech" accent="amber">
          {result?.audio_urls ? (
            Object.keys(result.audio_urls).map((lang) => (
              <div key={lang} className="speech-row">
                <span>{lang.toUpperCase()}</span>
                <button type="button" className="ghost-btn" onClick={() => playAudio(lang)}>
                  Play
                </button>
              </div>
            ))
          ) : (
            <p className="muted">
              {recognitionMode === "wlasl"
                ? "Record word signs, then press “Finish & speak”."
                : "Press “Finish & speak” after fingerspelling a word."}
            </p>
          )}
          {result?.notices?.length ? (
            <div className="notice-list">
              {result.notices.map((notice) => (
                <p key={notice} className="muted">
                  {notice}
                </p>
              ))}
            </div>
          ) : null}
        </InfoPanel>

        {cameraError ? (
          <InfoPanel title="Notice" accent="danger">
            <p className="muted">{cameraError}</p>
          </InfoPanel>
        ) : null}
      </aside>
    </div>
  );
}
