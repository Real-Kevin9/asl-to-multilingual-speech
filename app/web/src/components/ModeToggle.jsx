export default function ModeToggle({ mode, onChange }) {
  return (
    <div className="mode-toggle" role="tablist" aria-label="Communication mode">
      <button
        type="button"
        role="tab"
        aria-selected={mode === "sign_to_speech"}
        className={mode === "sign_to_speech" ? "active" : ""}
        onClick={() => onChange("sign_to_speech")}
      >
        Sign → Speech
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "speech_to_sign"}
        className={mode === "speech_to_sign" ? "active" : ""}
        onClick={() => onChange("speech_to_sign")}
      >
        Speech → Sign
      </button>
    </div>
  );
}
