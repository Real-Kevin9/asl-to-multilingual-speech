export default function ComingSoon({ onBack }) {
  return (
    <section className="coming-soon">
      <div className="coming-soon__card">
        <p className="eyebrow">Speech → Sign</p>
        <h2>Coming soon</h2>
        <p>
          This direction will let a hearing person speak into the microphone and show
          visual sign language for a deaf user. The pipeline is planned (Whisper speech
          recognition, English-to-gloss translation, and sign video clips) but is not
          built yet.
        </p>
        <ul>
          <li>Microphone capture and speech-to-text</li>
          <li>English and Nepali input</li>
          <li>Word-level sign clips with fingerspelling fallback</li>
        </ul>
        <button type="button" className="primary-btn" onClick={onBack}>
          Back to Sign → Speech
        </button>
      </div>
    </section>
  );
}
