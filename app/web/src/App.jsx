import { useState } from "react";
import SignToSpeechDashboard from "./components/SignToSpeechDashboard.jsx";
import ComingSoon from "./components/ComingSoon.jsx";
import ModeToggle from "./components/ModeToggle.jsx";

export default function App() {
  const [mode, setMode] = useState("sign_to_speech");

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Sign language · English · Nepali</p>
          <h1>ASL Multilingual Speech</h1>
        </div>
        <ModeToggle mode={mode} onChange={setMode} />
      </header>

      <main className="app-main">
        {mode === "sign_to_speech" ? (
          <SignToSpeechDashboard />
        ) : (
          <ComingSoon onBack={() => setMode("sign_to_speech")} />
        )}
      </main>
    </div>
  );
}
