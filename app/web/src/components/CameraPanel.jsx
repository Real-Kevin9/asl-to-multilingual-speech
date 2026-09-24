export default function CameraPanel({ videoRef, bboxPct, statusText, holdProgress, holdingLabel }) {
  const mirrored =
    bboxPct && bboxPct.length === 4
      ? [100 - bboxPct[2], bboxPct[1], 100 - bboxPct[0], bboxPct[3]]
      : null;

  const boxStyle =
    mirrored && mirrored.length === 4
      ? {
          left: `${mirrored[0]}%`,
          top: `${mirrored[1]}%`,
          width: `${mirrored[2] - mirrored[0]}%`,
          height: `${mirrored[3] - mirrored[1]}%`,
        }
      : null;

  return (
    <section className="camera-panel">
      <div className="camera-panel__frame">
        <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
        {boxStyle && <div className="hand-box" style={boxStyle} />}
        <div className="camera-hold">
          <span>{holdingLabel ? `Holding ${holdingLabel}` : "Waiting for sign"}</span>
          <div className="hold-bar">
            <div className="hold-bar__fill" style={{ width: `${Math.round(holdProgress * 100)}%` }} />
          </div>
        </div>
        <div className="camera-status">{statusText}</div>
      </div>
    </section>
  );
}
