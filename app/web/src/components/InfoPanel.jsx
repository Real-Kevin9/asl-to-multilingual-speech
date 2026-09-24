export default function InfoPanel({ title, children, accent = "default" }) {
  return (
    <section className={`info-panel info-panel--${accent}`}>
      <h2>{title}</h2>
      <div className="info-panel__body">{children}</div>
    </section>
  );
}
