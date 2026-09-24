"""Capture web dashboard mockup as PNG/PDF for the progress review."""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
HTML = HERE / "figures" / "ui_mockup.html"
OUT_PNG = HERE / "figures" / "web_dashboard_words.png"
OUT_PDF = HERE / "figures" / "web_dashboard_words.pdf"

cmd = [
    "google-chrome",
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--window-size=1120,720",
    f"--screenshot={OUT_PNG}",
    f"file://{HTML.resolve()}",
]
subprocess.run(cmd, check=True, timeout=30)
print("Wrote", OUT_PNG)

# Optional PDF via img2pdf if available
try:
    import img2pdf
    OUT_PDF.write_bytes(img2pdf.convert(str(OUT_PNG)))
    print("Wrote", OUT_PDF)
except ImportError:
    pass
