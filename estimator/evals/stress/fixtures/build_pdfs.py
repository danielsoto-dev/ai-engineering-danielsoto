"""Generate synthetic PDF attachments at calibrated sizes (Block 3).

We size by *extracted text length*, not file bytes — that's what actually
hits the prompt budget (and the ``MAX_ATTACHMENT_CHARS`` cap). Each PDF is
filled with distinctive, greppable sentences so ``MemoryDriftMetric`` can
check whether the attachment content reached the response summary.

Sizes (KB ≈ thousands of characters of body text):

    0   -> no attachment (absence; not a file)
    5   -> ~2 pages
    20  -> ~8 pages
    50  -> ~20 pages
    100 -> beyond MAX_ATTACHMENT_CHARS (60_000) -> exercises truncation

The files are deterministic, so they are regenerated rather than committed.
Run directly:  uv run python -m evals.stress.fixtures.build_pdfs
"""

from __future__ import annotations

from pathlib import Path

ATTACHMENT_SIZES_KB: tuple[int, ...] = (5, 20, 50, 100)

_FIXTURES_DIR = Path(__file__).resolve().parent
_OUT_DIR = _FIXTURES_DIR / "generated"

# A distinctive marker sentence so we can later detect the attachment's
# content in a response. Kept short and repeated to reach the target size.
_MARKER = (
    "Requirement REQ-{n:05d}: the Falcon platform must support encrypted "
    "audit trails and role-based access control for compliance. "
)


def _body_text(target_chars: int) -> str:
    """Build ~``target_chars`` of greppable, numbered requirement text."""
    parts: list[str] = []
    total = 0
    n = 0
    while total < target_chars:
        chunk = _MARKER.format(n=n)
        parts.append(chunk)
        total += len(chunk)
        n += 1
    return "".join(parts)[:target_chars]


def build_pdf(size_kb: int, out_dir: Path | None = None) -> Path:
    """Write ``attach_{size_kb}kb.pdf`` and return its path.

    ``size_kb`` is the approximate *text* length in thousands of characters.
    """
    from fpdf import FPDF

    out_dir = out_dir or _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target_chars = size_kb * 1000
    text = _body_text(target_chars)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    # multi_cell wraps the long string across lines/pages automatically.
    pdf.multi_cell(0, 5, text)

    path = out_dir / f"attach_{size_kb}kb.pdf"
    pdf.output(str(path))
    return path


def build_all(out_dir: Path | None = None) -> dict[int, Path]:
    """Regenerate every fixture. Returns ``{size_kb: path}``."""
    return {kb: build_pdf(kb, out_dir=out_dir) for kb in ATTACHMENT_SIZES_KB}


if __name__ == "__main__":  # pragma: no cover - manual regen
    for kb, path in build_all().items():
        print(f"{kb:>3} KB -> {path} ({path.stat().st_size} bytes)")
