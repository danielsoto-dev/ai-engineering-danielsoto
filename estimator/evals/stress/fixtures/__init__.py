"""Synthetic attachment fixtures for the stress test (Block 3).

The PDFs are NOT committed — they are deterministic, so the runner regenerates
them on demand via ``build_pdfs.build_all``. Only the generator is in git.
"""

from evals.stress.fixtures.build_pdfs import ATTACHMENT_SIZES_KB, build_all, build_pdf

__all__ = ["ATTACHMENT_SIZES_KB", "build_all", "build_pdf"]
