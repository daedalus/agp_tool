"""Tests for agp.pdf: PNG-to-PDF conversion with metadata."""
import io
import datetime
import re
import tempfile
import os

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from agp.pdf import read_png_metadata, png_to_pdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(tmp_path, width=200, height=100, dpi=96, **text_fields):
    """Create a minimal PNG file and return its path."""
    img = Image.new("RGB", (width, height), color="white")
    pnginfo = PngInfo()
    for key, value in text_fields.items():
        pnginfo.add_text(key, value)
    path = str(tmp_path / "test.png")
    img.save(path, format="PNG", pnginfo=pnginfo, dpi=(dpi, dpi))
    return path


def _pdf_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# read_png_metadata
# ---------------------------------------------------------------------------

class TestReadPngMetadata:
    def test_all_fields_present(self, tmp_path):
        png = _make_png(
            tmp_path,
            Title="My Title",
            Description="My Description",
            Author="Jane Doe",
            Keywords="glucose cgm",
            CreationTime="2026-01-01T00:00:00",
            Software="AGP 1.0",
            Copyright="Copyright 2026",
            License="MIT",
        )
        meta = read_png_metadata(png)
        assert meta["Title"] == "My Title"
        assert meta["Description"] == "My Description"
        assert meta["Author"] == "Jane Doe"
        assert meta["Keywords"] == "glucose cgm"
        assert meta["CreationTime"] == "2026-01-01T00:00:00"
        assert meta["Software"] == "AGP 1.0"
        assert meta["Copyright"] == "Copyright 2026"
        assert meta["License"] == "MIT"

    def test_missing_fields_are_none(self, tmp_path):
        png = _make_png(tmp_path)  # no text fields
        meta = read_png_metadata(png)
        for field in ("Title", "Description", "Author", "Keywords",
                      "CreationTime", "Software", "Copyright", "License"):
            assert meta[field] is None, f"Expected None for {field}"

    def test_partial_fields(self, tmp_path):
        png = _make_png(tmp_path, Title="Partial", Author="Bob")
        meta = read_png_metadata(png)
        assert meta["Title"] == "Partial"
        assert meta["Author"] == "Bob"
        assert meta["Description"] is None
        assert meta["Keywords"] is None


# ---------------------------------------------------------------------------
# png_to_pdf — file creation
# ---------------------------------------------------------------------------

class TestPngToPdfCreatesFile:
    def test_creates_pdf_file(self, tmp_path):
        png = _make_png(tmp_path)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        assert os.path.exists(pdf_path)

    def test_pdf_has_nonzero_size(self, tmp_path):
        png = _make_png(tmp_path)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        assert os.path.getsize(pdf_path) > 0

    def test_pdf_starts_with_pdf_header(self, tmp_path):
        png = _make_png(tmp_path)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert data[:4] == b"%PDF", "File does not start with %PDF"


# ---------------------------------------------------------------------------
# png_to_pdf — metadata in PDF
# ---------------------------------------------------------------------------

class TestPngToPdfMetadata:
    def test_title_in_pdf(self, tmp_path):
        png = _make_png(tmp_path, Title="AGP Report")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"AGP Report" in data

    def test_author_in_pdf(self, tmp_path):
        png = _make_png(tmp_path, Author="Dr. Smith")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"Dr. Smith" in data

    def test_description_subject_in_pdf(self, tmp_path):
        png = _make_png(tmp_path, Description="Ambulatory Glucose Profile")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"Ambulatory Glucose Profile" in data

    def test_keywords_in_pdf(self, tmp_path):
        png = _make_png(tmp_path, Keywords="glucose diabetes")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"glucose diabetes" in data

    def test_software_creator_in_pdf(self, tmp_path):
        png = _make_png(tmp_path, Software="AGP v1.0")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"AGP v1.0" in data

    def test_copyright_in_xmp(self, tmp_path):
        png = _make_png(tmp_path, Copyright="Copyright 2026 Test")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"Copyright 2026 Test" in data

    def test_license_in_xmp(self, tmp_path):
        png = _make_png(tmp_path, License="MIT License")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert b"MIT License" in data

    def test_no_metadata_still_creates_valid_pdf(self, tmp_path):
        png = _make_png(tmp_path)  # no metadata
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        assert data[:4] == b"%PDF"

    def test_creation_time_in_pdf(self, tmp_path):
        png = _make_png(tmp_path, CreationTime="2026-01-15T10:30:00")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        # PDF creation date is encoded as D:YYYYMMDDHHmmSS
        assert b"D:20260115103000" in data

    def test_invalid_creation_time_does_not_raise(self, tmp_path):
        png = _make_png(tmp_path, CreationTime="not-a-date")
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)  # should not raise
        assert os.path.exists(pdf_path)


# ---------------------------------------------------------------------------
# CLI — --pdf flag
# ---------------------------------------------------------------------------

class TestCliPdfFlag:
    def test_pdf_defaults_false(self):
        from agp.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["dummy.xlsx"])
        assert args.pdf is False

    def test_pdf_set_when_passed(self):
        from agp.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["dummy.xlsx", "--pdf"])
        assert args.pdf is True
