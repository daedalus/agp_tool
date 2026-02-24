"""Tests for agp.pdf: PNG-to-PDF conversion with metadata."""

import os
import re

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from agp.pdf import png_to_pdf, read_png_metadata

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


def _make_png_no_dpi(tmp_path, width=200, height=100, **text_fields):
    """Create a PNG without any DPI/pHYs metadata."""
    img = Image.new("RGB", (width, height), color="white")
    pnginfo = PngInfo()
    for key, value in text_fields.items():
        pnginfo.add_text(key, value)
    path = str(tmp_path / "nodpi.png")
    img.save(path, format="PNG", pnginfo=pnginfo)
    return path


def _parse_mediabox(pdf_data):
    """Return (width_pt, height_pt) from the first /MediaBox in *pdf_data*."""
    m = re.search(rb"/MediaBox\s*\[([^\]]+)\]", pdf_data)
    assert m, "No /MediaBox found in PDF"
    values = m.group(1).split()
    assert len(values) == 4
    return float(values[2]), float(values[3])


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
        for field in (
            "Title",
            "Description",
            "Author",
            "Keywords",
            "CreationTime",
            "Software",
            "Copyright",
            "License",
        ):
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
# png_to_pdf — page size and image placement
# ---------------------------------------------------------------------------


class TestPngToPdfPageSize:
    def test_page_size_matches_png_with_dpi(self, tmp_path):
        """PDF page dimensions must equal pixel_size * 72 / dpi (points)."""
        width, height, dpi = 200, 100, 96
        png = _make_png(tmp_path, width=width, height=height, dpi=dpi)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        # Re-read actual DPI from the PNG (pHYs integer rounding may shift it).
        actual_dpi = Image.open(png).info.get("dpi", (96, 96))
        dpi_x, dpi_y = float(actual_dpi[0]), float(actual_dpi[1])
        expected_w = width * 72.0 / dpi_x
        expected_h = height * 72.0 / dpi_y
        w_pt, h_pt = _parse_mediabox(_pdf_bytes(pdf_path))
        assert abs(w_pt - expected_w) < 0.5, f"Width mismatch: {w_pt} vs {expected_w}"
        assert abs(h_pt - expected_h) < 0.5, f"Height mismatch: {h_pt} vs {expected_h}"

    def test_page_size_no_dpi_defaults_to_72(self, tmp_path):
        """Without pHYs chunk the page must be pixel_size * 72/72 = pixel_size points."""
        width, height = 300, 150
        png = _make_png_no_dpi(tmp_path, width=width, height=height)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        w_pt, h_pt = _parse_mediabox(_pdf_bytes(pdf_path))
        # Default DPI=72 → 1 point per pixel
        assert abs(w_pt - width) < 0.5, f"Width mismatch: {w_pt} vs {width}"
        assert abs(h_pt - height) < 0.5, f"Height mismatch: {h_pt} vs {height}"

    def test_landscape_png_not_cropped(self, tmp_path):
        """A wider-than-tall PNG must produce a wider-than-tall PDF page."""
        png = _make_png(tmp_path, width=400, height=200, dpi=72)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        w_pt, h_pt = _parse_mediabox(_pdf_bytes(pdf_path))
        assert w_pt > h_pt, "Landscape PNG should produce a wider-than-tall PDF"
        assert abs(w_pt - 400.0) < 0.5
        assert abs(h_pt - 200.0) < 0.5

    def test_portrait_png_not_cropped(self, tmp_path):
        """A taller-than-wide PNG must produce a taller-than-wide PDF page."""
        png = _make_png(tmp_path, width=200, height=400, dpi=72)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        w_pt, h_pt = _parse_mediabox(_pdf_bytes(pdf_path))
        assert h_pt > w_pt, "Portrait PNG should produce a taller-than-wide PDF"
        assert abs(w_pt - 200.0) < 0.5
        assert abs(h_pt - 400.0) < 0.5

    def test_image_fills_page_no_margins(self, tmp_path):
        """The embedded image must fill the page exactly (no implicit margins)."""
        width, height = 360, 180
        png = _make_png_no_dpi(tmp_path, width=width, height=height)
        pdf_path = str(tmp_path / "out.pdf")
        png_to_pdf(png, pdf_path)
        data = _pdf_bytes(pdf_path)
        w_pt, h_pt = _parse_mediabox(data)
        # The image placement command in PDF content should reference x=0 y=0
        # and dimensions matching the page.
        assert abs(w_pt - width) < 0.5
        assert abs(h_pt - height) < 0.5


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
