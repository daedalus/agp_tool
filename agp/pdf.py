from datetime import datetime

from fpdf import FPDF
from PIL import Image


def read_png_metadata(png_path):
    """Read text metadata from a PNG file.

    Returns a dict with keys: Title, Description, Author, Keywords,
    CreationTime, Software, Copyright, License.  Values are ``None``
    when the field is absent in the PNG.
    """
    img = Image.open(png_path)
    info = img.info or {}
    fields = [
        "Title",
        "Description",
        "Author",
        "Keywords",
        "CreationTime",
        "Software",
        "Copyright",
        "License",
    ]
    return {field: info.get(field) for field in fields}


def _build_xmp(meta):
    """Return an XMP XML string for Copyright and License, or None if both absent."""
    rights = meta.get("Copyright")
    license_url = meta.get("License")
    if not rights and not license_url:
        return None

    parts = []
    if rights:
        escaped = rights.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f"    <dc:rights><rdf:Alt>"
            f'<rdf:li xml:lang="x-default">{escaped}</rdf:li>'
            f"</rdf:Alt></dc:rights>"
        )
    if license_url:
        escaped_url = (
            license_url.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        parts.append(f'    <cc:license rdf:resource="{escaped_url}"/>')

    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:cc="http://creativecommons.org/ns#">'
        + "".join(parts)
        + "</rdf:Description>"
        "</rdf:RDF>"
        "</x:xmpmeta>"
    )


_DEFAULT_DPI = 72


def png_to_pdf(png_path, pdf_path):
    """Create a PDF from *png_path*, embedding it as an image and copying metadata.

    The PDF page size is set to exactly match the PNG pixel dimensions converted
    to PDF points using the PNG's stored DPI (from the ``pHYs`` chunk, defaulting
    to 72 DPI when not present).  The image is placed at the origin with no
    margins so it fills the page completely without cropping.

    Args:
        png_path (str): Path to the source PNG file.
        pdf_path (str): Destination path for the generated PDF.
    """
    img = Image.open(png_path)
    meta = read_png_metadata(png_path)

    dpi_info = img.info.get("dpi")
    if dpi_info is None:
        dpi_x = dpi_y = float(_DEFAULT_DPI)
    elif isinstance(dpi_info, (tuple, list)):
        dpi_x, dpi_y = float(dpi_info[0]), float(dpi_info[1])
    else:
        dpi_x = dpi_y = float(dpi_info)

    width_px, height_px = img.size
    # Convert pixel dimensions to PDF points (1 pt = 1/72 inch).
    width_pt = width_px * 72.0 / dpi_x
    height_pt = height_px * 72.0 / dpi_y

    # Always use portrait orientation so fpdf2 does not swap the format dimensions.
    pdf = FPDF(orientation="P", unit="pt", format=(width_pt, height_pt))
    pdf.set_margins(0, 0, 0)
    pdf.set_auto_page_break(False)
    pdf.add_page()

    if meta.get("Title"):
        pdf.set_title(meta["Title"])
    if meta.get("Author"):
        pdf.set_author(meta["Author"])
    if meta.get("Description"):
        pdf.set_subject(meta["Description"])
    if meta.get("Keywords"):
        pdf.set_keywords(meta["Keywords"])
    if meta.get("Software"):
        pdf.set_creator(meta["Software"])
    if meta.get("CreationTime"):
        try:
            dt = datetime.fromisoformat(meta["CreationTime"])
            pdf.set_creation_date(dt)
        except (ValueError, TypeError):
            pass

    xmp = _build_xmp(meta)
    if xmp:
        pdf.set_xmp_metadata(xmp)

    pdf.image(png_path, x=0, y=0, w=width_pt, h=height_pt)
    pdf.output(pdf_path)
