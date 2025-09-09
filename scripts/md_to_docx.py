#!/usr/bin/env python3
"""
Minimal Markdown (.md) → .docx builder (no external deps).

Supported:
- Headings: #, ##, ### mapped to Title, Heading1, Heading2.
- Paragraphs: plain text lines grouped by blank lines.
- Bullet-like lines starting with '- ' are kept with a leading '• '.
- Fonts: Body -> SimSun 12pt; Headings -> SimHei bold (Title 16pt, H1 14pt, H2 13pt).

Usage:
  python3 scripts/md_to_docx.py input.md output.docx
"""

from __future__ import annotations

import sys
from pathlib import Path
import zipfile
import datetime as dt


def _xml_header() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def _content_types() -> bytes:
    xml = _xml_header() + (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '</Types>'
    )
    return xml.encode('utf-8')


def _rels_root() -> bytes:
    xml = _xml_header() + (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '</Relationships>'
    )
    return xml.encode('utf-8')


def _core_props() -> bytes:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    xml = _xml_header() + (
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<dc:creator>Codex CLI</dc:creator>'
        '<cp:lastModifiedBy>Codex CLI</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        '</cp:coreProperties>'
    )
    return xml.encode('utf-8')


def _app_props() -> bytes:
    xml = _xml_header() + (
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Codex CLI</Application>'
        '<DocSecurity>0</DocSecurity>'
        '<ScaleCrop>false</ScaleCrop>'
        '<HeadingPairs><vt:vector size="2" baseType="variant">'
        '<vt:variant><vt:lpstr>Title</vt:lpstr></vt:variant>'
        '<vt:variant><vt:i4>1</vt:i4></vt:variant>'
        '</vt:vector></HeadingPairs>'
        '<TitlesOfParts><vt:vector size="1" baseType="lpstr">'
        '<vt:lpstr>Document</vt:lpstr>'
        '</vt:vector></TitlesOfParts>'
        '</Properties>'
    )
    return xml.encode('utf-8')


def _styles() -> bytes:
    # Define defaults and key styles (Normal, Title, Heading1, Heading2)
    xml = _xml_header() + (
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults>'
        '  <w:rPrDefault>'
        '    <w:rPr>'
        '      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="SimSun"/>'
        '      <w:sz w:val="24"/>'
        '      <w:szCs w:val="24"/>'
        '    </w:rPr>'
        '  </w:rPrDefault>'
        '  <w:pPrDefault><w:pPr/></w:pPrDefault>'
        '</w:docDefaults>'

        # Normal
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '  <w:name w:val="Normal"/>'
        '  <w:qFormat/>'
        '  <w:rPr>'
        '    <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="SimSun"/>'
        '    <w:sz w:val="24"/>'
        '    <w:szCs w:val="24"/>'
        '  </w:rPr>'
        '</w:style>'

        # Title
        '<w:style w:type="paragraph" w:styleId="Title">'
        '  <w:name w:val="Title"/>'
        '  <w:basedOn w:val="Normal"/>'
        '  <w:next w:val="Normal"/>'
        '  <w:qFormat/>'
        '  <w:rPr>'
        '    <w:b/>'
        '    <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/>'
        '    <w:sz w:val="32"/>'
        '    <w:szCs w:val="32"/>'
        '  </w:rPr>'
        '</w:style>'

        # Heading1
        '<w:style w:type="paragraph" w:styleId="Heading1">'
        '  <w:name w:val="Heading 1"/>'
        '  <w:basedOn w:val="Normal"/>'
        '  <w:next w:val="Normal"/>'
        '  <w:uiPriority w:val="9"/>'
        '  <w:qFormat/>'
        '  <w:rPr>'
        '    <w:b/>'
        '    <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/>'
        '    <w:sz w:val="28"/>'
        '    <w:szCs w:val="28"/>'
        '  </w:rPr>'
        '</w:style>'

        # Heading2
        '<w:style w:type="paragraph" w:styleId="Heading2">'
        '  <w:name w:val="Heading 2"/>'
        '  <w:basedOn w:val="Normal"/>'
        '  <w:next w:val="Normal"/>'
        '  <w:uiPriority w:val="9"/>'
        '  <w:qFormat/>'
        '  <w:rPr>'
        '    <w:b/>'
        '    <w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/>'
        '    <w:sz w:val="26"/>'
        '    <w:szCs w:val="26"/>'
        '  </w:rPr>'
        '</w:style>'
        '</w:styles>'
    )
    return xml.encode('utf-8')


def _escape(t: str) -> str:
    return (t.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))


def _p_xml(text: str, style: str | None = None) -> str:
    # Split on explicit line breaks
    runs = []
    for part in text.split('\n'):
        runs.append(f'<w:r><w:t xml:space="preserve">{_escape(part)}</w:t></w:r>')
        runs.append('<w:r><w:br/></w:r>')
    if runs:
        runs.pop()  # remove trailing br
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
    return f'<w:p>{ppr}'+''.join(runs)+'</w:p>'


def _build_document(paragraphs: list[tuple[str, str | None]]) -> bytes:
    # paragraphs: list of (text, styleId)
    body = []
    for text, style in paragraphs:
        body.append(_p_xml(text, style))
    xml = _xml_header() + (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:body>' + ''.join(body) + '</w:body></w:document>'
    )
    return xml.encode('utf-8')


def parse_md(md_text: str) -> list[tuple[str, str | None]]:
    paragraphs: list[tuple[str, str | None]] = []
    buf: list[str] = []
    current_style: str | None = None

    def flush_paragraph():
        nonlocal buf, current_style
        if not buf:
            return
        text = '\n'.join(buf).strip('\n')
        paragraphs.append((text, current_style))
        buf = []
        current_style = None

    for line in md_text.splitlines():
        if line.startswith('# '):
            flush_paragraph()
            current_style = 'Title'
            buf = [line[2:].strip()]
            flush_paragraph()
            continue
        if line.startswith('## '):
            flush_paragraph()
            current_style = 'Heading1'
            buf = [line[3:].strip()]
            flush_paragraph()
            continue
        if line.startswith('### '):
            flush_paragraph()
            current_style = 'Heading2'
            buf = [line[4:].strip()]
            flush_paragraph()
            continue
        if not line.strip():
            flush_paragraph()
            continue
        # Bullet heuristic
        if line.startswith('- '):
            flush_paragraph()
            current_style = None
            buf = ['• ' + line[2:].strip()]
            flush_paragraph()
            continue
        # Normal paragraph line
        if not buf:
            current_style = None
        buf.append(line)
    flush_paragraph()
    return paragraphs


def md_to_docx(md_path: Path, docx_path: Path) -> None:
    text = md_path.read_text(encoding='utf-8')
    paras = parse_md(text)
    with zipfile.ZipFile(docx_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _content_types())
        z.writestr('_rels/.rels', _rels_root())
        z.writestr('docProps/core.xml', _core_props())
        z.writestr('docProps/app.xml', _app_props())
        z.writestr('word/styles.xml', _styles())
        z.writestr('word/document.xml', _build_document(paras))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('Usage: md_to_docx.py input.md output.docx', file=sys.stderr)
        return 2
    in_path = Path(argv[0]).expanduser().resolve()
    out_path = Path(argv[1]).expanduser().resolve()
    if not in_path.exists():
        print(f'Input not found: {in_path}', file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_to_docx(in_path, out_path)
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

