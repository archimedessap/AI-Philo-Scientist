#!/usr/bin/env python3
"""
Lightweight .docx → Markdown converter using only Python stdlib.

- Extracts paragraphs, basic bold/italic, line breaks, and hyperlinks.
- Heuristically maps Heading1..6 to #..###### when available.
- Detects simple lists and prefixes with '- ' (no nesting levels).

Usage:
  python3 scripts/docx_to_md.py path/to/file.docx [--out out_dir]
  python3 scripts/docx_to_md.py path/to/folder [--out out_dir]

Outputs a .md with the same basename in the output directory (default: alongside input).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}


def _load_xml_from_docx(zf: zipfile.ZipFile, name: str) -> ET.Element | None:
    try:
        with zf.open(name) as f:
            return ET.fromstring(f.read())
    except KeyError:
        return None


def _rels_map(zf: zipfile.ZipFile) -> dict[str, str]:
    rels: dict[str, str] = {}
    root = _load_xml_from_docx(zf, 'word/_rels/document.xml.rels')
    if root is None:
        return rels
    for rel in root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
        r_id = rel.attrib.get('Id')
        target = rel.attrib.get('Target')
        if r_id and target:
            # Targets can be relative paths; we only need raw URL-like targets
            rels[r_id] = target
    return rels


def _text_from_runs(elem: ET.Element) -> str:
    parts: list[str] = []
    for node in elem:
        tag = node.tag
        if tag.endswith('}r'):  # w:r
            rpr = node.find('w:rPr', NS)
            is_bold = rpr is not None and rpr.find('w:b', NS) is not None
            is_italic = rpr is not None and rpr.find('w:i', NS) is not None

            text_parts: list[str] = []
            for t in node.findall('.//w:t', NS):
                text_parts.append(t.text or '')
            # handle explicit line breaks inside run
            for br in node.findall('.//w:br', NS):
                text_parts.append('\n')
            txt = ''.join(text_parts)
            if not txt:
                continue
            if is_bold:
                txt = f"**{txt}**"
            if is_italic:
                txt = f"*{txt}*"
            parts.append(txt)
        elif tag.endswith('}hyperlink'):
            # hyperlink may wrap runs; capture displayed text and link target
            r_id = node.attrib.get(f'{{{NS["r"]}}}id')
            link_text = _text_from_runs(node)
            if r_id and link_text:
                parts.append(f"[{link_text}](__DOCX_LINK__:{r_id})")
            else:
                parts.append(link_text)
        elif tag.endswith('}br'):
            parts.append('\n')
        else:
            # Recurse into any unexpected container
            parts.append(_text_from_runs(node))
    return ''.join(parts)


def _heading_prefix(p: ET.Element) -> str:
    ppr = p.find('w:pPr', NS)
    if ppr is None:
        return ''
    pstyle = ppr.find('w:pStyle', NS)
    if pstyle is None:
        return ''
    val = pstyle.attrib.get(f'{{{NS["w"]}}}val', '')
    v = val.lower()
    if v.startswith('heading'):
        try:
            level = int(''.join(ch for ch in v if ch.isdigit()))
        except ValueError:
            level = 1
        level = max(1, min(level, 6))
        return '#' * level + ' '
    # Some docs use Title/Subtitle styles
    if v in ('title', 'subtitle'):
        return '# ' if v == 'title' else '## '
    return ''


def _is_list_paragraph(p: ET.Element) -> bool:
    ppr = p.find('w:pPr', NS)
    if ppr is None:
        return False
    return ppr.find('w:numPr', NS) is not None


def docx_to_markdown(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        root = _load_xml_from_docx(zf, 'word/document.xml')
        if root is None:
            raise RuntimeError('word/document.xml not found in docx')
        rels = _rels_map(zf)

        lines: list[str] = []
        for p in root.findall('.//w:p', NS):
            heading = _heading_prefix(p)
            is_list = _is_list_paragraph(p) and not heading
            text = _text_from_runs(p).strip()
            if not text:
                continue
            # Resolve hyperlink placeholders
            def resolve_links(s: str) -> str:
                out = []
                i = 0
                while i < len(s):
                    marker = '__DOCX_LINK__:'
                    idx = s.find(marker, i)
                    if idx == -1:
                        out.append(s[i:])
                        break
                    # find preceding '['
                    out.append(s[i:idx])
                    # find end ) after id
                    end = s.find(')', idx)
                    if end == -1:
                        out.append(s[idx:])
                        break
                    r_id = s[idx + len(marker): end]
                    url = rels.get(r_id, '')
                    # Replace the placeholder pattern with actual link target
                    out.append(url)
                    i = end
                return ''.join(out)

            text = resolve_links(text)

            if heading:
                lines.append(f"{heading}{text}")
            elif is_list:
                lines.append(f"- {text}")
            else:
                lines.append(text)
        # Ensure readable spacing between blocks
        markdown = '\n\n'.join(lines) + '\n'
        return markdown


def convert_path(p: Path, out_dir: Path | None = None) -> list[Path]:
    outputs: list[Path] = []
    if p.is_dir():
        for docx in sorted(p.rglob('*.docx')):
            outputs.extend(convert_path(docx, out_dir))
        return outputs
    if p.suffix.lower() != '.docx':
        return outputs
    md_text = docx_to_markdown(p)
    if out_dir is None:
        out_dir = p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (p.stem + '.md')
    out_path.write_text(md_text, encoding='utf-8')
    return [out_path]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description='Convert .docx to Markdown (no external deps).')
    ap.add_argument('input', type=str, help='Path to .docx file or directory')
    ap.add_argument('--out', type=str, default=None, help='Output directory for .md files')
    args = ap.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else None

    if not in_path.exists():
        ap.error(f'Input path not found: {in_path}')

    outputs = convert_path(in_path, out_dir)
    if not outputs:
        print('No .docx files converted.', file=sys.stderr)
        return 1
    for o in outputs:
        print(f'Wrote {o}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

