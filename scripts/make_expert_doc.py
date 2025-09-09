#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path
from datetime import datetime


def latest_top10_dir(base: Path) -> Path | None:
    cands = sorted(base.glob('top10_*'))
    return cands[-1] if cands else None


def load_summary(dirpath: Path):
    summ = dirpath / 'summary.csv'
    rows = []
    if not summ.exists():
        return rows
    with summ.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def extract_texts(theory: dict) -> dict:
    """Extract main sections with robust fallbacks."""
    out = {}
    out['name'] = theory.get('name') or theory.get('theory_name') or 'Unnamed Theory'
    out['summary'] = theory.get('summary') or theory.get('description') or ''

    # Core principles
    cps = theory.get('core_principles') or theory.get('core_assumptions') or []
    if isinstance(cps, dict):
        cps = [str(cps)]
    out['core_principles'] = []
    for item in cps:
        if isinstance(item, dict):
            p = item.get('principle') or item.get('axiom') or item.get('statement') or ''
            if p:
                out['core_principles'].append(p)
        elif isinstance(item, str):
            out['core_principles'].append(item)

    # Formalism
    formalism = theory.get('formalism') or theory.get('mathematical_formalism') or {}
    parts = []
    if isinstance(formalism, dict):
        if formalism.get('axioms'):
            ax = formalism['axioms']
            if isinstance(ax, list):
                parts.append('Axioms: ' + '; '.join([a.get('axiom') if isinstance(a, dict) else str(a) for a in ax]))
        if formalism.get('equations'):
            eq = formalism['equations']
            if isinstance(eq, dict):
                eqs = []
                for k, v in eq.items():
                    eqs.append(f"{k}: {v}")
                parts.append('Equations: ' + '; '.join(eqs))
        if formalism.get('comparison_with_sqm'):
            cmpv = formalism['comparison_with_sqm']
            if isinstance(cmpv, list):
                parts.append('Comparison with SQM: ' + '; '.join([str(x.get('aspect', '')) for x in cmpv if isinstance(x, dict)]))
    elif isinstance(formalism, str):
        parts.append(formalism)
    out['formalism'] = '\n'.join(parts)

    # Predictions & verifiability
    pv = theory.get('predictions_and_verifiability') or theory.get('empirical_predictions') or {}
    pred_lines = []
    if isinstance(pv, dict):
        if 'deviations_from_sqm' in pv and isinstance(pv['deviations_from_sqm'], list):
            for d in pv['deviations_from_sqm']:
                if isinstance(d, dict):
                    pred_lines.append(d.get('prediction_name') or d.get('description') or '')
    elif isinstance(pv, list):
        pred_lines.extend([str(x) for x in pv])
    out['predictions'] = [p for p in pred_lines if p]

    # Measurement/Philosophy (optional)
    phil = theory.get('philosophy') or {}
    meas = ''
    if isinstance(phil, dict):
        meas = phil.get('measurement') or phil.get('epistemology') or ''
        if isinstance(meas, dict):
            meas = json.dumps(meas, ensure_ascii=False)
    out['measurement'] = meas

    # Generation meta (from metadata)
    gi = theory.get('metadata', {}).get('generation_info', {}) if isinstance(theory.get('metadata'), dict) else {}
    llm = gi.get('llm_model', {}) if isinstance(gi, dict) else {}
    out['model_src'] = llm.get('model_source') or ''
    out['model_name'] = llm.get('model_name') or ''

    return out


def try_make_docx(cards: list, out_path: Path) -> bool:
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except Exception:
        return False

    doc = Document()
    doc.add_heading('Top-10 Generated Quantum Theories — Expert Review Pack', level=0)
    p = doc.add_paragraph()
    p.add_run('Generated on: ').bold = True
    p.add_run(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    doc.add_paragraph('Instructions: For each theory card, review the summary and key fields, then fill the scoring section.').italic = True

    for i, c in enumerate(cards, 1):
        doc.add_page_break()
        doc.add_heading(f"#{i} {c['name']}", level=1)
        if c.get('model_src') or c.get('model_name'):
            doc.add_paragraph(f"Model: {c.get('model_src','')}/{c.get('model_name','')}")

        if c.get('summary'):
            doc.add_heading('Summary', level=2)
            doc.add_paragraph(c['summary'])

        if c.get('core_principles'):
            doc.add_heading('Core Principles', level=2)
            for cp in c['core_principles']:
                doc.add_paragraph(cp, style='List Bullet')

        if c.get('formalism'):
            doc.add_heading('Formalism (key points)', level=2)
            doc.add_paragraph(c['formalism'])

        if c.get('predictions'):
            doc.add_heading('Predictions & Verifiability', level=2)
            for pr in c['predictions']:
                doc.add_paragraph(pr, style='List Number')

        if c.get('measurement'):
            doc.add_heading('Measurement Treatment / Philosophy (brief)', level=2)
            doc.add_paragraph(c['measurement'])

        # Scoring section
        doc.add_heading('Expert Scoring (0–10)', level=2)
        grid = [
            'Mathematical Rigor', 'Testability', 'Ontological Clarity', 'Internal Coherence', 'Overall'
        ]
        for field in grid:
            run = doc.add_paragraph().add_run(f"{field}: __/10    ")
            run.bold = True
        doc.add_paragraph('Comments:')
        doc.add_paragraph('\n' * 2)

    doc.save(out_path)
    return True


def make_rtf(cards: list, out_path: Path):
    # Simple RTF container with plain text and separators (ASCII-safe)
    lines = []
    lines.append('{\\rtf1\\ansi\\deff0')
    lines.append('\\b Top-10 Generated Quantum Theories — Expert Review Pack\\b0\\par')
    lines.append(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\\par')
    lines.append('Instructions: For each theory card, review the summary and key fields, then fill the scoring section.\\par')
    lines.append('\\par')

    def esc(s: str) -> str:
        # very simple escaping for backslashes and braces
        return s.replace('\\', r'\\').replace('{', r'\{').replace('}', r'\}')

    for i, c in enumerate(cards, 1):
        lines.append('\\par')
        lines.append('\\pard\\qr____________________________________________________________\\par')
        lines.append('\\pard')
        lines.append(f'\\b #{i} {esc(c["name"])}\\b0\\par')
        if c.get('model_src') or c.get('model_name'):
            lines.append(f'Model: {esc(c.get("model_src",""))}/{esc(c.get("model_name",""))}\\par')
        if c.get('summary'):
            lines.append('\\b Summary\\b0\\par')
            lines.append(esc(c['summary']) + '\\par')
        if c.get('core_principles'):
            lines.append('\\b Core Principles\\b0\\par')
            for cp in c['core_principles']:
                lines.append('• ' + esc(cp) + '\\par')
        if c.get('formalism'):
            lines.append('\\b Formalism (key points)\\b0\\par')
            lines.append(esc(c['formalism']) + '\\par')
        if c.get('predictions'):
            lines.append('\\b Predictions & Verifiability\\b0\\par')
            for pr in c['predictions']:
                lines.append('- ' + esc(pr) + '\\par')
        if c.get('measurement'):
            lines.append('\\b Measurement Treatment / Philosophy (brief)\\b0\\par')
            lines.append(esc(c['measurement']) + '\\par')
        # scoring block
        lines.append('\\b Expert Scoring (0–10)\\b0\\par')
        for f in ['Mathematical Rigor', 'Testability', 'Ontological Clarity', 'Internal Coherence', 'Overall']:
            lines.append(f'{f}: __/10\\par')
        lines.append('Comments:\\par\\par')

    lines.append('}')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    base = Path('expert_scoring')
    topdir = latest_top10_dir(base)
    if not topdir:
        print('No expert_scoring/top10_* directory found. Please export top-10 first.')
        return
    summary = load_summary(topdir)
    if not summary:
        print('summary.csv not found in', topdir)
        return

    # Build cards from JSON files
    cards = []
    for row in summary:
        jf = topdir / row['file']
        if not jf.exists():
            # try absolute fallback
            jf = Path(topdir) / Path(row['file']).name
        try:
            obj = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        c = extract_texts(obj)
        # prefer summary.csv model info if present
        c['model_src'] = row.get('model_src') or c.get('model_src')
        c['model_name'] = row.get('model_name') or c.get('model_name')
        cards.append(c)

    out_docx = topdir / 'top10_expert_review.docx'
    out_rtf = topdir / 'top10_expert_review.rtf'

    if try_make_docx(cards, out_docx):
        print('Wrote', out_docx)
    else:
        make_rtf(cards, out_rtf)
        print('Wrote', out_rtf)


if __name__ == '__main__':
    main()

