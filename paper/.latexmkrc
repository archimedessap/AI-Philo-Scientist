$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1';
$bibtex = 'bibtex';
$recorder = 1;   # generate .fls (file list)

# Keep synctex artifact around; latexmk cleans aggressively otherwise.
push @generated_exts, 'synctex.gz';

