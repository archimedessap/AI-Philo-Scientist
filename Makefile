.PHONY: figs tables paper fast clean

figs:
	python3 scripts/gen_figs.py paper/figs

tables:
	python3 scripts/gen_tables.py paper/tables
	python3 scripts/gen_registry_tables.py paper/tables

paper: figs tables
	cd paper && latexmk -pdf -interaction=nonstopmode -file-line-error ai-philo-crossAI-arxiv.tex

fast:
	cd paper && latexmk -pdf -interaction=nonstopmode -file-line-error -silent ai-philo-crossAI-arxiv.tex

clean:
	cd paper && latexmk -C
