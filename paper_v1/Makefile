MAIN = main
DOCKER_IMG = latex-small

.PHONY: all clean

all:
	docker run --rm -v $(CURDIR):/workdir -w /workdir -u "$(shell id -u):$(shell id -g)" $(DOCKER_IMG) sh -c \
	"pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex && \
	bibtex $(MAIN) && \
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex && \
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex && \
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.bak *.dvi $(MAIN).pdf
