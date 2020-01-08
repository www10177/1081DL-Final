ifeq ($(OS),Windows_NT)
    OPEN = cmd \/C start 
    RM = cmd \/C del 
else
    RM = rm -f -r
	OPEN := open
endif
MAIN = Main
BODY = body
BODY_FILES = $(wildcard $(BODY)/*.tex)
REF = reference.bib


all : $(MAIN).pdf 

$(MAIN).pdf : $(MAIN).tex $(BODY_FILES) $(REF)
	xelatex $(MAIN)
	bibtex $(MAIN)
	xelatex $(MAIN)
	xelatex $(MAIN)


view: 
	$(OPEN) $(MAIN).pdf

open: all 
	$(OPEN) $(MAIN).pdf

clean:
	$(RM) *.log *.aux *.bbl *.blg

clean-pdf:
	$(RM) *.pdf

clean-all : clean clean-pdf
