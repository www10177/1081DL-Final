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


all : $(MAIN).pdf 

$(MAIN).pdf : $(MAIN).tex $(BODY_FILES)
	xelatex $(MAIN)

view: 
	$(OPEN) $(MAIN).pdf

open: all 
	$(OPEN) $(MAIN).pdf

clean:
	$(RM) *.log *.aux 

clean-pdf:
	$(RM) *.pdf

clean-all : clean clean-pdf
