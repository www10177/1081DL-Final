# Proposal  

## Enviroments  
* OS : Windows, macOS(might be OK) 
* latex engine: xelatex 
* GNU make 

## Make Commands 
* `make` : compile and get pdf file  
* `make open` : compile and open pdf file  
* `make view` : open pdf file only ( Do not compile)  
* `make clean` : remove  all temporary file(.aux, .log)  
* `make clean-pdf` : remove pdf only  
* `make clean-all`: remove all temporary file and pdf 
## How to edit 
Adding/Editing file in `body`directory, and add section and `\input{body/NEW_SECTION}` in Main.tex  
