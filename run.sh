#!/bin/bash

echo "Run handin 2"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plot" ]; then
  echo "Directory does not exist create it!"
  mkdir plot
fi


# Script for excercise 1
echo "Run the first script ..."
python3 NUR_handin2_ex1.py > NUR_handin2_ex1.txt

# Script for excercise 2
echo "Run the second script ..."
python3 NUR_handin2_ex2.py > NUR_handin2_ex2.txt


echo "Generating the pdf"

pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex


