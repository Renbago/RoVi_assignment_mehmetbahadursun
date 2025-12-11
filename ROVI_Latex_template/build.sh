#!/bin/bash
# build.sh - Compile LaTeX to PDF
# Usage: ./build.sh

cd "$(dirname "$0")"

echo "Compiling LaTeX..."
pdflatex -interaction=nonstopmode Main_Thesis_File.tex
biber Main_Thesis_File
pdflatex -interaction=nonstopmode Main_Thesis_File.tex
pdflatex -interaction=nonstopmode Main_Thesis_File.tex

echo ""
echo "Done! Output: Main_Thesis_File.pdf"
