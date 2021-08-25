#! /usr/bin/bash

mv README.md README.md.0
jupyter nbconvert dOCT_drusen_example.ipynb --to markdown
jupyter nbconvert dOCT_drusen_example.ipynb --to pdf
mv dOCT_drusen_example.md README.md
mv dOCT_drusen_example.pdf README.pdf
