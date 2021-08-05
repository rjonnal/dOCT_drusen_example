#! /usr/bin/bash

mv README.md README.md.0
jupyter nbconvert dOCT_drusen_example.ipynb --to markdown
mv dOCT_drusen_example.md README.md
