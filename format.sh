#!/usr/bin/zsh

docformatter --wrap-descriptions 72 --wrap-summaries 72 --recursive --in-place ml
black ml
docformatter --wrap-descriptions 72 --wrap-summaries 72 --recursive --in-place nn
black nn

echo "formatting done!"

