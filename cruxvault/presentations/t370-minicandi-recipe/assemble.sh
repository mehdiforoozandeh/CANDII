#!/bin/zsh
# deck_src.html = head + template style + extra css + slides + data script + schematics + nav
HERE=${0:A:h}; P=$HERE/parts
cat $P/head.html $P/style.css $P/extra.css $P/slides.html $P/script.html $P/schematics.html $P/nav.html > $HERE/deck_src.html
