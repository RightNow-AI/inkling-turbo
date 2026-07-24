#!/bin/bash
# Rebuild every figure. Requires tectonic and pdftoppm (poppler).
set -e
cd "$(dirname "$0")"
for f in fig1_latency fig2_correctness fig3_status; do
  echo "building $f"
  tectonic "$f.tex" >/dev/null 2>&1
  pdftoppm -png -r 220 "$f.pdf" "$f"
  mv "$f-1.png" "$f.png"
done
echo "done"
