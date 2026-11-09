#!/bin/sh
set -eu
cd "$(dirname "$0")"
task_tex=${TASK_TECTONIC:-tectonic}
for file in main supplement main supplement main_blinded supplement_blinded; do
  "$task_tex" --untrusted --keep-intermediates --keep-logs "$file.tex"
done
