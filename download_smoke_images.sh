#!/usr/bin/env zsh

BASE_URL="https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html"

wget -r -l1 --no-parent -nd -A ".tgz" "${BASE_URL}" -P /home/sora/Documents/datasets/hpwren_data