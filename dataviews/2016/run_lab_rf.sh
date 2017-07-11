#!/bin/sh

for file in *.csv
do
  ~/sorghum/random_forest/random_forest ds "$file" label ${file%.*} option regression > stdout.txt && mkdir -p ../../labCodeOutputs/${file%.*} && mv *.csv stdout.txt ../../labCodeOutputs/${file%.*}
done
