#!/bin/sh

for file in *.csv
do
  ~/sorghum/random_forest/random_forest ds "$file" label ${file%.*} option regression
done
