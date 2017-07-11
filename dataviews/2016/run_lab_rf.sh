#!/bin/sh

cur=`pwd`
for file in *.csv
do
  pushd ~/sorghum/random_forest/
  ./random_forest ds "$cur"/"$file" label ${file%.*} option regression > stdout.txt && mkdir -p "$cur"/../../labCodeOutputs/${file%.*} && mv *.csv stdout.txt "$cur"/../../labCodeOutputs/${file%.*}
  popd
done
