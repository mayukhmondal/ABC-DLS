#!/bin/bash
ls ../examples/*.csv.gz |awk -F "/" '{system ("zcat "$0"|head -n 6 > "substr($NF,1,length($NF)-3))}'
cat ../examples/Model.info |awk -F "/" '{print $NF}' |awk '{print substr($1,1,length($1)-3)"\t"$2}' > Model.info
#python --version

