#!/bin/bash
declare -a arr=("do_apt_get.sh" "config_conda.sh" "prepare_enhsp.sh" "install_fast_downward.sh")
# Save the working directory. We will come back here after sourcing each script
cd /scoping_supplement/scoping
cwd=$(pwd)

for i in "${arr[@]}"
do
   echo "$i"
   source ubuntu_scripts/"$i"
   cd $cwd
done