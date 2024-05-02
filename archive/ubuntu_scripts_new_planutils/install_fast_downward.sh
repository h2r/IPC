#!/bin/bash
apt-get update -y
# We may not need all of these
apt-get install git jq cmake g++ make -y
cd ../
git clone https://github.com/aibasel/downward.git
cd downward
./build.py