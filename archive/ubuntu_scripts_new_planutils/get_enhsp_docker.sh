#!/bin/bash
docker pull aiplanning/planutils
docker run --privileged -v /etc/localtime:/etc/localtime -v /home/michael/repos/OO-Scoping-IPC:/scoping_supplement/ -it aiplanning/planutils bash
# docker run --privileged -v /etc/localtime:/etc/localtime -v /home/ubuntu/scoping_supplement:/scoping_supplement/ -it aiplanning/planutils bash
echo "loaded planutils container"