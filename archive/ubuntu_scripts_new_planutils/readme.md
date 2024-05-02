# Ubuntu scripts for new plantutils image
We never go this quite working. This was meant to cope with changes in planutils.

Got to the point where we could install enhsp-2020, but the `.sif` was put in the working directory, rather than wherever planutils expected it.

```sh
source ubuntu_scripts/prepare_enhsp.sh
```
After the config_all.sh.