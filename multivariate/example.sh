#!/bin/bash -ex

# TEST is a virtualenv
. ../../TEST/bin/activate

python main.py $@
