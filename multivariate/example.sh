#!/bin/bash -ex

# TEST is a virtualenv
. ../TEST/bin/activate

python2 main.py ../data/beijing/interpolate.csv bj_result.csv
