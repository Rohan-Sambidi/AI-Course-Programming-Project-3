#!/bin/bash

# Execute this script to kill all processes related to hw3!
#
# Warning: This will kill any process with hw3 in the string. We hope that its # only cse471 related, but if you suspect otherwise then do NOT run this script.

killall gzserver
killall gzclient
pkill -f "python .*hw3.*" --signal sigkill
pkill -f roscore
