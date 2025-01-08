#!/bin/bash

./run.sh $1 && zip -r $1.zip $1_* && rm -rf $1_*

