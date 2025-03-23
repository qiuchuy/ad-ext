#!/bin/bash

python test_lstm.py

if [ $? -ne 0 ]; then
    exit 1
fi

