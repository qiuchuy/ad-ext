#!/bin/bash

python test_lstm_eager.py

if [ $? -ne 0 ]; then
    exit 1
fi

python test_lstm_graph.py

if [ $? -ne 0 ]; then
    exit 1
fi
