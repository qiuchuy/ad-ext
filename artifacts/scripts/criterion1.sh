#!/bin/bash

# 检查是否传入了 --log 参数
LOG_OUTPUT=false
if [[ "$1" == "--log" ]]; then
    LOG_OUTPUT=true
fi

# 执行 Python 脚本，并根据参数决定是否重定向到文件
if $LOG_OUTPUT; then
    python /root/AILang/artifacts/criterion1/testsoftmax.py --data_shape 5 7 >/root/AILang/artifacts/results/criterion1/softmax.log 2>&1
    python /root/AILang/artifacts/criterion1/testconvolution.py >/root/AILang/artifacts/results/criterion1/convolution.log 2>&1
    python /root/AILang/artifacts/criterion1/testmatmul.py >/root/AILang/artifacts/results/criterion1/matmul.log 2>&1
else
    python /root/AILang/artifacts/criterion1/testsoftmax.py --data_shape 5 7
    python /root/AILang/artifacts/criterion1/testconvolution.py
    python /root/AILang/artifacts/criterion1/testmatmul.py
fi
