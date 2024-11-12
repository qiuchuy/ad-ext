#!/bin/bash

# 检查是否传入了 --log 参数
LOG_OUTPUT=false
if [[ "$1" == "--log" ]]; then
    LOG_OUTPUT=true
fi

# 执行 Python 脚本，并根据参数决定是否重定向到文件
if $LOG_OUTPUT; then
    python /root/AILang/artifacts/criterion4/resnet_backward.py  >/root/AILang/artifacts/results/criterion4/resnet_backward.log 2>&1
    python /root/AILang/artifacts/criterion4/lstm_backward.py  >/root/AILang/artifacts/results/criterion4/lstm_backward.log 2>&1
    python /root/AILang/artifacts/criterion4/attention_backward.py >/root/AILang/artifacts/results/criterion4/attention_backward.log 2>&1
else
     python /root/AILang/artifacts/criterion4/resnet_backward.py  
    python /root/AILang/artifacts/criterion4/lstm_backward.py 
    python /root/AILang/artifacts/criterion4/attention_backward.py


fi
