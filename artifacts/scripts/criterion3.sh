#!/bin/bash

# 检查是否传入了 --log 参数
LOG_OUTPUT=false
if [[ "$1" == "--log" ]]; then
    LOG_OUTPUT=true
fi

# 执行 Python 脚本，并根据参数决定是否重定向到文件
if $LOG_OUTPUT; then
    python /root/AILang/artifacts/criterion3/resnet.py  >/root/AILang/artifacts/results/criterion3/resnet.log 2>&1
    python /root/AILang/artifacts/criterion3/resnet_jit.py  >/root/AILang/artifacts/results/criterion3/resnet_jit.log 2>&1

    python /root/AILang/artifacts/criterion3/lstm.py  >/root/AILang/artifacts/results/criterion3/lstm.log 2>&1
    python /root/AILang/artifacts/criterion3/lstm_jit.py  >/root/AILang/artifacts/results/criterion3/lstm_jit.log 2>&1

    python /root/AILang/artifacts/criterion3/attention.py >/root/AILang/artifacts/results/criterion3/attention.log 2>&1
    python /root/AILang/artifacts/criterion3/attention_jit.py >/root/AILang/artifacts/results/criterion3/attention_jit.log 2>&1
else
    python /root/AILang/artifacts/criterion3/resnet.py 
    python /root/AILang/artifacts/criterion3/resnet_jit.py 

    python /root/AILang/artifacts/criterion3/lstm.py
    python /root/AILang/artifacts/criterion3/lstm_jit.py 

    python /root/AILang/artifacts/criterion3/attention.py 
    python /root/AILang/artifacts/criterion3/attention_jit.py 

fi
