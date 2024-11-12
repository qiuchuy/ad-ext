#!/bin/bash

# 检查是否传入了 --log 参数
LOG_OUTPUT=false
if [[ "$1" == "--log" ]]; then
    LOG_OUTPUT=true
fi

if $LOG_OUTPUT; then
bash /root/AILang/artifacts/scripts/criterion1.sh --log
bash /root/AILang/artifacts/scripts/criterion3.sh --log
bash /root/AILang/artifacts/scripts/criterion4.sh --log
else
bash /root/AILang/artifacts/scripts/criterion1.sh 
bash /root/AILang/artifacts/scripts/criterion3.sh 
bash /root/AILang/artifacts/scripts/criterion4.sh 
fi