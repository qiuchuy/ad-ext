# Artifacts Evaluation
## Criterion 1
### 指标描述
新的自主 AI编程语言，能够支持x86 CPU 、Nvidia GPU，支持寒武纪、昇腾、燧原等不少于两种的国产AI芯片。
### 验证方式
测试AI编程语言对单个算子的支持，使用语言支持的算子编写Sigmoid、Convolution和Matmul三个函数。使用Numpy库统一初始化随机种子，确保测试过程中数据一致性。使用from_numpy()接口将初始化后的数据迁移至不同的测试框架中。对比AI编程语言框架与主流开源框架（如PyTorch）的测试结果，分析其误差范围是否在允许范围内。
### 验证详细结果

### 执行代码
```shell
cd artifacts/scripts
bash reproduce.sh
```
验证结果
