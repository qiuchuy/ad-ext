# 注意!(2023/12/2):
IREE目前的主分支代码存在问题，导致在构建IREE Runtime API的时候会产生编译失败的问题
+ 临时解决方案：参考这个[issue](https://github.com/openxla/iree/issues/15761)，目前这一patch还没有被合并到主分支。具体来说，在同步完成git submodule之后，参考该issue的patch手动修改一部分代码，再进行cmake配置与项目构建
# Build with Docker
+ build the docker image
```bash
docker build -t . ailang-dev 
docker run -it ailang-dev /bin/bash
```
+ inside the container
```
git clone https://github.com/kom113/AILang
cd AILang
./AILang/build_tools/build.sh
```