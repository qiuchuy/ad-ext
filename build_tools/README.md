# Build with Docker
+ build the docker image
```bash
docker build -t . ailang-dev 
docker run -it ailang-dev /bin/bash
```
+ pull [AILang](https://github.com/kom113/AILang) repository inside the container
+ build IREE & AILang
```
./AILang/build_tools/build_iree.sh
./AILang/build_tools/build.sh
```
> note: you may need a docker proxy when pulling images from gcr.io



# try
```
cd AILang
docker run -it --gpus all --name ailang-dev-hzy -v ./AILang:/root/haochaoyang  -w /root/AILang ailang-dev-hzy /bin/bash
```