# Build with Docker
+ build the docker image
```bash
docker build -t . ailang-dev 
docker run -it ailang-dev /bin/bash
```
+ build AINL
```
git clone https://github.com/kom113/AILang
cd AILang
./AILang/build_tools/build.sh
<<<<<<< HEAD
```
> note: you may need a docker proxy when pulling images from gcr.io



# try
```
cd AILang
docker run -it --gpus all --name ailang-dev-hzy -v ./AILang:/root/haochaoyang  -w /root/AILang ailang-dev-hzy /bin/bash
=======
>>>>>>> origin/master
```