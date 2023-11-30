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
```