# PetS Guide

1. 首先通过 `git clone` 命令，将整个项目克隆到本地

2. 进入 `workspace` 文件夹下，在该目录下拉取镜像

   ```bash
   docker pull eliaswyq/pets_gpu:latest
   ```

3. 拉取镜像后，执行以下命令查看镜像是否成功拉取

   ```
   docker images
   ```

4. 利用以下命令创建并进入容器

   ```bash
   nvidia-docker run --gpus 1  -it   -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=pets_dev eliaswyq/pets_gpu:latest
   ```

5. 进入容器后，查看cuda版本，应该会是版本 cuda 11.7

   ```
   nvcc -V
   ```

6. 查看python和torch版本，应该是python==3.7.7，torch==1.13.1

   ```bash
   python -V
   pip show torch
   ```

7. 查看transformers版本，应该是transformers==4.11.1

   ```
   pip show transformers
   ```

8. 版本没问题后，在workspace目录下进行编译

   ```bash
   rm -rf build && mkdir -p build && cd build 
   cmake .. -DWITH_GPU=ON  -DWITH_PROFILER=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCUDA_ARCHS="60;61;70;75;86" 
   make -j  
   pip uninstall -y turbo-transformers 
   pip install  `find . -name *whl`
   ```

9. 如果想升级python版本，可以通过conda创建一个新的环境，进入新环境后，重新利用以下命令进行编译

   ```bash
   rm -rf build && mkdir -p build && cd build 
   cmake .. -DPYTHON_EXECUTABLE=$(which python3.9) -DWITH_GPU=ON  -DWITH_PROFILER=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCUDA_ARCHS="60;61;70;75;86" 
   make -j  
   pip uninstall -y turbo-transformers 
   pip install  `find . -name *whl`
   ```

   