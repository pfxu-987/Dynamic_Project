rm -rf build && mkdir -p build && cd build 
cmake .. -DPYTHON_EXECUTABLE=$(which python3.9) -DWITH_GPU=ON  -DWITH_PROFILER=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCUDA_ARCHS="60;61;70;75;86" 
make -j  
pip uninstall -y turbo-transformers 
pip install  `find . -name *whl`