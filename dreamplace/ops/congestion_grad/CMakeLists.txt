# 设置算子名称，与文件夹名一致
set(OP_NAME congestion_grad)

# 包含路径：包含父目录和当前算子目录
set(INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接库：使用 DREAMPlace 通用的 utility 库
set(LINK_LIBS 
    utility_cxx
)

# 1. 编译 C++ 版本的算子 (CPU 版本，通常用于兼容性)
add_pytorch_extension(congestion_grad_cpp
    src/congestion_grad.cpp
    EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
    EXTRA_LINK_LIBRARIES ${LINK_LIBS}
)

# 2. 如果开启了 CUDA，编译 CUDA 版本的算子
if(TORCH_ENABLE_CUDA)
    add_pytorch_extension(congestion_grad_cuda
        src/congestion_grad.cpp
        src/congestion_grad_cuda_kernel.cu
        EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
        EXTRA_LINK_LIBRARIES ${LINK_LIBS}
    )
    
    # 安装 CUDA 动态库
    install(TARGETS congestion_grad_cuda DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

# 3. 安装 CPU 动态库
install(TARGETS congestion_grad_cpp DESTINATION dreamplace/ops/${OP_NAME})

# 4. 安装 Python 脚本
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})