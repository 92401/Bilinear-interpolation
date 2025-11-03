from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension  #导入c++ build的库
# 定义 setup ，固定的形式复制粘贴即可
setup(
    name='cppcuda_tutorial',     #pip时的名字
    version='1.0',   #版本号
    author='sykl23',  #作者
    author_email='1799967694@qq.com',  #邮箱
    description='cppcuda_example', #关于包的备注描述
    long_description='cppcuda_example',
    ext_modules=[
        CUDAExtension(  # 使用 CUDAExtension 而不是 CppExtension自动处理所有 CUDA 相关的编译和链接，更简单可靠
            name='cppcuda_tutorial',  #python调用时包的名字import cppcuda_tutorial
            sources=['f_b.cpp', 'f_b_cu.cu'], #文件列表，转到这两个文件中
            include_dirs=['./include']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)