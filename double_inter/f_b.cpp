#include <torch/extension.h>  //用来让c++来识别torch
#include "utils.h"  //自定义的库


#pragma message("utils.h included successfully") // 输出预处理器消息


//定义一个c++函数，输入为两个tensor，输出为一个cu文件中函数的返回值
torch :: Tensor fw_cu(
    const torch::Tensor feats,
    const torch::Tensor points
)
{   CHECK_INPUT(feats);  //检查输入是否为cuda tensor且连续，在utils.h中定义
    CHECK_INPUT(points);
    return trilinear_fw_cu(feats,points); //在utils.h中定义引入
}

//pybind用于从C++代码中创建Python扩展的库
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_b", &fw_cu, "feature bilinear interpolation");  //f_b是python中调用的函数名，fw_cu是c++中定义的函数名
}