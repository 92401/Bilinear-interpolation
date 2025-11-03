#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Dispatch.h> // 
// 1.trilinear_fw_cu  2.trilinear_fw_kernel    1调用2

//2.核函数   
//创建通用代码，让同一个函数可以处理多种数据类型
template <typename scalar_t>

__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
    // 计算当前线程处理的索引  n为x方向索引 f为y方向索引 直接用公式计算
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
       
    // 检查索引是否在范围内
    if (n >= points.size(0) || f >= feats.size(2)) return;

     // 当前点属于哪个四边形
    const scalar_t u = points[n][0];  // 四边形内的u坐标
    const scalar_t v = points[n][1];  // 四边形内的v坐标

    
    // 直接从feats中获取四个角点的特征值
    feat_interp[n][f] = 
        feats[n][0][f] * (1 - u) * (1 - v) +  // 角点0特征
        feats[n][1][f] * u * (1 - v) +        // 角点1特征
        feats[n][2][f] * (1 - u) * v +        // 角点2特征
        feats[n][3][f] * u * v;               // 角点3特征
}

//1c++函数

torch::Tensor trilinear_fw_cu(    //重复函数签名，可以让cpp文件识别到
    const torch::Tensor feats,
    const torch::Tensor points
) {
    const int N = feats.size(0), F = feats.size(2);   //获取第一个和第三个维度
    // 初始化输出张量，调用的2.核函数只能是viod类型所以不会有返回值，所以需要提前初始化
    torch::Tensor feat_interp = torch::empty({N, F}, feats.options());

    // 启动CUDA核函数，定义所需的线程块block和线程数threads
    const dim3 threads(16, 16);  // 每个线程块处理16x16个元素，自定义
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);//线程块数量用公式计算

    //feats.scalar_type(): 获取输入张量的数据类型 报错时提示"trilinear_fw_cu": 函数名  [&] 表示捕获所有外部变量引用
    AT_DISPATCH_FLOATING_TYPES(feats.scalar_type(), "trilinear_fw_cu", ([&] {
        //调用核函数   packed_accessor访问器类型，创建访问器实例
        //  scalar_t 根据输入张量类型自动推变量类型 3是张量的维度  
        //torch::RestrictPtrTraits指针不会指向重叠的内存区域，允许编译器进行更激进的优化size_t作用：索引类型指定
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));  
    //

    return feat_interp;
}
