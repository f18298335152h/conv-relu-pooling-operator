
#include "fuse_conv_pool_relu.h"

namespace intel {

FuseConvReluPooling::FuseConvReluPooling()
{
    //one_blob_only = true;
    //support_inplace = false;
}

FuseConvReluPooling::~FuseConvReluPooling()
{
}

int FuseConvReluPooling::load_param(const ParamDict& conv_pd, const ParamDict& pooling_pd, float* weight_data, float* bias_data)
{
    
    conv.load_param(conv_pd);
    pooling.load_param(pooling_pd);
    
    conv.bias_data.data = weight_data;
    conv.weight_data.data = bias_data;
    return 0;
}


int FuseConvReluPooling::forward(const IntelMat& bottom_blob, IntelMat& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias
    printf("coming xxx");
    intel::IntelMat top_blob_pooling;
    conv.forward(bottom_blob, top_blob);
    
    pooling.forward(top_blob_pooling, top_blob);
    relu->forward_inplace(top_blob);
    
    return 0;
}

} // namespace intel
