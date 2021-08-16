#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H


#include "mat.h"
namespace intel {

class Convolution
{
public:
    Convolution();
    ~Convolution();

    int load_param(const ParamDict& pd);
    
    //int set_weight_bias(const IntelMat& weight_data, const IntelMat& bias_data);

    //int forward(const IntelMat& bottom_blob, IntelMat& top_blob, const IntelMat& weight_datax, const IntelMat& bias_datax) const;
    int forward(const IntelMat& bottom_blob, IntelMat& top_blob) const;

public:
    // param
    int fuse = 0;
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;
    int weight_data_size;

    IntelMat bias_data;
    IntelMat weight_data;
};

} // namespace intel

#endif // LAYER_CONVOLUTION_H
