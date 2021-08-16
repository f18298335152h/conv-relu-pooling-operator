#ifndef LAYER_POOLING_H
#define LAYER_POOLING_H

#include "mat.h"
namespace intel {

class Pooling 
{
public:
    Pooling();

    
    int load_param(const ParamDict& pd);
    
    int forward(const IntelMat& bottom_blob, IntelMat& top_blob) const;

    enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

public:
    // param
    int pooling_type;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int global_pooling;
};

} // namespace intel

#endif // LAYER_POOLING_H
