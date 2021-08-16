#ifndef LAYER_FUSECONVRELUPOOLING_H
#define LAYER_FUSECONVRELUPOOLING_H


#include "mat.h"
#include "relu.h"
#include "pooling.h"
#include "convolution.h"
namespace intel {

class FuseConvReluPooling 
{
public:
    FuseConvReluPooling();
    ~FuseConvReluPooling();

    int load_param(const ParamDict& conv_pd, const ParamDict& pooling_pd, float* weight_data, float* bias_data);
    
    int forward(const IntelMat& bottom_blob, IntelMat& top_blob) const;

public:
    
    intel::ReLU *relu;
    intel::Convolution conv;
    intel::Pooling pooling;
};

} // namespace intel

#endif // LAYER_FUSECONVRELUPOOLING_H 
