
#ifndef LAYER_RELU_H
#define LAYER_RELU_H

#include "mat.h"
namespace intel {

class ReLU
{
public:
    ReLU();

    //int load_param(FILE* paramfp);

    int forward(const IntelMat& bottom_blob, IntelMat& top_blob, float slope = 0.f) const;

    int forward_inplace(IntelMat& bottom_top_blob, float slope = 0.f) const;
};

} // namespace intel

#endif // LAYER_RELU_H
