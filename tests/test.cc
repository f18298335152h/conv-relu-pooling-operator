#include <iostream>
#include <vector>
#include <stdio.h>
#include <dirent.h>
#include <algorithm>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <dirent.h>
#include <math.h>
#include <numeric>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <list>
#include <unistd.h>
#include <chrono>
#include "mat.h"
#include "../src/operator/relu.h"
#include "../src/operator/pooling.h"
#include "../src/operator/convolution.h"
#include "../src/operator/fuse_conv_pool_relu.h"
using namespace std;


void test_relu()
{

    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    }; 
    intel::IntelMat mat_in(5,5,1,in);
    intel::ReLU *relu;
    auto startTime = std::chrono::steady_clock::now();
    relu->forward_inplace(mat_in);
    auto endTime = std::chrono::steady_clock::now();
    std::cout <<"relu cost time "<<std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()<<" microseconds"<<std::endl;

    printf("relu ouput w = %d h = %d c = %d \n", mat_in.w, mat_in.h, mat_in.c);
    for (int q = 0; q < mat_in.c; ++q) {
        float* ptr = mat_in.channel(q); 
        for (int i=0; i<mat_in.w*mat_in.h; i++) {
            printf("relu output channel=%d i=%d  output= %f\n", q, i, ptr[i]);
        }
    }
}

intel::IntelMat& test_pooling()
{


    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    }; 


    intel::IntelMat mat_in(5,5,1,in);
    intel::IntelMat top_blob;
    intel::ParamDict pd;
    pd.pooling_type = 0;
    pd.kernel_w = 1;
    pd.kernel_h = 1;
    pd.stride_w = 2;
    pd.stride_h = 2;
    pd.pad_w = 0; 
    pd.pad_h = 0;
    pd.global_pooling = 0;
    
    intel::Pooling pooling;
    pooling.load_param(pd);
    auto startTime = std::chrono::steady_clock::now();
    pooling.forward(mat_in, top_blob);
    auto endTime = std::chrono::steady_clock::now();
    std::cout <<"pooling cost time "<<std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()<<" microseconds"<<std::endl;
    
    printf("pooling ouput w = %d h = %d c = %d \n", top_blob.w, top_blob.h, top_blob.c);

    for (int q = 0; q < top_blob.c; ++q) {
        float* ptr = top_blob.channel(q); 
        for (int i=0; i<top_blob.w*top_blob.h; i++) {
            printf("pooling output channel=%d i=%d output=%f\n", q,i, ptr[i]);
        }
    }
    return top_blob;
}

intel::IntelMat& test_convolution()
{

    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    }; 
    float_t expected_out[] = {
        9.5f, 18.5f,
        18.5f, 27.5f
    };
    // weights & bias
    float_t w[] = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };
    float_t b[] = {
        0.5f
    };

    intel::IntelMat mat_in(5,5,1,in);
    intel::IntelMat top_blob;
    intel::ParamDict pd;
    pd.kernel_w = 3;
    pd.kernel_h = 3;
    pd.stride_w = 2;
    pd.stride_h = 2;
    pd.pad_w = 0; 
    pd.pad_h = 0;
    pd.num_output = 1;
    pd.dilation_w = 1;
    pd.dilation_h = 1;
    pd.bias_term = 1;
    pd.weight_data_size = 9;
    
    intel::Convolution conv;
    conv.load_param(pd);
    conv.bias_data.data = b;
    conv.weight_data.data = w;
    auto startTime = std::chrono::steady_clock::now();
    conv.forward(mat_in, top_blob);
    auto endTime = std::chrono::steady_clock::now();
    std::cout <<"conv cost time "<<std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()<<" microseconds"<<std::endl;
    
    printf("conv ouput w = %d h = %d c = %d \n", top_blob.w, top_blob.h, top_blob.c);
    for (int q = 0; q < top_blob.c; ++q) {
        float* ptr = top_blob.channel(q); 
        for (int i=0; i<top_blob.w*top_blob.h; i++) {
            printf("conv output channel=%d i=%d output=%f\n", q,i, ptr[i]);
        }
    }
    return top_blob;
}

void test_pipeline()
{

    float_t in[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    }; 
    float_t expected_out[] = {
        9.5f, 18.5f,
        18.5f, 27.5f,
        9.5f, 18.5f,
        18.5f, 27.5f,
        9.5f, 18.5f,
        18.5f, 27.5f
    };
    // weights & bias
    float_t w[] = {
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f
    };
    float_t b[] = {
        0.5f,
        0.5f,
        0.5f
    };

    intel::IntelMat mat_in(5,5,3,in);
    intel::IntelMat top_blob;
    
    intel::ParamDict conv_pd;
    conv_pd.kernel_w = 3;
    conv_pd.kernel_h = 3;
    conv_pd.stride_w = 1;
    conv_pd.stride_h = 1;
    conv_pd.pad_w = 0; 
    conv_pd.pad_h = 0;
    conv_pd.num_output = 1;
    conv_pd.dilation_w = 1;
    conv_pd.dilation_h = 1;
    conv_pd.bias_term = 1;
    conv_pd.weight_data_size = 9;

    intel::ParamDict pool_pd;
    pool_pd.pooling_type = 0;
    pool_pd.kernel_w = 1;
    pool_pd.kernel_h = 1;
    pool_pd.stride_w = 1;
    pool_pd.stride_h = 1;
    pool_pd.pad_w = 0; 
    pool_pd.pad_h = 0;
    pool_pd.global_pooling = 0;

    intel::FuseConvReluPooling Fuse_convReluPooling;
    Fuse_convReluPooling.load_param(conv_pd, pool_pd, w, b);
    auto startTime = std::chrono::steady_clock::now();
    Fuse_convReluPooling.forward(mat_in, top_blob);
    auto endTime = std::chrono::steady_clock::now();
    std::cout <<"Fuse cost time "<<std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()<<" microseconds"<<std::endl;
    printf("Fuse output top w = %d h = %d c = %d \n", top_blob.w, top_blob.h, top_blob.c);
    for (int q = 0; q < top_blob.c; ++q) {
        float* ptr = top_blob.channel(q); 
        for (int i=0; i<top_blob.w*top_blob.h; i++) {
            printf("fuse output channel=%d i=%d output=%f\n", q,i, ptr[i]);
        }
    }
}

int main()
{
    
    test_relu();
    test_pooling();
    test_convolution();
    test_pipeline();
    return 1;
}


