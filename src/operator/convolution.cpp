
#include "convolution.h"
#include "relu.h"
namespace intel {

Convolution::Convolution()
{
    //one_blob_only = true;
    //support_inplace = false;
}

Convolution::~Convolution()
{
}

int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.num_output;
    kernel_w = pd.kernel_w;
    kernel_h = pd.kernel_h;
    stride_w = pd.stride_w;
    stride_h = pd.stride_h;
    dilation_w = pd.dilation_w;
    dilation_h = pd.dilation_h;
    pad_w = pd.pad_w;
    pad_h = pd.pad_h;
    bias_term = pd.bias_term;
    weight_data_size = pd.weight_data_size;
    return 0;
}


int Convolution::forward(const IntelMat& bottom_blob, IntelMat& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    
    
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    IntelMat bottom_blob_bordered = bottom_blob;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // num_output
    const float* weight_data_ptr = weight_data;
#if (defined(__arm64__) && defined(__APPLE__)) || defined(__aarch64__)
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data.data[p];

                const float* kptr = weight_data_ptr + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const IntelMat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }

                    kptr += maxk;
                }

                outptr[j] = sum;
            }
            outptr += outw;
        }
    }

#elif   defined(__x86_64__) || defined(_M_X64)
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data.data[p];

                const float* kptr = weight_data_ptr + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const IntelMat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }
                    kptr += maxk;
                }
                outptr[j] = sum;
            }
            outptr += outw;
        }
    }
#endif

    if (fuse)
    {
        intel::IntelMat res_blob;
        intel::ReLU *relu;
        relu->forward_inplace(top_blob);
        //pooling 参数配置
        //intel::Pooling *pooling;
        //pooling->load_param(pd);
        //pooling->forward(top_blob, res_blob);
    }

    return 0;
}

} // namespace intel
