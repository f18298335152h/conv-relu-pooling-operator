
#include "pooling.h"
#include <algorithm>

namespace intel {


Pooling::Pooling()
{
    //one_blob_only = true;
    //support_inplace = false;
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.pooling_type;
    kernel_w = pd.kernel_w;
    kernel_h = pd.kernel_h;
    stride_w = pd.stride_w;
    stride_h = pd.stride_h;
    pad_w = pd.pad_w;
    pad_h = pd.pad_h;
    global_pooling = pd.global_pooling; 

    return 0;
}

int Pooling::forward(const IntelMat& bottom_blob, IntelMat& top_blob) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    fprintf(stderr, "Pooling input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(1, 1, channels);
        if (top_blob.empty())
            return -100;

        int size = w * h;

#if (defined(__arm64__) && defined(__APPLE__)) || defined(__aarch64__)
        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                outptr[0] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                outptr[0] = sum / size;
            }
        }

#elif   defined(__x86_64__) || defined(_M_X64)
        if (pooling_type == PoolMethod_MAX)
        {
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }
                outptr[0] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                outptr[0] = sum / size;
            }
        }
#endif

        return 0;
    }

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
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    int wtail = (w - kernel_w) % stride_w;
    int htail = (h - kernel_h) % stride_h;
    if ((pad_w == -233 && pad_h == -233) || (pad_w == -2333 && pad_h == -2333))
    {
        wtail = 0;
        htail = 0;
    }
    if (wtail != 0 || htail != 0)
    {
        int wtailpad = 0;
        int htailpad = 0;
        if (wtail != 0)
            wtailpad = kernel_w - wtail;
        if (htail != 0)
            htailpad = kernel_h - htail;

        IntelMat bottom_blob_bordered2;
        if (pooling_type == PoolMethod_MAX)
        {
            copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_REPLICATE, 0.f);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_CONSTANT, 0.f);
        }
        if (bottom_blob_bordered2.empty())
            return -100;

        bottom_blob_bordered = bottom_blob_bordered2;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        if (wtail != 0)
            outw += 1;
        if (htail != 0)
            outh += 1;
    }

    top_blob.create(outw, outh, channels);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

#if (defined(__arm64__) && defined(__APPLE__)) || defined(__aarch64__)
    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const IntelMat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const IntelMat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix tail pad
            if (wtail != 0)
            {
                const float scale = (float)kernel_w / wtail;

                outptr = top_blob.channel(q) + outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (htail != 0)
            {
                const float scale = (float)kernel_h / htail;

                outptr = top_blob.channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
        }
    }

#elif   defined(__x86_64__) || defined(_M_X64)
    if (pooling_type == PoolMethod_MAX)
    {
        for (int q=0; q<channels; q++)
        {
            const IntelMat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        for (int q=0; q<channels; q++)
        {
            const IntelMat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix tail pad
            if (wtail != 0)
            {
                const float scale = (float)kernel_w / wtail;

                outptr = top_blob.channel(q) + outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (htail != 0)
            {
                const float scale = (float)kernel_h / htail;

                outptr = top_blob.channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
        }
    }
#endif
    return 0;
}

} // namespace intel
