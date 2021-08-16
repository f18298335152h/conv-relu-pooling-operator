
#include "relu.h"

namespace intel {

int ReLU::forward(const IntelMat& bottom_blob, IntelMat& top_blob, float slope) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

#if (defined(__arm64__) && defined(__APPLE__)) || defined(__aarch64__)
    if (slope == 0.f)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    outptr[i] = 0;
                else
                    outptr[i] = ptr[i];
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    outptr[i] *= slope;
                else
                    outptr[i] = ptr[i];
            }
        }
    }
#elif   defined(__x86_64__) || defined(_M_X64)
    if (slope == 0.f)
    {
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    outptr[i] = 0;
                else
                    outptr[i] = ptr[i];
            }
        }
    }
    else
    {
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    outptr[i] *= slope;
                else
                    outptr[i] = ptr[i];
            }
        }
    }
#endif

    return 0;
}

int ReLU::forward_inplace(IntelMat& bottom_top_blob, float slope) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    printf("relu start");
#if (defined(__arm64__) && defined(__APPLE__)) || defined(__aarch64__)
    if (slope == 0.f)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }
#elif   defined(__x86_64__) || defined(_M_X64)
    if (slope == 0.f)
    {
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }
#endif

    return 0;
}

} // namespace intel
