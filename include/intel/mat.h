
#ifndef Intel_MAT_H
#define Intel_MAT_H

#include <stdlib.h>
#include <string.h>
#include <vector>

namespace intel {

    struct ParamDict {
        int  pooling_type;
        int  kernel_w;
        int  kernel_h;
        int  stride_w;
        int  stride_h;
        int  pad_w; 
        int  pad_h;
        int  global_pooling;
        int num_output;
        int dilation_w;
        int dilation_h;
        int bias_term;
        int weight_data_size;
    };



// the three dimension matrix
class IntelMat
{
public:
    // empty
    IntelMat();
    // vec
    IntelMat(int w);
    // image
    IntelMat(int w, int h);
    // dim
    IntelMat(int w, int h, int c);
    // copy
    IntelMat(const IntelMat& m);
    // external vec
    IntelMat(int w, float* data);
    // external image
    IntelMat(int w, int h, float* data);
    // external dim
    IntelMat(int w, int h, int c, float* data);
    // release
    ~IntelMat();
    // assign
    IntelMat& operator=(const IntelMat& m);
    // set all
    void fill(float v);
    // deep copy
    IntelMat clone() const;
    // reshape vec
    IntelMat reshape(int w) const;
    // reshape image
    IntelMat reshape(int w, int h) const;
    // reshape dim
    IntelMat reshape(int w, int h, int c) const;
    // allocate vec
    void create(int w);
    // allocate image
    void create(int w, int h);
    // allocate dim
    void create(int w, int h, int c);
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    IntelMat channel(int c);
    const IntelMat channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    operator float*();
    operator const float*() const;

    enum
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB       = 1,
        PIXEL_BGR       = (1 << 1),
        PIXEL_GRAY      = (1 << 2),
        PIXEL_RGBA      = (1 << 3),

        PIXEL_RGB2BGR   = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY  = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB   = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY  = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB  = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR  = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB  = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR  = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static IntelMat from_pixels(const unsigned char* pixels, int type, int w, int h);
    // convenient construct from pixel data and resize to specific size
    static IntelMat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type);
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height);

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precisoin floating point data
    static IntelMat from_float16(const unsigned short* data, int size);

    // the dimensionality
    int dims;
    // pointer to the data
    float* data;

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    int w;
    int h;
    int c;

    size_t cstep;
};

// misc function
// image pixel bilinear resize
void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);

// mat process
enum
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
};
void copy_make_border(const IntelMat& src, IntelMat& dst, int top, int bottom, int left, int right, int type, float v);
void copy_cut_border(const IntelMat& src, IntelMat& dst, int top, int bottom, int left, int right);
void resize_bilinear(const IntelMat& src, IntelMat& dst, int w, int h);

// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define Intel_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define Intel_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define Intel_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define Intel_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define Intel_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define Intel_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
static inline void Intel_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

inline IntelMat::IntelMat()
    : dims(0), data(0), refcount(0), w(0), h(0), c(0), cstep(0)
{
}

inline IntelMat::IntelMat(int _w)
    : dims(0), data(0), refcount(0)
{
    create(_w);
}

inline IntelMat::IntelMat(int _w, int _h)
    : dims(0), data(0), refcount(0)
{
    create(_w, _h);
}

inline IntelMat::IntelMat(int _w, int _h, int _c)
    : dims(0), data(0), refcount(0)
{
    create(_w, _h, _c);
}

inline IntelMat::IntelMat(const IntelMat& m)
    : dims(m.dims), data(m.data), refcount(m.refcount)
{
    if (refcount)
        Intel_XADD(refcount, 1);

    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;
}

inline IntelMat::IntelMat(int _w, float* _data)
    : dims(1), data(_data), refcount(0)
{
    w = _w;
    h = 1;
    c = 1;

    cstep = w;
}

inline IntelMat::IntelMat(int _w, int _h, float* _data)
    : dims(2), data(_data), refcount(0)
{
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;
}

inline IntelMat::IntelMat(int _w, int _h, int _c, float* _data)
    : dims(3), data(_data), refcount(0)
{
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * sizeof(float), 16) >> 2;
}

inline IntelMat::~IntelMat()
{
    release();
}

inline IntelMat& IntelMat::operator=(const IntelMat& m)
{
    if (this == &m)
        return *this;
    //if (m.refcount)
    //    Intel_XADD(m.refcount, 1);

    release();

    dims = m.dims;
    data = m.data;
    refcount = m.refcount;

    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    printf("copy copy 3");
    return *this;
}

inline void IntelMat::fill(float _v)
{
    size_t _total = total();
    for (size_t i = 0; i < _total; i++)
    {
        data[i] = _v;
    }
}

inline IntelMat IntelMat::clone() const
{
    if (empty())
        return IntelMat();

    IntelMat m;
    if (dims == 1)
        m.create(w);
    else if (dims == 2)
        m.create(w, h);
    else if (dims == 3)
        m.create(w, h, c);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * sizeof(float));
    }

    return m;
}

inline IntelMat IntelMat::reshape(int _w) const
{
    if (w * h * c != _w)
        return IntelMat();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        IntelMat m;
        m.create(_w);

        // flatten
        for (int i=0; i<c; i++)
        {
            const float* ptr = data + i * cstep;
            float* mptr = m.data + i * w * h;
            memcpy(mptr, ptr, w * h * sizeof(float));
        }

        return m;
    }

    IntelMat m = *this;

    m.dims = 1;

    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

inline IntelMat IntelMat::reshape(int _w, int _h) const
{
    if (w * h * c != _w * _h)
        return IntelMat();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        IntelMat m;
        m.create(_w, _h);

        // flatten
        for (int i=0; i<c; i++)
        {
            const float* ptr = data + i * cstep;
            float* mptr = m.data + i * w * h;
            memcpy(mptr, ptr, w * h * sizeof(float));
        }

        return m;
    }

    IntelMat m = *this;

    m.dims = 2;

    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.cstep = _w * _h;

    return m;
}

inline IntelMat IntelMat::reshape(int _w, int _h, int _c) const
{
    if (w * h * c != _w * _h * _c)
        return IntelMat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize(_w * _h * sizeof(float), 16) >> 2)
        {
            IntelMat m;
            m.create(_w, _h, _c);

            // align channel
            for (int i=0; i<_c; i++)
            {
                const float* ptr = data + i * _w * _h;
                float* mptr = m.data + i * m.cstep;
                memcpy(mptr, ptr, _w * _h * sizeof(float));
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        IntelMat tmp = reshape(_w * _h * _c);
        return tmp.reshape(_w, _h, _c);
    }

    IntelMat m = *this;

    m.dims = 3;

    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.cstep = alignSize(_w * _h * sizeof(float), 16) >> 2;

    return m;
}

inline void IntelMat::create(int _w)
{
    release();

    dims = 1;

    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = total() * sizeof(float);
        data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void IntelMat::create(int _w, int _h)
{
    release();

    dims = 2;

    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = total() * sizeof(float);
        data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void IntelMat::create(int _w, int _h, int _c)
{
    release();

    dims = 3;

    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * sizeof(float), 16) >> 2;

    if (total() > 0)
    {
        size_t totalsize = total() * sizeof(float);
        data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void IntelMat::addref()
{
    if (refcount)
        Intel_XADD(refcount, 1);
}

inline void IntelMat::release()
{
    if (refcount && Intel_XADD(refcount, -1) == 1)
        fastFree(data);

    dims = 0;
    data = 0;

    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool IntelMat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t IntelMat::total() const
{
    return cstep * c;
}

inline IntelMat IntelMat::channel(int c)
{
    return IntelMat(w, h, data + cstep * c);
}

inline const IntelMat IntelMat::channel(int c) const
{
    return IntelMat(w, h, data + cstep * c);
}

inline float* IntelMat::row(int y)
{
    return data + w * y;
}

inline const float* IntelMat::row(int y) const
{
    return data + w * y;
}

inline IntelMat::operator float*()
{
    return data;
}

inline IntelMat::operator const float*() const
{
    return data;
}

} // namespace intel

#endif // Intel_MAT_H
