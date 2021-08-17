## used OpenMap, NEON, Inline Assembly speed on arm architecture platform

## speed tech
1. OpenMap
2. NEON
3. Inline Assembly
## (NEON Inline Assembly) make use of SMID 
## operator
1. Relu
2. Convolution2D
3. Pooling2D
4. FuseConvReluPooling

## for the detail speed use OpenMap, NEON, Inline Assembly, please checkout code on src

## do not use MKL for benchmark because coding on MAC  

## FLOPS please reference cost time on Mac Pro 
    1. Convolution2D cost time 21 microseconds:(inputs tensor: 5x5x1)
    2. pooling cost time 20 microseconds :(inputs tensor: 5x5x1)
    3. Relu cost time 12 microseconds :(inputs tensor: 5x5x1)
    4. Fuse cost time 3 microseconds :(inputs tensor: 5x5x3)
## FLOPS please reference cost time on Iphone or android 
    


### build-android-armv7a-with-neon contain executable file test , please run on arm architecture platform.
### build-x86-64 contain executable file test, please run on pc platform such as Mac , Windows, Linux ect.

## how to build 
### open the build.sh file to seeting cmake path for x86 or cross build cmake

1. ./build.sh build for armv7a or x86_64
2. pytorch_test.py can verify the result ,please run test print result compare with pytorch_test.py with the same input (install torch)


### cost time will be printed on different platform (time: microseconds)

### libintel.a will be builded you can depedency on you project


