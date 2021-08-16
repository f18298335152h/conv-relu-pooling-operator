#ABI="armeabi-v7a"
ABI="x86-64"

function build_x86_64() {
    mkdir -p build-x86-64
    pushd build-x86-64
    /usr/local/bin/cmake -DABI=$ABI ..
    make -j32
    popd
}

function build_armv7a_with_neon() {
    rm -rf build-android-armv7a-with-neon
    mkdir -p build-android-armv7a-with-neon
    pushd build-android-armv7a-with-neon
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="armeabi-v7a with NEON" \
        -DANDROID_STL=c++_static \
        -DANDROID_PLATFORM=android-16 \
        -DABI=$ABI \
        ..
    make -j32
    popd
}

case "$ABI" in
    "x86-64")
        build_x86_64
        ;;
    "armeabi-v7a")
        build_armv7a_with_neon
        ;;
esac
