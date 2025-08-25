#!/bin/bash
build(){
    TARGET=$1
    if cmake --build build --target=$TARGET; then
        echo "build finished!"
    else
        echo "build failed!"
        exit 1
    fi
}
run(){
    TARGET=$1
    TARGET_PATH=$(find build -type f -name ${TARGET} -executable | head -n 1)
    if [ -n "$TARGET_PATH" ]; then
        echo "Found executable: $TARGET_PATH"
        echo "Running..."
        echo "###############################"
        "$TARGET_PATH"
        echo "###############################"
    else
        echo "Executable '$TARGET' not found in $BUILD_DIR"
        exit 1
    fi
    cd ..
}

TARGET=$1
build $TARGET
run $TARGET