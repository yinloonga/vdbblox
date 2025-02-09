#!/usr/bin/env zsh
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Conda initialization script not found. Check your Conda installation."
    exit 1
fi

conda deactivate && \


# Release 模式
catkin config -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=$(which gcc) \
-DCMAKE_CXX_COMPILER=$(which g++) \
-DCMAKE_CXX_STANDARD=17 

catkin build -j "$(nproc --all)"
