#/bin/bash

echo $#
if [ $# -eq 0 ];
then 
    echo "please input image path"
    exit
fi
./build/src/test/test_trt $1  data/plans/resnet_v2_152.plan input 299 299 resnet_v2_152/SpatialSqueeze 1001 preprocess_inception 3  float 1 0 0 works
