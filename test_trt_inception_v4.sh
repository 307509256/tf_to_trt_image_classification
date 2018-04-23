#/bin/bash
echo $#
if [ $# -eq 0 ];
then
    echo "please input image path"
    exit
fi
./build/src/test/test_trt $1  data/plans/inception_v4.plan input 299 299 InceptionV4/Logits/Logits/BiasAdd 1001 preprocess_inception 3  float 1 0 0 works
