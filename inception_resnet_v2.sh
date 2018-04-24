# convert
# python scripts/convert_plan.py data/frozen_graphs/inception_resnet_v2.pb data/plans/inception_resnet_v2.plan input 299 299 InceptionResnetV2/Logits/Logits/BiasAdd  1 0 float

# test camera
./build/examples/classify_image/classify_image  data/plans/inception_resnet_v2.plan data/imagenet_labels_1001.txt input InceptionResnetV2/Logits/Logits/BiasAdd inception
