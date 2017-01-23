
iter=$1

./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb voc_2007_trainval --iters $iter --cfg experiments/cfgs/faster_rcnn_end2end.yml

python tools/demo.py --model output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_${iter}.caffemodel
