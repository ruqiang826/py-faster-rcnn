
iter=$1

# 1. copy data from CR_AI
#cp ../CR_AI/data/annotation/* data/VOCdevkit2007/VOC2007/Annotations/
#cp ../CR_AI/data/img/* data/VOCdevkit2007/VOC2007/JPEGImages/

# 2. generate training list
ls -1 data/VOCdevkit2007/VOC2007/JPEGImages | sed 's/.png//g' | sed 's/.jpg//g' > data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt



# 3. clean cache:
#rm data/cache/voc_2007_trainval_gt_roidb.pkl
#rm data/VOCdevkit2007/annotations_cache/annots.pkl



./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb voc_2007_trainval --iters $iter --cfg experiments/cfgs/faster_rcnn_end2end.yml

python tools/demo.py --model output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_${iter}.caffemodel

