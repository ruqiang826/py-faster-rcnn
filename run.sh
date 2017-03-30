
homedir=`pwd`
iters="1300 320 340"
today=`date -d '1 day ago' +'%Y%m%d'`
classes="archers archersC  archersE  arena  arrowE arrowL  arrowC earena  darena  bomb  bombC  bombE  dragon  dragonC  dragonE  fireballE fireballL fireballC king  eking  giant  giantC  giantE  goblins  goblinsC  goblinsE  knight  knightC  knightE  minipekka  minipekkaC  minipekkaE  musketeer  musketeerC  musketeerE  prince  princeC  princeE  skeleton  skeletonC  skeletonE  skeletonarmy  skeletonarmyC  skeletonarmyE  speargoblins  speargoblinsC  speargoblinsE  witch  witchC  witchE"

# replace CLASSES list. "../labelImg/data/predefined_classes.txt" is the standard class list
T1=`cat ../labelImg/data/predefined_classes.txt | sed ':1;$!N;s/\n/", "/;t1'` ;T2=`echo "CLASSES = (\"__background__\", \"$T1\")"` ;  sed -i "s/^CLASSES.*$/${T2}/g" lib/fast_rcnn/config.py



for iter in `echo $iters`
do
  echo $iter
  # 1. copy data from CR_AI
  rm data/VOCdevkit2007/VOC2007/Annotations/*.xml
  rm data/VOCdevkit2007/VOC2007/JPEGImages/*.jpg
  rm data/VOCdevkit2007/VOC2007/JPEGImages/*.png
  
  cp ../CR_AI/data/annotation/* data/VOCdevkit2007/VOC2007/Annotations/
  cp ../CR_AI/data/img/* data/VOCdevkit2007/VOC2007/JPEGImages/
  
  cd ../CR_AI/
  
  test_list=`python script/split_data.py data/img`
  
  cd -
  rm test_data/img/*
  rm test_data/annotation/*
  cd data/VOCdevkit2007/VOC2007/Annotations/
  mv $test_list  $homedir/test_data/annotation
  cd -
  cd data/VOCdevkit2007/VOC2007/JPEGImages/
  mv $test_list  $homedir/test_data/img
  cd -
  
  # 2. generate training list
  ls -1 data/VOCdevkit2007/VOC2007/JPEGImages | sed 's/.png//g' | sed 's/.jpg//g' > data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt
  
  
  # 3. clean cache:
  rm data/cache/voc_2007_trainval_gt_roidb.pkl
  rm data/VOCdevkit2007/annotations_cache/annots.pkl
  
  
  cp ../CR_AI/script/model_test.py ./lib/utils/
  cp ../CR_AI/script/pascal_voc_io.py ./lib/utils/
  ./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb voc_2007_trainval --iters $iter --cfg experiments/cfgs/faster_rcnn_end2end.yml > test_model.${iter}
  
  #python tools/demo.py --model output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_${iter}.caffemodel
  
  
  for i in `echo $classes` ;do grep class_score test_model.$iter | grep " $i " > /tmp/${i}_${iter}.txt ;done
  
  > count_label_${iter}
  for i in `echo $classes` ;do echo -n $i" "; grep "<name>${i}</name>" data/VOCdevkit2007/VOC2007/Annotations/* | wc -l >> count_label_${iter};done
  
  for i in `echo $classes` ;do gnuplot -e "set terminal png size 1280,800; set output '/tmp/${i}_${iter}.png';plot '/tmp/${i}_${iter}.txt' using 2:7 with lp";done
  

done

for iter in `echo $iters`
do
  echo "========================"
  echo -n "model of iter $iter "
  grep " total_score " test_model.${iter} | tail -1000 | awk '{t += $5;w += $4} END{print w,t,w/t}'

  for i in `echo $classes` ;
  do 
    echo -n "class $i "
    grep class_score test_model.$iter | grep " $i " | tail -n 1000 | awk '{t+=$7; n+=1} END {print t/n}';
  done  | sort -k3 -rn
  echo "========================"
done



cd $homedir
