### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
There are slight differences between the two implementations.
In particular, this Python port
 - is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 220ms / image vs. 200ms / image for VGG16)
 - gives similar, but not exactly the same, mAP as the MATLAB version
 - is *not compatible* with models trained using the MATLAB code due to the minor implementation differences
 - **includes approximate joint training** that is 1.5x faster than alternating optimization (for VGG16) -- see these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more information

# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
  ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
2017.1.21:

1. caffe
这个代码有9个月没有更新了， cudnn升级到v5, 这里对应的应该是4,所以自带的caffe和cudnn不匹配了，编译不过。caffe主线上已经升级到cudnn 5了。之前把caffe 最新版拷过来，覆盖这里的caffe。但后来觉得这个做法不妥，应该最小修改这里自带的caffe。

py-faster-rcnn commit号 96dc9f1,caffe用release rc4(支持cudnn 5).

从caffe rc4拷贝如下文件到py-faster-rcnn自带的caffe里, 当前在caffe rc4目录下。
cp ./include/caffe/util/cudnn.hpp ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/include/caffe/util/
cp src/caffe/layers/cudnn_sigmoid_layer.cu ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/src/caffe/layers/ 
cp src/caffe/layers/cudnn_sigmoid_layer.cpp ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/src/caffe/layers/ 
cp include/caffe/layers/cudnn_sigmoid_layer.hpp ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/include/caffe/layers/ 

cp include/caffe/layers/cudnn_relu_layer.hpp ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/include/caffe/layers/ 
cp src/caffe/layers/cudnn_relu_layer.c* ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/src/caffe/layers/


cp include/caffe/layers/cudnn_tanh_layer.hpp ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/include/caffe/layers/ 
cp src/caffe/layers/cudnn_tanh_layer.c* ../py-faster-rcnn.ruqiang826/caffe-fast-rcnn/src/caffe/layers/

然后编译还是会错，出错文件src/caffe/layers/cudnn_conv_layer.cu.里面有两个带v3后缀的函数，把"_v3"去掉,就能编译过了。这是py-faster-rcnn自带caffe支持cudnn 5的最小修改集合
从github上下载Makefile.config. 或者从这个repository下。
然后 make -j8 ;make pycaffe

2. 
在py-faster-rcnn/lib 目录make

 编译时有个python头文件pyconfig.h找不到，需要 
export CPLUS_INCLUDE_PATH=/usr/include/python2.7

3. 运行
用脚本下载下来data目录的VOCdevkit2007 ，并下载data/imagenet_models，就可以运行了。
./experiments/scripts/faster_rcnn_end2end.sh  0 VGG_CNN_M_1024 pascal_voc
修改./experiments/scripts/faster_rcnn_end2end.sh 的iter数量，如果不想用imagenet的模型初始化，可以把train的--weight那一行删掉。然后执行。
！！！这里的weight初始化，其实不仅仅是神经网络的权重，还包括一些hyperparameter、设置等等。如果不用这个imagenet的模型初始化，训练不出来的。这里走了很多弯路。我一直以为自己train的model和imagenet完全不同，所以就把这个初始化删除了。怎么也训练不出来像样的东西。找了很久才发现这个原因。！！！

4. demo
修改 tools/demo.py 来读入训练的模型，并画图。可以看commit的代码

5. 
改成自己的数据,4分类(含background)： 把84都改成16, 把21都改成4
修改 models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/train.prototxt ， 把bbox_pred 的num_output 改成 16,  input-data 的param_str: "'num_classes': 4",  roi-data 的 num_classes 也改成4.
修改 models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt 内容与train 类似.

修改lib/datasets/pascal_voc.py 关于类的名字，和数据集合一致。我这里的四个类是 self._classes = ('__background__', 'king', 'eking', 'giant')
修改 tools/demo.py 的CLASS，内容与上面pascal_voc.py类似。
把data目录下的数据替换成自己的训练数据


6. cache
这样，train 和test都可以跑过了。
另外注意train和test都会生成cache文件，两次运行改了图片内容，注意删掉cache，否则会报错。
这是train的cache：/home/ruqiang/github/py-faster-rcnn_ruqiang826/data/cache/voc_2007_trainval_gt_roidb.pkl
test cache :      /home/ruqiang/github/py-faster-rcnn_ruqiang826/data/VOCdevkit2007/annotations_cache/annots.pkl

最后如何运行，看run.sh 


## 7. 代码栈
1. tools/train_net.py  是入口  
2. 首要目标是找到train数据的逻辑。  
  train_net.py   
    -> from datasets.factory import get_imdb  这里其实执行了factory.py，初始化了imdb的__sets。注意初始化只是一个lambda表达式，下面get_imdb的时候才执行lambda表达式.  
    -> tools/train_net.py:combined_roidb  ..  
    -> tools/train_net.py:get_roidb   ..  
      -> lib/datasets/factory.py:get_imdb  只是返回了刚才初始化好的imdb sets中的一个元素。这里执行了lambda表达式，获取了一个pascal_voc的类。先执行了imdb的init函数，又执行自己的init函数  
      -> set_proposal_method 这是pascal_vod的基类imdb的函数，在lib/datasets/imdb.py， 这里把roidb_handler 设置成了 gt_roidb，稍后就会执行到，这个是数据处理的关键。  
      -> get_training_roidb 在 lib/fast_rcnn/train.py。  
        -> imdb.append_flipped_images() 还是在imdb.py。   
          -> self.roidb[i]['boxes'].copy() 首次调用roidb，  
            -> imdb.py 的 roidb(self)。如果已经调用过，roidb就可以直接返回了。如果没有调用过，走下面：  
              -> self.roidb_handler() 这个就是gt_roidb了,这玩意在上面set的，找了好久.  
                -> lib/datasets/pascal_voc.py:gt_roidb()  
                  -> _load_pascal_annotation : 在VOCdevkit2007/VOC2007/Annotations 目录下，有每个jpg的xml标注，包括了object的范围和类别. 这个函数返回了每个图片的object 坐标、object类别、overlaps(第一维是object id，第二维是类别), object区域面积  
          -> 回到append_flipped_images()，这个是翻转图像，多得到两张图片  
      -> 回到 lib/fast_rcnn/train.py:get_training_roidb   
        -> lib/roi_data_layer/roidb.py:prepare_roidb:对每张图像增加了max_classes 和 max_overlaps两个变量。还不清楚这两个是做什么的。  
    -> 回到 tools/train_net.py:combined_roidb ,后面对多个imdb的情况做了处理，然后返回imdb、roidb。   
    -> 回到tools/train_net.py:main。数据处理完毕。  
    另外，数据标注在github上有个工具，tzutalin/labelImg，输出就是PASCAL VOC格式。  
  
3. main 的下一步是 lib/fast_rcnn/train.py:train_net  
  -> filter_roidb: 这里大致的逻辑是，如果是少数object，认为是foreground， 如果是多个同类型object ，认为是background 。无论那个，至少得有满足数量的object。这里没太仔细看。  
  fast rcnn的逻辑应该在上面跑通阶段，拷过去的两个cpp文件roi_pooling_layer. 这是自己的逻辑和caffe框架merge的地方。没有看这里。  
  
4. 先看test_net.py  
  找到test的地方，获取预测的边界的地方。用cv2.rectangle 和 cv2.imwrite 把画了框的图像保存下来。  

5. 标注自己的数据
    想替换自己的数据集，用labelImg标注了十几个
  


