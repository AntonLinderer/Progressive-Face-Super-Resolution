# Progressive Face Super-Resolution
Deokyun Kim, Minseon Kim, Gihyun Kwon, and Dae-shik Kim, [Progressive Face Super-Resolution via Attention to Facial Landmark](https://arxiv.org/abs/1908.08239), The British Machine Vision Conference 2019 (BMVC 2019)

## Overview
![our structure](./figure/our_structure.png)

> **Progressive Face Super-Resolution via Attention to Facial Landmark**<br>
> Deokyun Kim (KAIST), Minseon Kim (KAIST), Gihyun Kwon (KAIST), et al.<br>
> **Abstract:** *Face Super-Resolution (SR) is a subfield of the SR domain that specifically targets the reconstruction of face images. The main challenge of face SR is to restore essential facial features without distortion. We propose a novel face SR method that generates photo-realistic 8× super-resolved face images with fully retained facial details. To that end, we adopt a progressive training method, which allows stable training by splitting the network into successive steps, each producing output with a progressively higher resolution. We also propose a novel facial attention loss and apply it at each step to focus on restoring facial attributes in greater details by multiplying the pixel difference and heatmap values. Lastly, we propose a compressed version of the state-of-the-art face alignment network (FAN) for landmark heatmap extraction. With the proposed FAN, we can extract the heatmaps suitable for face SR and also reduce the overall training time. Experimental results verify that our method outperforms state-of-the-art methods in both qualitative and quantitative measurements, especially in perceptual quality*


### Prerequisites
* Python 3.6
* Pytorch 1.0.0
* CUDA 9.0 or higher

This code support [NVIDIA apex-Distributed Training in Pytorch](https://github.com/NVIDIA/apex), please follow description. 
Also, we refered state-of-the-art [Face Alignment Network](https://github.com/1adrianb/face-alignment) in order to get face SR-oriented facial landmark heatmap.

### Data Preparation

* [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

create a folder:

```bash
 mkdir dataset

```
and then, download dataset. Anno & Img.


#### Train model

* Pytorch parallel training(or none parallel) 
```bash
$ python train.py --data-path './dataset'\
                  --workers $NUM_WORKERS \
                  --gpu-ids $GPU_ID(s)\
                  --lr $LEARNING_RATES

```

* Pytorch distributed training
```bash
$ python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=4 train.py --distributed \
                                                                                       --data-path './dataset'\
                                                                                       --lr $LEARNING_RATES
```
(<b>nproc_per_node</b>: number of GPUs using to training.)

#### Test model
```bash
$ python eval.py --data-path './dataset' --checkpoint-path 'CHECKPOINT_PATH/****.ckpt'
```

## Acknowledgement

This work was supported by Engineering Research Center of Excellence (ERC) Program supported by National Research Foundation (NRF), Korean Ministry of Science & ICT (MSIT) (Grant No. NRF-2017R1A5A1014708), Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2016- 0-00563, Research on Adaptive Machine Learning Technology Development for Intelligent Autonomous Digital Companion) and Hyundai Motor Company through HYUNDAITECHNION-KAIST Consortium.
