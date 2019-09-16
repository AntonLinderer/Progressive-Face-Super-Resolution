# Progressive Face Super Resolution
Deokyun Kim, Minseon Kim, Gihyun Kwon, and Dae-shik Kim, [Progressive Face Super-Resolution via Attention to Facial Landmark](https://arxiv.org/abs/1908.08239), The British Machine Vision Conference 2019 (BMVC 2019)


### Prerequisites
* Python 3.6
* Pytorch 1.0.0
* CUDA 9.0 or higher

This code support [NVIDIA apex-Distribute Training in Pytorch](https://github.com/NVIDIA/apex), please follow description. 

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
