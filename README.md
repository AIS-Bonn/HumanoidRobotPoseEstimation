# Real-time Pose Estimation from Images for Multiple Humanoid Robots
This is the implementation of Real-time Pose Estimation from Images for Multiple Humanoid Robots, RoboCup Symposium 2021. 

## Dataset
The HumanoidRobotPose dataset provided in this repository also can be downloaded as zip file from [here](https://www.ais.uni-bonn.de/~hfarazi/RC2021/hrp.zip).
The dataset format is the same as [the COCO dataset](https://cocodataset.org/#format-data).

To use the dataset, install pycocotools:
```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## Training
To train the model, run:
```
python src/main.py --config experiments/train.yaml --output_dir output_path --dataset_path dataset/hrp/
```

## Evaluation
To evaluate the model, run:
```
python src/main.py --eval --config experiments/test.yaml --dataset_path dataset/hrp/ --checkpoint checkpoint_path
```

The pretrained model is available [here](https://www.ais.uni-bonn.de/~hfarazi/RC2021/checkpoint.pth).

## Baselines
- https://github.com/CMU-Perceptual-Computing-Lab/openpose
- https://github.com/princeton-vl/pose-ae-train
- https://github.com/openpifpaf/openpifpaf
- https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
