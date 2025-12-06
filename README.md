<p align="center">

  <h1 align="center">GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting</h1>

  <p align="center">
    <a href="https://github.com/xieyuser/GS-LIVM/tree/main"><img src="https://img.shields.io/badge/-c++-black?logo=c%2B%2B&style=social" /></a>
    <a href="https://robostack.github.io/index.html"><img src="https://img.shields.io/ros/v/noetic/vision_msgs" /></a>
    <a href="https://arxiv.org/abs/2410.17084"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/xieyuser/GS-LIVM/tree/main/LICENSE"><img src="https://img.shields.io/badge/License-GPLv2-blue.svg?style=flat-square" /></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2410.17084">Paper</a> | <a href="https://www.youtube.com/watch?v=bVkwWXFCbVsE">Video</a> | <a href="https://www.youtube.com/watch?v=acTHQIK4QIs">Guideline</a> | <a href="https://www.bilibili.com/video/BV1JeSKYUEFp/?vd_source=0211af3bb655d5627b96e8718e3af59f">Bilibili</a> </h3>
  <div align="center"></div>
</p>




# GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting

<!-- [Paper](https://arxiv.org/pdf/2410.17084)

[Bilibili](https://www.bilibili.com/video/BV1JeSKYUEFp/?vd_source=0211af3bb655d5627b96e8718e3af59f) -->

<div align="center">
<a href="https://www.youtube.com/watch?v=bVkwWXFCbVsE" target="_blank"><img src="./doc/demo.png" alt="video" width="100%" /></a>
</div>


## Features

A. Real-time Dense Mapping via 3D Gaussian Splatting in large-scale unbounded outdoor environments.

B. Conventional LiDAR-Inertial-Visual Odometry (LIVO) is used to estimate the pose of the sensor.

C. Self-designed Voxel Gaussian Process Regression (VGPR) is used to handle the spaisity of the LiDAR data.

D. Variance centerd Framework is developed to calculate the initialization parameters of 3D gaussians.

E. Easy-to-use. ROS-related code is provided. Any bags contains image, LiDAR points, IMU can be processed.


## 1.Overview and Contributions (2024-10-23 Update)
The system takes input from point cloud data collected by LiDAR, motion information collected by an Inertial Measurement Unit (IMU), and color and texture information captured by a camera. In the tracking thread, the ESIKF algorithm is used for tracking, achieving odometry output at the IMU frequency. In the mapping thread, the rendered color point cloud is used for Voxel-GPR, and then the data initialized 3D gaussian is input into the dense 3D gaussian map for rendering optimization. The final output is a high-quality dense 3D gaussian map. C, D, and S represent the rasterized color image, depth image, and silhouette image, respectively.

<div align="center">
<img src="./doc/overview.png" width=80.0% />
</div>


## Images & Demo Video (2024-10-01 Update)

<!-- <div align="center">
<img src="./doc/demo.png" width=80.0% />
</div> -->


<div align="center">
<img src="./doc/render-all.png" width=80.0% />
</div>

<div align="center">
<img src="./doc/render-supply-all.png" width=80.0% />
</div>

<div align="center">
<img src="./doc/hkucampus.gif" width=80.0% />
</div>

<!-- You can find the demo video [here](https://www.youtube.com/watch?v=j9Kne47aS_0&t=76s). -->

Demo video will be released soon.

## 2. Prerequisites

The equipment of this repository is as follows. And this repo contains **<u>CPP</u>**, **<u>TorchLib</u>** and **<u>ROS</u>** in **<u>conda</u>**, so maybe it's a little difficult to install. If you are not familiar with the following steps, you can refer to the video of environment deploy [Youtube](https://youtu.be/acTHQIK4QIs?si=oPEd01YCcR_YdLne) in and [Bilibili](https://www.bilibili.com/video/BV1qPx5eoELm/?vd_source=0211af3bb655d5627b96e8718e3af59f).

2.1 Ubuntu and ROS. 

We build this repo by [RoboStack](https://robostack.github.io/). You can install different ROS distributions in **Conda Environment** via [RoboStack Installation](https://robostack.github.io/). Source code has been tested in **ROS Noetic**. Building in **conda** may be more difficult, but the ability to isolate the environment is worth doing so.

2.2 Create conda environment
``` Bash
# create env
mamba create -n {ENV_NAME} python=3.9
mamba activate {ENV_NAME}

# install ros in conda
mamba install ros-noetic-desktop-full -c RoboStack
```

2.2 (Optional) Build Livox-SDK2 & livox_ros_driver2 in conda
``` bash
# download
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src

git clone https://github.com/Livox-SDK/Livox-SDK2
cd Livox-SDK2 && mkdir build && cd build

# cmake options, -DCMAKE_INSTALL_PREFIX is path of your conda environment
cmake -DCMAKE_INSTALL_PREFIX=/home/xieys/miniforge3/envs/{ENV_NAME}  ..

# make && make install
make -j60 && make install

#clone livox_ros_driver2 and put it in your catkin_ws/src. If you don not use Livox, you can skip this step by changing -DBUILD_LIVOX=OFF in CMakeLists.txt
cd ~/catkin_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2

cd livox_ros_driver2

(Important)(****NOTE, I have chaned the source code in livox_ros_driver2/CMakeLists.txt to support build. Please refer to the video in this operation.)

./build.sh ROS1
```

2.3 (Important) Install Torch
``` Bash
mamba search pytorch=2.0.1

# Please find appropriate version of torch in different channels
mamba install pytorch=2.0.1 -c conda-forge
```

2.4 Some packages can be installed by:
``` Bash
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src

# clone
git clone https://github.com/xieyuser/GS-LIVM.git

# install other packages
cd GS-LIVM
mamba install --file conda_pkgs.txt -c nvidia -c pytorch -c conda-forge
```

## 3. Build GS-LIVM and Source
Clone the repository and catkin_make:
``` Bash
# build
cd ~/catkin_ws
catkin build   # change some DEFINITIONS

# source
# (either) temporary
source ~/catkin_ws/devel/setup.bash

# (or) start with conda activate
echo "ROS_FILE=/home/xieys/catkin_ws/devel/setup.bash
if [ -f \"\$ROS_FILE\" ]; then
    echo \$ROS_FILE
    source \$ROS_FILE
fi" >> ~/miniforge3/envs/{ENV_NAME}/setup.sh
```


## 4.Run on Public Datasets

```Bash
# Noted: change the path in  line 40 of /home/xieys/catkin_ws/src/GS-LIVM/include/gs/gs/parameters.cuh
std::filesystem::path output_path = "/home/xieys/catkin_ws/output";

# for saving results
mkdir /home/xieys/catkin_ws/output/training -p

# for visualize
mv ${ROOT}/doc/sparse /home/xieys/catkin_ws/output
```



###  1). Run on [*R3Live_Dataset*](https://github.com/ziv-lin/r3live_dataset)

Before running, please type the following command to examine the image message type of ROS bag file:

```bash
rosbag info SEQUENCE_NAME.bag
```

If the image message type is **sensor_msgs/CompressedImage**, please type:

```bash
# for compressed image sensor type
roslaunch gslivm livo_r3live_compressed.launch
```

If the image message type is **sensor_msgs/Image**, please type:

```bash
# for original image sensor type
roslaunch gslivm livo_r3live.launch
```


###  2). Run on [*NTU_VIRAL*](https://ntu-aris.github.io/ntu_viral_dataset/)

```bash
roslaunch gslivm livo_ntu.launch
```

###  3). Run on [*FAST-LIVO*](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/Esiqlmaql0dPreuOhiHlXl4Bqu5RRRIViK1EyuR4h1_n4w?e=fZdVn0)


```bash
roslaunch gslivm livo_fastlivo.launch
```

###  4). Run on [*Botanic Garden Dataset*](https://github.com/robot-pesg/BotanicGarden)

Please go to the workspace of **GS-LIVM** and type:

```bash
# for Velodyne VLP-16
roslaunch gslivm livo_botanic_garden.launch

# for Livox-Avia
roslaunch gslivm livo_botanic_garden_livox.launch
```

## 5.Visualization
Please refer to [Gaussian-Splatting-Cuda](https://github.com/MrNeRF/gaussian-splatting-cuda) to build SIBR_viewer to visualize the 3D gaussian model. Certainly it can be built in the same conda environment. I have installed the dependencies (cos7) in *conda_pkgs.txt*.

## Acknowledgments

Thanks for [RoboStack](https://robostack.github.io/GettingStarted.html), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Gaussian-Splatting-Cuda](https://github.com/MrNeRF/gaussian-splatting-cuda), [depth-diff-gaussian-rasterization
](https://github.com/ingra14m/depth-diff-gaussian-rasterization), [R3LIVE](https://github.com/hku-mars/r3live), [CT-ICP](https://github.com/jedeschaud/ct_icp), [sr_livo](https://github.com/ZikangYuan/sr_livo), [Fast-LIO](https://github.com/hku-mars/FAST_LIO) and [Open-VINS](https://github.com/rpng/open_vins).


## Citation
```
@article{xie2024gslivm,
  title={{GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting}},
  author={Xie, Yusen and Huang, Zhenmin and Wu, Jin and Ma, Jun},
  journal={arXiv preprint arXiv:2410.17084},
  year={2024}
}
```

## License
The source code of this package is released under GPLv3 license. For any technical issues, please feel free to contact yxie827@connect.hkust-gz.edu.cn.

