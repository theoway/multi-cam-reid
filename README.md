## Multiple people reidentification and pose estimation across multiple camera setup in real-time
This projects uses RTSP from multiple cameras and tracks people with in it, also reidentifying them across multiple cameras and estimating their poses in real-time. This project is compatible with **Python >= 3.6** and uses both _Pytorch_ and _Tensorflow_ at its backend. It is built on top of these amazing projects:
- [Multi camera people tracking and reidentifcation](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification): This repo provides an implementation but it had to be modified for real-time use case. For this, **multiprocessing**, **shared memory objects** and **multithreading** has been used.
- [Openpose-pytorch implementation](https://github.com/Hzzone/pytorch-openpose): This implementation estimates poses of re-identified people with **18 keypoints**.

### Before running the code:
- Download this [model data](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification) folder and it to your local repo with the same name.
- Download this [model](https://drive.google.com/file/d/1EtkBARD398UW93HwiVO9x3mByr0AeWMg/view?usp=sharing) and add it to `model_data/models/`
- Download the [body_pose_model.pth](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG). Create a folder and name it `model`. Add the downloaded body pose model there.
- Run `pip install -r requirements.txt`

### Running the code:
- To run the code with multiple RTSP feeds: `python demo.py -u host1:port1 host2:port2 ...` \
  _(*No need to prefix http/https with host names)_
- To debug the app and use computer webcam: `python demo.py -u 0`

### Results and performance:

https://user-images.githubusercontent.com/41969735/174279980-23926ae0-07fe-4213-85f7-f6e55e3d6a73.mp4


On my setup (i5 8th gen, 8 Gb RAM and NVIDI GTX 1650), I got 10 fps without pose estimation and 2 fps with pose esimtation. With better hardware, it can be improved.

### Contributing:
- Raise an issue with mentioning the issue being faced, your system specifications along with OS and Python version
- PRs are welcome! 
