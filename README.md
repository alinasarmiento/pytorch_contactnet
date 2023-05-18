# Pytorch Contact-GraspNet
Pytorch implementation of Contact-GraspNet. Original Tensorflow implementation can be found at: https://github.com/NVlabs/contact_graspnet

DISCLAIMER: This is research code I'm using as a stepping stone toward a different project - your mileage may vary. I'll update this repo periodically but am not actively maintaining it. That being said, please raise issues when you find them and I'll try and resolve them as quickly as I can.

## General notes
The train file is a little broken and will be updated soon along with tools to evaluate the model in Pybullet and on a real world setup. Stay tuned for a cleaner datloader + instructions for interfacing with ACRONYM as well.

Right now, a current checkpoint of the weights that I've been using is in `/checkpoints/current.pth` and can be loaded and used as seen in the demo file.

## Demo file

To see a demo of the predictions, first download the requirements:
`pip3 install -r requirements.txt`
We're doing our visualizations in MeshCat. In a separate tab, start a meshcat server with `meshcat-server`

From here you can run `python3 eval.py`

To visualize different confidence threshold masks on the grasps, use the `--threshold` argument such as `--threshold=0.8`

To visualize different cluttered scenes (8-12 tabletop objects rendered in pyrender from ACRONYM), use the argument `--scene` and manually feed in a file name such as `--scene=002330.npz`. Sorry that this is so inconvenient right now. Your possible files are:
- 002330.npz
- 004086.npz
- 005274.npz

## Predicting grasps on your own pointclouds
The model should work on any pointcloud of shape (Nx3). For most consistent results, please make sure to put the pointcloud in the world frame and center it by subtracting the mean. Do not normalize the pointcloud to a unit sphere or unit box, as "graspability" naturally changes depending on the size of the objects (so we don't want to lose that information about the scene by scaling it).

Happy grasping!
