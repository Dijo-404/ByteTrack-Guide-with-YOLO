## ByteTrack-Guide-with-YOLO
This is a guide for using ByteTrack along with YOLO for object detection and tracking 


## Step-1 : Clone repo and execute the following commands-
```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
pip install cython lap
pip install cython_bbox
```
if error occours with ONYX use the this repo's requirements.txt ive modified the versions.
```
# Old ONYX versions 
onnx==1.8.1
onnxruntime==1.8.0
onnx-simplifier==0.3.5
# Modified ONYX versions
onnx==1.13.0
onnxruntime==1.16.3
onnx-simplifier==0.4.33
# ive already changed it soo dw this is just info on the changes ive done.
```
## Step-2 : Install yolox (needed for running byte with yolo)-
```bash
cd ~/ByteTrack/YOLOX
pip install -e .
```

thats all there is to bytetrack installation , ive attached a sample code above for testing purpose refer the code.
check out DeepSort and well as yolo tracker 

```
hope u have installed packages like ultralytics , numpy , cv2
```
