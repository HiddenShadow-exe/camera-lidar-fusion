## Preview camera image
https://github.com/realsenseai/librealsense/releases

## Code used to get camera data
https://github.com/XuanKyVN/intel-realsens-camera-with-Python-YoloV11

### Before filters
![](images/image-before.png)
![](images/image-before-2.png)

### After filters
![](images/image-after.png)
![](images/image-after-2.png)

## Measuring depth accuracy
### Short range (Box distance: 1m)
![](images/depth-acc-1.png)
* Reading: 0.41 - 0.43 meters
* Actual: 0.42 meters

### Mid range (Box distance: 2m)
![](images/depth-acc-2.png)
* Reading: 0.40 - 0.45 meters
* Actual: 0.41 meters

### Long range (Box distance: 3m)
![](images/depth-acc-3.png)
* Reading: 0.29 - 0.42 meters
* Actual: 0.42 meters

## Simple box detection
![](images/box-detection.png)

## Box sidewall completion
In this picture, we can see the raw output of the depth camera converted to a mesh file.
![](images/box-side-completion-before.png)

After the algorithm runs, the sides of the box are inserted into the point cloud (no faces in the picutre this time).
![](images/box-side-completion.png)

## Multi object detection
Edge mask from depth data:
![](images/detection-multi-mask.png)

Depth data with detected boxes shown:
![](images/detection-multi-depth.png)

## Multi object, multi height sidewall completion
Before:
![](images/side-completion-multi-before.png)

After:
![](images/side-completion-multi-after.png)