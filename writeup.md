## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortedImage.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binaryThreshold_test1.png "Binary Example"
[image4]: ./output_images/binaryThreshold_test1_warped.jpg "Warp Example"
[image5]: ./output_images/drawn_lines.jpg "Fit Visual"
[image6]: ./output_images/test5.jpg "Output"
[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this can be found in advanced_lane_lines.ipynb. The class "PreprocessImage" provides helper functions to calibrate the camera and undistort the image. The user of the class has to first populate the objpoints using the call "populate_calibration_points". Once all the obj points are generated, "calibrate_camera" has to be called to generate the calibration matrix. Finally, the member function call "undistort_image" is used to undistort Images. The below Image is an example for preprocessed iamge.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (function binary_threshold in `advanced_lane_lines.ipynb`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transformed is defined in WarpImage. Member function `create_warped` is used to create warped image and `create_unwarped` to unwarp and image. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[552,462], [760,462] , [1350,668],[140,668]])
        
        
dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 552, 462      | 100, 100      | 
| 760, 462      | 1180, 100     |
| 1350, 668     | 1180, 710     |
| 140, 668      | 100, 710      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I have used sliding window method to find the lane-line pixels

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature using the function call `calcCurvature`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The class `DemaracteLanes` wraps and abstracts all preprocessing of an image. The image below shows a sample image with lanes drawn:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have used sliding window method to identify the lanes. Initially, I faced an issue when the lane lines overflown the area of intereset. I am listing down few of the areas where the implementation has shortcoming. 
* Radius of curvature of 2 lanes do not match
* If the lanes near the base of the image has broken lines, algorithm finds it difficut to correct the starting position
* The lanes in frames of the video are not smooth.

The current implementation might fail when there are varying brightness in the images. 
The algorithm might also have issues on old roads where the lane markings become dull. 
If there is another vehicle on the same lane or a vehicle driving on close by lanes the pipeline is likely to fail to identify the left and right base position.

To overcome this, region of intereset should be masked. On the filtered image, binary thresholding should be performed.


