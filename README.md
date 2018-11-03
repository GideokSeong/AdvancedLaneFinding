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

[image1]: ./output_images/UndistortedCalibrationImage1.jpg
[image2]: ./output_images/UndistortedTestImage1.jpg
[image3]: ./output_images/BinaryTestImage1.jpg
[image4]: ./output_images/WarpedTestImage1.jpg
[image5]: ./output_images/PolynomialTestImage1.jpg
[image6]: ./output_images/PlottedBackTestImage1.jpg
---
### Camera Calibration and Undistort Image

For this part, first I used ```cv2.findChessboardCorners(gray, (9, 6), None)``` function to find the ```corner``` about each camera calibration image and put that value into ```imgpoints``` list which is used for undisorting the image. Same as the corner variable, ```objp``` variable is used to append into ```objpoints``` list. In this project, I used 20 camera calibration images to define more objective image points and object points. 

![CalibrationImage][image1]


Following image is undistorted image about test image. To do that, first of all, I needed to convert image to gray scale image using ```gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)``` then followed by ```cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1] , None, None)``` function which uses objpoints and imgpoints I got above. Finally next function ```cv2.undistort(test_img, mtx, dist, None, mtx)``` is utilized with image points and object points. 
    

![UndistortedTestImage][image2]


### Thresholded Binary Image

To convert to binary image, I used sobel, magnitude, and direction of the gradient. First, sobel function, ```abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))``` x or y gradient is applied with sobel function and then depending on the threshold values, I can get the binary output. Next, I apply magnitude of the gradient. This function is similar to the sobel function except for the magnitude function, ```abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)``` In other words, both sobelx and sobely are used to sobelxy. Also, direction of the gradient is taken advantage of here. Other steps are the same as previous but ```direction = np.arctan2(abs_sobely, abs_sobelx)```. Lastly, ```HLS``` threshold was used here to capture the object better. All of those function can be put together with different thresholds values like following way ```combined[(abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1] = 1``` 


![BinaryImage][image3]


### Transformed Image
    
After applied above different gradient functions, I am able to grasp ```WarpedPerspective``` Image. In this step, one thing you have to be cautious is you need to define proper source points and destination points. Otherwise, next progress will not be going well. First, you can get ```M = cv2.getPerspectiveTransform(src, dst)``` the perspective transform matrix. ```M``` value is utilized in following function ```warped = cv2.warpPerspective(test_img, M, img_size)``` to seek a warped and perspective image. 
    
![BinaryImage][image4]
Above image is undistorted and transformed image after seizing on the destination points.


### Polynomial Image

Once it finds non-zero points both coordination of x and y, it sets the area of search based on activated x and y values like following equation. ```left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) ``` Then, extract left and right line pixel positions for new polynomials ```left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)```. Finally, I draw the windows based on those left and right line pixel values. 
![BinaryImage][image5]


### Plotted Back Image

In this step, I am using the previous function which is ```leftx,lefty,rightx,righty,out_img = search_around_poly(top_down)```, but for the better ease of calculating offset from the center and curvature of the each lane, I separately implemented ```left_fitx, right_fitx, ploty, center_dist= poly(top_down)``` function. To apply pixel data into real world data, I use ```left,right = measure_curvature_real(out_img,leftx,lefty,rightx,righty)``` function. Once, I get the curvature and offset value, I draw the line onto the warped blank image.
![BinaryImage][image6]