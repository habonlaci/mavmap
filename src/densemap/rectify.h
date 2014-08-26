#ifndef _RECTIFY_H
#define _RECTIFY_H


#include "base3d/projection.h"
#include "base2d/image.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/contrib.hpp>
#include <cstddef>
#include <opencv2/gpu.hpp>
#include <fstream>


#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "base2d/image.h"
#include <Eigen/Core>
#include <vector>

/**
 * @brief rectifies an image pair
 *
 * @param calib_matrix_ camera calibration matrix
 * @param camera_poses_ poses of the two cameras
 * @param lidx left image indes
 * @param ridx right image index
 * @return void
 **/
void rectify_images (const Eigen::Matrix3d& calib_matrix_, 
		     const std::vector<Image>& camera_poses_,
		     int lidx, int ridx );


#endif