#ifndef _REPROJECT_H
#define _REPROJECT_H

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Core>

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/contrib.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu.hpp>

#include <cstddef>

#include "util/defs.h"






using namespace cv;

/**
 * @brief Removes non number values from disparity map like inf or -inf
 *
 * @param disparity the disparity map to be filterd
 * @return void
 **/
void filterDisparity(cv::Mat& disparity);

/**
 * @brief 		Reprojects a disparitymap to 3d coordinates
 *
 * @param Q 		reprojection matrix computed by cv::stereorectify();
 * @param disparity 	disparitymap
 * @param _3dimage 	3d points
 * @return void
 **/
void reprojectTo3d(const cv::Mat& Q, const cv::Mat& disparity, cv::Mat& _3dimage);

/**
 * @brief writes pointcloud to VRML file
 *
 * @param _3dimage points in X, Y, Z format
 * @param imageSize size of the disparity image
 * @return void
 **/
void writeVRML(const cv::Mat& _3dimage);


#endif




