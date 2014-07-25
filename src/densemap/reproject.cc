#include "reproject.h"


void filterDisparity(cv::Mat& disparity)
{
  
  for (int i = 0; i < disparity.rows; i++)
  {
    for (int j = 0; j < disparity.cols; j++)
    {
      if(isnan(disparity.at<short>(i,j))||disparity.at<short>(i,j)<0)
	disparity.at<short>(i,j)=0;
      
    }
    
  }
  
  
}//filterDisparity





void reprojectTo3d(const cv::Mat& Q, const cv::Mat& disparity, cv::Mat& _3dimage)
{
  
  double Q03, Q13, Q23, Q32, Q33;
  
  Q03 = Q.at<double>(0,3);
  Q13 = Q.at<double>(1,3);
  Q23 = Q.at<double>(2,3);
  Q32 = Q.at<double>(3,2);
  Q33 = Q.at<double>(3,3);

  
  
  double px, py, pz;
  
//TODO check disparity type  
  
  
  for (int i = 0; i < disparity.rows; i++)
  {
    for (int j = 0; j < disparity.cols; j++)
    {
      short d= disparity.at<short>(i,j);
      if ( d == 0 ) continue;
      double pw = 1.0 * static_cast<double>(d) * Q32 + Q33; 
      px = static_cast<double>(j) + Q03;
      py = static_cast<double>(i) + Q13;
      pz = Q23;
      
      px = px/pw;
      py = py/pw;
      pz = pz/pw;
      
      //std::cout<<px<<std::endl;
      
      _3dimage.at<Vec3f>(i,j)[0] = static_cast<float>(px);
      
      
      //cv::waitKey(0);
      
      
      _3dimage.at<Vec3f>(i,j)[1] = static_cast<float>(py);
      _3dimage.at<Vec3f>(i,j)[2] = static_cast<float>(pz);
      
    }
  }
 
 
 //std::cout<<_3dimage<<std::endl;
 
}//reprojectTo3d

void writeVRML(const cv::Mat& _3dimage)
{
    
    Size imageSize=_3dimage.size();
    std::ofstream pointcloud;
    
    pointcloud.open("points.wrl");
     
    pointcloud << "#VRML V2.0 utf8\n";
    pointcloud << "Background { skyColor [1.0 1.0 1.0] } \n";
    pointcloud << "Shape{ appearance Appearance {\n";
    pointcloud << " material Material {emissiveColor 1 1 1} }\n";
    pointcloud << " geometry PointSet {\n";
    pointcloud << " coord Coordinate {\n";
    pointcloud << "  point [\n";
    
    for(int i=0; i<imageSize.height; i++)
    {
	 for(int j=0; j<imageSize.width; j++)  
	 {
	      pointcloud<<_3dimage.at<Vec3f>(i,j)[0]<<", "
			<<_3dimage.at<Vec3f>(i,j)[1]<<", "
			<<_3dimage.at<Vec3f>(i,j)[2]<<"\n";
	 }
     }   
     pointcloud<<"\n] }}}\n";
     //pointcloud<<" color Color { color [\n";
     pointcloud.close();
}//writeVRML