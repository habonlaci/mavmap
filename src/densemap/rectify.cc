#include "rectify.h"
#include "reproject.h"


   


using namespace std;
using namespace cv;
using namespace Eigen;

void rectify_images(const Matrix3d& calib_matrix_, 
		    const vector<Image>& camera_poses_, 
		    int lidx, int ridx )
{

    Mat cvCalibMatrix;
    eigen2cv(calib_matrix_,cvCalibMatrix);
    

    Vector3d rvec, tvec;
    Matrix<double, 3, 4> right_Rt, left_Rt, delta_Rt;
    Matrix<double, 4, 4> right_Rt4 = Matrix4d::Identity(), 
			  left_Rt4 = Matrix4d::Identity(), 
			  delta_Rt4 = Matrix4d::Identity(),  
			  left_Rt4i = Matrix4d::Identity(),
			  right_Rt4i = Matrix4d::Identity();


    Mat dist_coeffs = Mat::zeros(1,8,CV_32F);
//Mat rimage, limage;								//resizing
    Mat const rimage = camera_poses_[ridx].read();				//rimageBig = camera_poses_[ridx].read();
//resize(rimageBig, rimage, Size(), 0.4, 0.4, INTER_LINEAR);
    Mat const limage = camera_poses_[lidx].read();
//resize(limageBig, limage, Size(), 0.4, 0.4, INTER_LINEAR);
    
    
    Size imageSize = rimage.size();
    Mat roi= Mat::zeros(imageSize.height, imageSize.width, CV_8U), lroi, rroi, roirect;

    //TODO IMPLEMENT PARAMETER FROM TERMINAL
    rectangle(roi, Point(15,15),Point(imageSize.width-15,imageSize.height-15),CV_RGB(255, 255, 255) ,CV_FILLED);
   
    rvec[0] = camera_poses_[ridx].roll;
    rvec[1] = camera_poses_[ridx].pitch;
    rvec[2] = camera_poses_[ridx].yaw;

    tvec[0] = camera_poses_[ridx].tx;
    tvec[1] = camera_poses_[ridx].ty;
    tvec[2] = camera_poses_[ridx].tz;

    
    right_Rt = compose_Rt_matrix (rvec, tvec);

    rvec[0] = camera_poses_[lidx].roll;
    rvec[1] = camera_poses_[lidx].pitch;
    rvec[2] = camera_poses_[lidx].yaw;

    tvec[0] = camera_poses_[lidx].tx;
    tvec[1] = camera_poses_[lidx].ty;
    tvec[2] = camera_poses_[lidx].tz;

    
    left_Rt = compose_Rt_matrix (rvec, tvec);
    
    right_Rt4.block<3,4>(0,0)= right_Rt;
    left_Rt4.block<3,4>(0,0)= left_Rt;
    
    
    left_Rt4i = left_Rt4.inverse();
    right_Rt4i = right_Rt4.inverse();
    
       
    delta_Rt4 = right_Rt4i * left_Rt4i.inverse();	
    
    


      
    Matrix3d eigR = delta_Rt4.block<3,3>(0,0); 
    Vector3d eigT = delta_Rt4.block<3,1>(0,3);
         
    
    
    Mat R1, R2, P1, P2, Q, cvR, cvT;
    eigen2cv(eigR,cvR);
    eigen2cv(eigT,cvT);
       
//cvR=cvR*0.4;						//rescaling
//cvT=cvT*0.4;
//cvCalibMatrix=cvCalibMatrix*0.4;
    
    Rect validRoi[2];
    stereoRectify (cvCalibMatrix, dist_coeffs,
                       cvCalibMatrix, dist_coeffs,
		       imageSize,
                       cvR, cvT,
		       R1,R2,P1,P2, Q,
		       CALIB_ZERO_DISPARITY,
		       -1,
		       imageSize, &validRoi[0], &validRoi[1]);
    
//cout<< validRoi[0]<< endl << validRoi[1,1]<< endl;
    
    
   // Mat * limageptr = limage;
    

    
    waitKey(0);


    Mat maplx, maply, disp(imageSize.width, imageSize.height, CV_32F), disp8, disp_eq, disp_small;
    cout<<endl<<"Type: "<<disp.type()<< endl;
 

    initUndistortRectifyMap(cvCalibMatrix, dist_coeffs, R1, P1, imageSize, CV_32F, maplx, maply); 
    Mat limagerect ;
    remap(limage, limagerect, maplx, maply, INTER_LINEAR);
    remap(roi, lroi, maplx, maply, INTER_LINEAR);

    
   
    
    
    initUndistortRectifyMap(cvCalibMatrix, dist_coeffs, R2, P2, imageSize, CV_32F, maplx, maply); 
    Mat rimagerect ;
    remap(rimage, rimagerect, maplx, maply, INTER_LINEAR);
    remap(roi, rroi, maplx, maply, INTER_LINEAR);

    bitwise_and(lroi,rroi,roirect);
    
    
 /*   
    //Check for rectification in X or Y direction 
    if(tvec[1]>tvec[0])
    {
      Mat rimagerectT, limagerectT;
      transpose(rimagerect, rimagerectT);
      transpose(limagerect, limagerectT);
      limagerect=limagerectT;
      rimagerect=rimagerectT;
    }
    
   */
    
    imshow("rimgrect", rimagerect);
    imshow("limgrect", limagerect);
    imwrite("rightrect.bmp",rimagerect);
    imwrite("leftrect.bmp",limagerect);
    
    
    
    
    int depthmapMethod=1;  //	1: cpu sbgm	mukodik mindennel
		   //	2: gpu bm	
		   //	3: cpu bm
	
    int reprojection=3;  	//1: Opencv GPU reproject function 
				//2: Opencv CPU reprojection function
				//3: Own reprojection function
   
    

    switch(depthmapMethod)
    {
      
      
      case 1: //cpu sgbm
      {
	  //********************************************PARAMETERTUNER  CPU SGBM *******************************************************   
	      //moreorless good parameters    
	      int SADWindowSize = 1;	//1 volt 16 al kevesebb hiba a 22-23 kepnel
	      int numDisparities = 6;    //%16=0  (!!!)
	      int preFilterCap = 0;
	      int minDisparity = 0;
	      int uniquenessRatio= 5;
	      int speckleWindowSize = 0;
	      int speckleRange = 0;
	      int disp12MaxDiff = 0;
	      int fullDP = true;
	      int P11 = 8;
	      int P22 = 32;
	    

	      namedWindow("CPU SGBM", 1);
	      
	      
	      
	      createTrackbar( "SADWindowSize ", "CPU SGBM", &SADWindowSize, 96, 0 );
	      createTrackbar( "numDisparities *16", "CPU SGBM", &numDisparities, 16, 0 );
	      createTrackbar( "preFilterCap", "CPU SGBM", &preFilterCap, 1000, 0 );
	      createTrackbar( "minDisparity", "CPU SGBM", &minDisparity, 100, 0 );
	      createTrackbar( "uniquenessRatio", "CPU SGBM", &uniquenessRatio, 15, 0 );
	      createTrackbar( "speckleWindowSize", "CPU SGBM", &speckleWindowSize, 200, 0 );
	      createTrackbar( "speckleRange", "CPU SGBM", &speckleRange, 10, 0 );
	      createTrackbar( "disp12MaxDiff", "CPU SGBM", &disp12MaxDiff, 100, 0 );
	      createTrackbar( "P1", "CPU SGBM", &P11, 128, 0 );
	      createTrackbar( "P2", "CPU SGBM", &P22, 128, 0 );
	      
	      while(1)
	      {
		Ptr<StereoSGBM> sgbm;
	      
		sgbm = createStereoSGBM(minDisparity, numDisparities*16, SADWindowSize,
				    P11, P22, disp12MaxDiff, preFilterCap, uniquenessRatio,
				    speckleWindowSize, speckleRange, fullDP);
		
		sgbm->compute(limagerect, rimagerect, disp);
	      
		
		
		char key=waitKey(0);
		if(key == 27)
		break;

	      
		double minVal, maxVal;
		
		//disp.convertTo( disp8, CV_8UC1, 255/(maxVal- minVal));
		//disp.convertTo(disp8, CV_8U);
		cv::normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);
		imwrite("disp_sgbm.bmp",disp8);  
		minMaxLoc( disp, &minVal, &maxVal );
		cout<<"min: "<<minVal<<" max: "<<maxVal<<endl;
		imshow("SGBM", disp8);
		
	      }//while
		//disp.convertTo()
		//disp = disp /(float)16.f;
	  
		
	      break;
      }//case 1
      
      case 2: //gpu bm
      {
	 
	    
	    int NumDisparities_GPU=13, //12
		BlockSize_GPU=12, 
		MinDisparity_GPU=0, 
		SpeckleWindowSize_GPU=5, //29
		SpeckleRange_GPU=8, //0
		Disp12MaxDiff_GPU=5, 
		PreFilterCap_GPU=0, 
		UniquenessRatio_GPU=5;
		
	    // //PARAMETERTUNER  GPU BM  
	    namedWindow("GPU BM", 1);
	    
	    
	    
	    createTrackbar( "numDisparities *16", "GPU BM", &NumDisparities_GPU, 32, 0 );
	    createTrackbar( "minDisparity", "GPU BM", &MinDisparity_GPU, 500, 0 );
	    createTrackbar( "BlockSize ", "GPU BM", &BlockSize_GPU, 96, 0 );
	    createTrackbar( "speckleWindowSize", "GPU BM", &SpeckleWindowSize_GPU, 200, 0 );
	    createTrackbar( "speckleRange", "GPU BM", &SpeckleRange_GPU, 10, 0 );
	    createTrackbar( "disp12MaxDiff", "GPU BM", &Disp12MaxDiff_GPU, 100, 0 );
	    createTrackbar( "preFilterCap", "GPU BM", &PreFilterCap_GPU, 1000, 0 );
	    createTrackbar( "uniquenessRatio", "GPU BM", &UniquenessRatio_GPU, 15, 0 );
	    
	    
	    gpu::GpuMat d_disp, d_disp_small, d_disp_16;
	    gpu::GpuMat d_left, d_right;

	    d_left.upload(limagerect);
	    d_right.upload(rimagerect);
	    
	    
	    
	    while(1)
	    {
	      Ptr<gpu::StereoBM> gpubm;
	      gpubm = gpu::createStereoBM(48,31);
	      
	      if(BlockSize_GPU%2==0) //block size has to be odd
		BlockSize_GPU++;
	      
	      gpubm->setNumDisparities(NumDisparities_GPU*16);
	      gpubm->setMinDisparity(MinDisparity_GPU);
	      gpubm->setBlockSize(BlockSize_GPU);
	      //gpubm->setSpeckleWindowSize(SpeckleWindowSize_GPU);
	      //gpubm->setSpeckleRange(SpeckleRange_GPU);
	      gpubm->setDisp12MaxDiff(Disp12MaxDiff_GPU);
	      //gpubm->setPreFilterType();
	      gpubm->setPreFilterCap(PreFilterCap_GPU);
	      //gpubm->setTextureThreshold();
	      gpubm->setUniquenessRatio(UniquenessRatio_GPU);
	      
	    
	      
	      gpubm->compute(d_left, d_right, d_disp);
	      
	      char key=waitKey(0);
	      if(key == 27)
	      break;
	      
	      if(imageSize.width>2000)
	      {
		gpu::resize(d_disp, d_disp_small, Size(), 0.2, 0.2,INTER_LINEAR);
		d_disp_small.download(disp);
		
		cv::normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);
		double minVal, maxVal;
		minMaxLoc( disp, &minVal, &maxVal );
		cout<<"min: "<<minVal<<" max: "<<maxVal<<endl;
		imshow("gpu_bm", disp8);
		d_disp.download(disp);
		//filterSpeckles(disp, 0, SpeckleWindowSize_GPU, SpeckleRange_GPU);
	      }
	      else
	      {
		  //gpu::divide(d_disp_small,16,d_disp_16,1);
		  d_disp.download(disp);
		  //filterSpeckles(disp, 0, SpeckleWindowSize_GPU, SpeckleRange_GPU);
		  //disp.convertTo(disp8, CV_8U);
		  cv::normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);
		  double minVal, maxVal;
		  minMaxLoc( disp, &minVal, &maxVal );
		  cout<<"min: "<<minVal<<" max: "<<maxVal<<endl;
		  
		
		  // d_disp_16.download(disp8);
		  imshow("gpu_bm", disp8);
		
	      }
	      
	    }//while
      
      

	    //gpu::divide(d_disp,16,d_disp_16,1);
	    
	    
	    imwrite("disp_gpu_bm.bmp", disp8);
	    
	  disp.convertTo(disp, CV_16S);
	  
	  break;
      }//case 2

      case 3: //cpu bm
      {
        
	    int NumDisparities=11, 
		BlockSize=17, 
		MinDisparity=0, 
		SpeckleWindowSize=0,
		SpeckleRange=0, 
		Disp12MaxDiff=0, 
		PreFilterCap=1, 
		UniquenessRatio=0;
 	
	    namedWindow("CPU BM", 1);

	    createTrackbar( "numDisparities *16", "CPU BM", &NumDisparities, 64, 0 );
	    createTrackbar( "minDisparity", "CPU BM", &MinDisparity, 500, 0 );
	    createTrackbar( "BlockSize ", "CPU BM", &BlockSize, 96, 0 );
	    createTrackbar( "speckleWindowSize", "CPU BM", &SpeckleWindowSize, 200, 0 );
	    createTrackbar( "speckleRange", "CPU BM", &SpeckleRange, 10, 0 );
	    createTrackbar( "disp12MaxDiff", "CPU BM", &Disp12MaxDiff, 100, 0 );
	    createTrackbar( "preFilterCap", "CPU BM", &PreFilterCap, 1000, 0 );
	    createTrackbar( "uniquenessRatio", "CPU BM", &UniquenessRatio, 15, 0 );
	    
	    while(1)
	    {
	      Ptr<StereoBM> cpubm;
	      cpubm = createStereoBM(0,21);

	      if(BlockSize%2==0) //block size has to be odd
		BlockSize++;
	      
	      cpubm->setNumDisparities(NumDisparities*16);
	      cpubm->setMinDisparity(MinDisparity);
	      cpubm->setBlockSize(BlockSize);
	      cpubm->setSpeckleWindowSize(SpeckleWindowSize);
	      cpubm->setSpeckleRange(SpeckleRange);
	      cpubm->setDisp12MaxDiff(Disp12MaxDiff);
	      //cpubm->setPreFilterType();
	      cpubm->setPreFilterCap(PreFilterCap);
	      //cpubm->setTextureThreshold();
	      cpubm->setUniquenessRatio(UniquenessRatio);
	      
	    
cout<<endl<<endl<<"ittahiba"<<endl<<endl;		      
	      cpubm->compute(limagerect, rimagerect, disp);
	      	         
	      char key=waitKey(0);
	      if(key == 27)
	      break;
	      
	      //resize(disp, disp_small, Size(), 0.2, 0.2,INTER_LINEAR);
	      double minVal, maxVal;
	      
	      //disp.convertTo( disp8, CV_8UC1, 255/(maxVal- minVal));
	      //disp.convertTo(disp8, CV_8U); 
	      cv::normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);
	      //disp_small.convertTo(disp8, CV_8U, 1/8.); 
	      minMaxLoc( disp, &minVal, &maxVal );
	      cout<<"min: "<<minVal<<" max: "<<maxVal<<endl;
	       
	      //equalizeHist(disp8,disp_eq);
	      imshow("cpu BM", disp8);
	      //imshow("BM", disp8);
	      //imwrite("disp_bm_eq.bmp",disp_eq);  
	      //gpu::resize(d_disp, d_disp_small, Size(), 0.2, 0.2,INTER_LINEAR);
	      //gpu::divide(d_disp_small,16,d_disp_16,1);
	      //d_disp.download(disp8);
	      // d_disp_16.download(disp8);
	      
	      
	    }//while
      
	  imwrite("disp_bm.bmp",disp8); 
      
	  break;
      }//case 3
      
      default:
	cout<<"Not a valid method!"<<endl;
	break;
    }//switch method
  
  
  



      //crop with mask
      cv::Mat dispfilter; 
      disp8.copyTo(dispfilter, roirect);   
      imshow("filtered disparity", dispfilter);
      imwrite("disp_filtered.bmp",dispfilter); 
      imshow("mask", roirect);
      disp.copyTo(dispfilter, roirect);  
      dispfilter.copyTo(disp);



 



   
    
//     for(i=0; i<imageSize.width; i++)
//     {
// 	 for(j=0; j<imageSize.height; j++)  
// 	 {
// 	   if(disp.at<short>(i,j)==-16)
// 	   {  
// 	     
// 	     disp.at<short>(i,j) = 0;			//-16 means invalid pixel in disparity (not sure though)
// 	   //  disp8.at<CV_8U>(i,j)=0;
// 	   }
// 	   
// 	 }
//     }
    
  
    Mat _3dimage(disp.size(), CV_32FC3);	//float type 
 
    
    //write disp to file to check type
FileStorage fs("disp.yaml", FileStorage::WRITE);
fs<<"disp"<<disp;
fs.release();   
    
 
// cout<<disp;
// waitKey();
    filterDisparity(disp);
//TODO    bilateralFilter(disp, dispfilter, 5, 20, 20);
    
//    medianBlur(disp, dispfilter, 5);	//not that good
//    dispfilter.copyTo(disp);
// cout<<disp;
// waitKey();
    
  
    
    
    switch(reprojection)
    {
      
      
      case 1:
      {
      
    Mat Q_32F;
    gpu::GpuMat gpuQ, gpu_3dimage;
      
    Q.convertTo(Q_32F,CV_32F);
    gpuQ.upload(Q_32F);
      
    gpu::GpuMat gpudisp;
    gpudisp.upload(disp);
      
    gpu::reprojectImageTo3D(gpudisp,gpu_3dimage, Q_32F, 3);

    gpu_3dimage.download(_3dimage);
/*    
    	   if(disp.at<short>(i,j)!=-16 &&
		!( 		 
		  isinf(_3dimage.at<Vec3f>(i,j)[0]) ||						//TODO  only needed if using opencv reprojection function should move to the reprojection and set to 0
		  isinf(_3dimage.at<Vec3f>(i,j)[1]) ||
		  isinf(_3dimage.at<Vec3f>(i,j)[2])
		 )
	      )
	   {
      */
    break;
      }//case 1
    
      case 2:
      {
	reprojectImageTo3D(disp, _3dimage, Q);
	break;
      }//case 2
      
      case 3:
      {
	
	
cout<<"reprojecting..."<<endl;
	reprojectTo3d(Q, disp, _3dimage);
cout<<"reprojection done!"<<endl;
//cout<<_3dimage;
	
	break;
      }//case 3

     default:
	cout<<"Not a valid reprojection!"<<endl;
	break;
    
    }//switch reprojection
    

    
//TODO TODO TODO TODO
/*
 * + csinalni parameterezheto ROI-t
 * 
 * regi kepek + 3k kepek block matchingel osszeallitani a kepet, depth mapet, maszkot, pointcloudot osszehasonlitasra.
 * 
 * nagytakaritas
 */
    
    
cout<<"matrix:"<<endl;
//cout<<_3dimage;    
cout<<"matrix end"<<endl;
    

     writeVRML(_3dimage);
      
      
imwrite("1roi.bmp",roirect);
imwrite("1disp.bmp",disp8);
//      imshow("roi", roirect);
      

      
      
      
      
      cout<<endl<<"Type: "<<disp.type()<< endl;
      
      
      
      
      

      
    waitKey(0);





}//rectify




