#include <vector>
#include <tr1/unordered_map>
#include <iomanip>
#include <iostream>

#include <boost/assign/list_of.hpp>
#include <boost/program_options.hpp>

#include <glog/logging.h>

#include "base2d/feature.h"
#include "base3d/bundle_adjustment.h"
#include "base3d/projection.h"
#include "fm/feature_management.h"
#include "sfm/sequential_mapper.h"
#include "util/io.h"
#include "util/timer.h"
#include "densemap/rectify.h"
#include "base2d/image.h"


#define LOCAL_BA_POSE_FREE       0
#define LOCAL_BA_POSE_FIXED      1
#define LOCAL_BA_POSE_FIXED_X    2

using namespace std;

  

namespace config = boost::program_options;

const double FINAL_COST_THRESHOLD = 1;
const double LOSS_SCALE_FACTOR = 1;
const double MIN_INLIER_STOP = 0.5;
const double MIN_INLIER_THRESHOLD = 20;
const double MAX_REPROJ_ERROR = 2;
const double MIN_INIT_DISPARITY = 100;
const double MIN_DISPARITY = 50;
const std::vector<int> LOCAL_BA_POSE_STATE
= boost::assign::list_of ( LOCAL_BA_POSE_FIXED )
  ( LOCAL_BA_POSE_FIXED )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE )
  ( LOCAL_BA_POSE_FREE );

 
void print_report_heading ( const size_t image_idx,
                            const std::vector<Image>& image_data )
{
    std::cout << std::endl;
    std::cout << std::string ( 80, '=' ) << std::endl;
    std::cout << "Processing image #" << image_idx + 1
              << " (" << image_data[image_idx].timestamp << ")" << std::endl;
    std::cout << std::string ( 80, '=' ) << std::endl << std::endl;
}


void print_report_summary ( const size_t image_idx,
                            SequentialMapper& mapper )
{

    const size_t image_id = mapper.get_image_id ( image_idx );

    Eigen::Matrix4d matrix = Eigen::MatrixXd::Identity ( 4, 4 );
    matrix.block<3, 4> ( 0, 0 )
    = compose_Rt_matrix ( mapper.feature_manager.rvecs[image_id],
                          mapper.feature_manager.tvecs[image_id] );
    matrix = matrix.inverse().eval();

    const Eigen::Vector3d tvec = matrix.block<3, 1> ( 0, 3 );

    std::cout << "Global position" << std::endl;
    std::cout << "---------------" << std::endl;
    std::cout << std::setw ( 15 )
              << tvec ( 0 ) << std::endl
              << std::setw ( 15 )
              << tvec ( 1 ) << std::endl
              << std::setw ( 15 )
              << tvec ( 2 ) << std::endl
              << std::endl;

}


void adjust_local_bundle ( SequentialMapper& mapper,
                           const std::list<size_t>& image_idxs )
{

    if ( image_idxs.size() > LOCAL_BA_POSE_STATE.size() )
    {
        throw std::range_error ( "Number of local images in `image_idxs` must not "
                                 "be greater than number of poses defined in "
                                 "`LOCAL_BA_POSE_STATE`." );
    }

    std::vector<size_t> free_image_idxs;
    std::vector<size_t> fixed_image_idxs;
    std::vector<size_t> fixed_x_image_idxs;

    // Set parameters of image poses as constant or variable
    size_t i=0;
    for ( std::list<size_t>::const_iterator it=image_idxs.begin();
            it!=image_idxs.end(); it++, i++ )
    {
        switch ( LOCAL_BA_POSE_STATE[i] )
        {
        case LOCAL_BA_POSE_FREE:
            free_image_idxs.push_back ( *it );
            break;
        case LOCAL_BA_POSE_FIXED:
            fixed_image_idxs.push_back ( *it );
            break;
        case LOCAL_BA_POSE_FIXED_X:
            fixed_x_image_idxs.push_back ( *it );
            break;
        }
    }

    mapper.adjust_bundle ( free_image_idxs, fixed_image_idxs, fixed_x_image_idxs,
                           LOSS_SCALE_FACTOR,
                           false, // update_cov
                           true,  // print_summary
                           false  // print_progress
                         );

}


void adjust_global_bundle ( SequentialMapper& mapper,
                            const size_t start_image_idx,
                            const size_t end_image_idx )
{

    Timer timer;
    timer.start();

    // Globally adjust all poses with fixed initial two poses

    std::vector<size_t> free_image_idxs;
    std::vector<size_t> fixed_image_idxs;
    std::vector<size_t> fixed_x_image_idxs;

    // adjust first two poses as fixed and with fixed x-coordinate
    fixed_image_idxs.push_back ( mapper.get_first_image_idx() );
    fixed_x_image_idxs.push_back ( mapper.get_second_image_idx() );

    // All other poses as free
    for ( size_t image_idx=start_image_idx;
            image_idx<=end_image_idx; image_idx++ )
        if ( mapper.is_image_processed ( image_idx )
                && image_idx != mapper.get_first_image_idx()
                && image_idx != mapper.get_second_image_idx() )
            free_image_idxs.push_back ( image_idx );

    std::cout << std::endl;
    std::cout << std::string ( 80, '=' ) << std::endl;
    std::cout << "Global bundle adjustment" << std::endl;
    std::cout << std::string ( 80, '=' ) << std::endl << std::endl;

    mapper.adjust_bundle ( free_image_idxs, fixed_image_idxs, fixed_x_image_idxs,
                           LOSS_SCALE_FACTOR,
                           true,  // update_cov
                           true,  // print_summary
                           true   // print_progress
                         );

    timer.stop();
    timer.print();

}


void process_remaining_images ( SequentialMapper& mapper,
                                const size_t start_image_idx,
                                const size_t end_image_idx )
{

    Timer timer;

    for ( size_t image_idx=start_image_idx+1;
            image_idx<end_image_idx; image_idx++ )
    {

        if ( !mapper.is_image_processed ( image_idx ) )
        {

            // Find nearest processed images, whose pose is already estimated
            // (previous and next in image chain)
            size_t prev_proc_image_idx = image_idx - 1;
            size_t next_proc_image_idx;
            for ( next_proc_image_idx=image_idx+1;
                    next_proc_image_idx<end_image_idx;
                    next_proc_image_idx++ )
            {
                if ( mapper.is_image_processed ( next_proc_image_idx ) )
                {
                    break;
                }
            }
            if ( next_proc_image_idx == end_image_idx )
            {
                next_proc_image_idx = UINT_MAX;
            }

            // Process skipped images and use previous or next nearest processed
            // images as matching "partner"
            for ( ; image_idx<next_proc_image_idx && image_idx<end_image_idx;
                    image_idx++ )
            {
                size_t prev_dist = image_idx - prev_proc_image_idx;
                size_t next_dist = next_proc_image_idx - image_idx;
                size_t partner_image_idx;
                if ( prev_dist < next_dist )
                {
                    partner_image_idx = prev_proc_image_idx;
                }
                else
                {
                    partner_image_idx = next_proc_image_idx;
                }
                timer.restart();
                print_report_heading ( image_idx, mapper.image_data );
                mapper.process ( image_idx, partner_image_idx, FINAL_COST_THRESHOLD,
                                 LOSS_SCALE_FACTOR, MIN_INLIER_STOP,
                                 MIN_INLIER_THRESHOLD, MAX_REPROJ_ERROR, 0, 3 );
                timer.stop();
                timer.print();
            }

        }

    }
}


int main ( int argc, char* argv[] )
{
  
  cout<<"Opencv ver.: "<<CV_MAJOR_VERSION<<"."<< CV_MINOR_VERSION<<endl;
  
     // Program options
    std::string input_path;
    std::string output_path;
    std::string cache_path;
    std::string image_prefix;
    std::string image_suffix;
    std::string image_ext;
    size_t start_image_idx, end_image_idx;
    bool debug;
    int debug_delay;
    std::string debug_path;

    try
    {
        config::options_description desc ( "Allowed options" );
        desc.add_options()
        ( "help,h",
          "produce help message" )
        ( "input-path,i",
          config::value<std::string> ( &input_path )->required(),
          "path to imagedata.txt, calibrationdata.txt and image files" )
        ( "output-path,o",
          config::value<std::string> ( &output_path )->required(),
          "path to output files" )
        ( "cache-path,o",
          config::value<std::string> ( &cache_path )->required(),
          "path to cache files" )
        ( "image-prefix",
          config::value<std::string> ( &image_prefix )->default_value ( "" ),
          "prefix of image file names before timestamp" )
        ( "image-suffix",
          config::value<std::string> ( &image_suffix )->default_value ( "" ),
          "suffix of image file names after timestamp" )
        ( "image-ext",
          config::value<std::string> ( &image_ext )->default_value ( ".bmp" ),
          "image file name extension" )
        ( "start-image-idx",
          config::value<size_t> ( &start_image_idx )->default_value ( 0 ),
          "index of first image to be processed (position in image data file)" )
        ( "end-image-idx",
          config::value<size_t> ( &end_image_idx )->default_value ( UINT_MAX ),
          "index of last image to be processed (position in image data file)" )
        ( "debug",
          config::bool_switch ( &debug )->default_value ( false ),
          "change to debug mode" )
        ( "debug-path",
          config::value<std::string> ( &debug_path )->default_value ( "" ),
          "path to debug output files" )
        ( "debug-delay",
          config::value<int> ( &debug_delay )->default_value ( 0 ),
          "delay in debug mode for visualizing feature matches (0 for wait key,"
          "otherwise in [ms])" );
        config::variables_map variables_map;
        config::store ( config::parse_command_line ( argc, argv, desc ),
                        variables_map );
        variables_map.notify();

        if ( variables_map.count ( "help" ) )
        {
            std::cout << desc << std::endl;
            return 1;
        }

        // Add trailing slash from input_path
        if ( output_path.at ( output_path.length()-1 ) != '/' )
        {
            output_path += "/";
        }

    }
    catch ( std::exception& e )
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch ( ... )
    {
        std::cerr << "Unknown error!" << "\n";
        return 1;
    }

    // Read input data
   
    std::vector<Image> image_data
    = read_image_data ( input_path + "imagedata.txt", input_path,
                        image_prefix, image_suffix, image_ext );
    Eigen::Matrix3d calib_matrix
    = read_calib_matrix ( input_path + "calibrationdata.txt" );
    
    //Read Poses							!!!!!!!!nezd meg meg nem az outputot olvassa be
    std::vector<Image> camera_poses
    = read_image_data ( output_path + "imagedataout.txt", input_path,
                        image_prefix, image_suffix, image_ext );
    
  
    rectify_images(calib_matrix ,camera_poses, 33, 34); //22,23  //29,30
    
    

    
    
    /*
    SequentialMapper mapper ( image_data, calib_matrix,
                              cache_path, debug, debug_path, debug_delay );
    
    

    // Last image to be processed
    end_image_idx = std::min ( end_image_idx, image_data.size() - 1 );

    // Dynamically adjusted minimum track length for 2D-3D pose estimation
    size_t min_track_len = 2;

    Timer timer;

    size_t image_idx = -1;
    size_t prev_image_idx = -1;
    size_t prev_prev_image_idx = -1;
    size_t prev_prev_prev_image_idx = -1;

    // Image list for local bundle adjustment
    std::list<size_t> ba_image_idxs;

    for ( image_idx=start_image_idx; image_idx<end_image_idx; image_idx++ )
    {

        print_report_heading ( image_idx, image_data );

        timer.restart();

        if ( mapper.get_num_proc_images() == 0 )  // Initial processing
        {

            // Search for good initial pair with sufficiently large disparity
            while ( true )
            {
                image_idx++;
                const double disparity
                = calculate_image_disparity ( image_data[start_image_idx],
                                              image_data[image_idx] );
                if ( disparity > MIN_INIT_DISPARITY )
                {
                    break;
                }
            }

            mapper.process_initial ( start_image_idx, image_idx );

            print_report_summary ( start_image_idx, mapper );

            print_report_heading ( image_idx, image_data );
            print_report_summary ( image_idx, mapper );

            // Adjust first two poses
            std::vector<size_t> free_image_idxs;
            std::vector<size_t> fixed_image_idxs;
            std::vector<size_t> fixed_x_image_idxs;
            fixed_image_idxs.push_back ( start_image_idx );
            fixed_x_image_idxs.push_back ( image_idx );
            mapper.adjust_bundle ( free_image_idxs,
                                   fixed_image_idxs,
                                   fixed_x_image_idxs );

            // Add to local bundle adjustment
            ba_image_idxs.push_back ( mapper.get_first_image_idx() );
            ba_image_idxs.push_back ( mapper.get_second_image_idx() );

        }
        else if ( mapper.get_num_proc_images() >= 2 )
        {

            // Increase minimum track length for 2D-3D pose estimation
            if ( mapper.get_num_proc_images() > 4 )
            {
                min_track_len = 3;
            }
            else if ( mapper.get_num_proc_images() > 6 )
            {
                min_track_len = 4;
            }

            // try to process image
            bool success;
            while ( ! ( success = mapper.process ( image_idx, prev_image_idx,
                                                   FINAL_COST_THRESHOLD,
                                                   LOSS_SCALE_FACTOR,
                                                   MIN_INLIER_STOP,
                                                   MIN_INLIER_THRESHOLD,
                                                   MAX_REPROJ_ERROR,
                                                   MIN_DISPARITY,
                                                   min_track_len ) ) )
            {
                timer.stop();
                timer.print();
                timer.restart();
                image_idx++;
                if ( image_idx >= end_image_idx )
                {
                    break;
                }
                print_report_heading ( image_idx, image_data );
            }
            if ( !success )
            {
                break;
            }

            // if (prev_prev_image_idx != -1) {
            //   mapper.process(image_idx, prev_prev_image_idx, FINAL_COST_THRESHOLD,
            //                  LOSS_SCALE_FACTOR, MIN_INLIER_STOP,
            //                  MIN_INLIER_THRESHOLD, MAX_REPROJ_ERROR,
            //                  0, min_track_len);
            // }
            // if (prev_prev_prev_image_idx != -1) {
            //   mapper.process(image_idx, prev_prev_prev_image_idx,
            //                  FINAL_COST_THRESHOLD, LOSS_SCALE_FACTOR,
            //                  MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
            //                  MAX_REPROJ_ERROR, 0, min_track_len);
            // }

            // Adjust local bundle

            if ( ba_image_idxs.size() == LOCAL_BA_POSE_STATE.size() )
            {
                ba_image_idxs.pop_front();
            }
            ba_image_idxs.push_back ( image_idx );

            adjust_local_bundle ( mapper, ba_image_idxs );

        }

        print_report_summary ( image_idx, mapper );

        timer.stop();
        timer.print();

        prev_prev_prev_image_idx = prev_prev_image_idx;
        prev_prev_image_idx = prev_image_idx;
        prev_image_idx = image_idx;

    }

    // Process all skipped frames
    process_remaining_images ( mapper, start_image_idx, end_image_idx );

    // for (size_t i=start_image_idx; i<end_image_idx; i++) {
    //   for (size_t j=start_image_idx; j<end_image_idx; j++) {
    //     if (i != j) {
    //       mapper.process(i, j, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    //     }
    //   }
    // }

    // mapper.debug = true;
    // mapper.process(33, 31, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(29, 31, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(33, 30, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(32, 30, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(33, 29, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(31, 29, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(32, 29, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(1150, start_image_idx, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(1149, start_image_idx, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(1148, start_image_idx, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(1147, start_image_idx, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.process(1146, start_image_idx, FINAL_COST_THRESHOLD,
    //                LOSS_SCALE_FACTOR, MIN_INLIER_STOP, MIN_INLIER_THRESHOLD,
    //                MAX_REPROJ_ERROR, 0, min_track_len);
    // mapper.debug = false;

    // Global Bundle Adjustment
    adjust_global_bundle ( mapper, start_image_idx, end_image_idx );


    // Write output data

    write_image_data ( output_path + "imagedataout.txt", mapper );
    write_camera_poses ( output_path + "cameras.wrl", mapper.feature_manager,
                         0.05, 1, 0, 0 );
    write_point_cloud ( output_path + "point_cloud.wrl", mapper, 10, 1, 500 );

    // size_t N = feature_manager.points3D.size() / 2;
    // for (size_t i=N-100; i<N; i++) {
    //   write_track(output_path + "tracks", mapper, i);
    // }
*/
    return 0;
}
