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
/*
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
  ( LOCAL_BA_POSE_FREE );*/

/* 
void print_report_heading ( const size_t image_idx,
                            const std::vector<Image>& image_data )
{
    std::cout << std::endl;
    std::cout << std::string ( 80, '=' ) << std::endl;
    std::cout << "Processing image #" << image_idx + 1
              << " (" << image_data[image_idx].timestamp << ")" << std::endl;
    std::cout << std::string ( 80, '=' ) << std::endl << std::endl;
}
*/

// void print_report_summary ( const size_t image_idx,
//                             SequentialMapper& mapper )
// {
// 
//     const size_t image_id = mapper.get_image_id ( image_idx );
// 
//     Eigen::Matrix4d matrix = Eigen::MatrixXd::Identity ( 4, 4 );
//     matrix.block<3, 4> ( 0, 0 )
//     = compose_proj_matrix ( mapper.feature_manager.rvecs[image_id],
//                           mapper.feature_manager.tvecs[image_id] );
//     matrix = matrix.inverse().eval();
// 
//     const Eigen::Vector3d tvec = matrix.block<3, 1> ( 0, 3 );
// 
//     std::cout << "Global position" << std::endl;
//     std::cout << "---------------" << std::endl;
//     std::cout << std::setw ( 15 )
//               << tvec ( 0 ) << std::endl
//               << std::setw ( 15 )
//               << tvec ( 1 ) << std::endl
//               << std::setw ( 15 )
//               << tvec ( 2 ) << std::endl
//               << std::endl;
// 
// }

/*
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
*/

int main ( int argc, char* argv[] )
{

  google::InitGoogleLogging(argv[0]);

  // Program options
  std::string input_path;
  std::string output_path;
  std::string cache_path;
  std::string voc_tree_path;
  std::string image_prefix;
  std::string image_suffix;
  std::string image_ext;
  size_t start_image_idx, end_image_idx;
  int first_image_idx, second_image_idx;
  bool debug;
  std::string debug_path;

  // Feature detection and extraction options
  SURFOptions surf_options;

  // Sequential mapper options
  SequentialMapperOptions init_mapper_options;
  SequentialMapperOptions mapper_options;

  // Bundle adjustment options
  BundleAdjustmentOptions ba_options;
  BundleAdjustmentOptions ba_local_options;
  BundleAdjustmentOptions ba_global_options;

  // Processing options
  bool process_curr_prev_prev;

  // Failure options
  size_t max_subsequent_trials;
  size_t failure_max_image_dist;
  size_t failure_skip_images;

  // Loop detection options
  bool loop_detection;
  size_t loop_detection_num_images;
  size_t loop_detection_num_nh_images;
  size_t loop_detection_nh_dist;
  size_t loop_detection_period;

  // Bundle adjustment options
  size_t local_ba_window_size;

  bool merge;
  size_t merge_num_skip_images;

  bool use_control_points;
  std::string control_point_data_path;

  double filter_max_error;

  config::variables_map vmap;

  try {
    config::options_description options_description("Options");
    options_description.add_options()
      ("help,h",
       "Print this help message.")

      // Path options
      ("input-path",
       config::value<std::string>(&input_path)
         ->required(),
       "Path to imagedata.txt and image files.")
      ("output-path",
       config::value<std::string>(&output_path)
         ->required(),
       "Path to output files.")
      ("cache-path",
       config::value<std::string>(&cache_path)
         ->required(),
       "Path to cache files.")
      ("voc-tree-path",
       config::value<std::string>(&voc_tree_path)
         ->required(),
       "Path to vocabulary tree.")

      // Image filename options
      ("image-prefix",
       config::value<std::string>(&image_prefix)
         ->default_value(""),
       "Prefix of image file names before name.")
      ("image-suffix",
       config::value<std::string>(&image_suffix)
         ->default_value(""),
       "Suffix of image file names after name.")
      ("image-ext",
       config::value<std::string>(&image_ext)
         ->default_value(".bmp"),
       "Image file name extension.")

      // Start and end image index options
      ("start-image-idx",
       config::value<size_t>(&start_image_idx)
         ->default_value(0),
       "Index of first image to be processed (position in image data file).")
      ("end-image-idx",
       config::value<size_t>(&end_image_idx)
         ->default_value(UINT_MAX),
       "Index of last image to be processed (position in image data file).")
      ("first-image-idx",
       config::value<int>(&first_image_idx)
         ->default_value(-1, "auto"),
       "Index of first image index of the initial pair. This is useful if the "
       "automatically chosen initial pair yields bad reconstruction results. "
       "Default is -1 and determines the image automatically.")
      ("second-image-idx",
       config::value<int>(&second_image_idx)
         ->default_value(-1, "auto"),
       "Index of second image index of the initial pair. This is useful if "
       "the automatically chosen initial pair yields bad reconstruction "
       "results. Default is -1 and determines the image automatically.")

      // Debug options
      ("debug",
       config::value<bool>(&debug)
         ->default_value(false, "false"),
       "Enable debug mode.")
      ("debug-path",
       config::value<std::string>(&debug_path)
         ->default_value(""),
       "Path to debug output files.")

      // Feature detection and extraction options
      ("surf-hessian-threshold",
       config::value<double>(&surf_options.hessian_threshold)
         ->default_value(1000),
       "Hessian threshold for SURF feature detection.")
      ("surf-num-octaves",
       config::value<size_t>(&surf_options.num_octaves)
         ->default_value(4),
       "The number of a gaussian pyramid octaves for the SURF detector.")
      ("surf-num-octave-layers",
       config::value<size_t>(&surf_options.num_octave_layers)
         ->default_value(3),
       "The number of images within each octave of a gaussian pyramid "
       "for the SURF detector.")
      ("surf-adaptive",
       config::value<bool>(&surf_options.adaptive)
         ->default_value(true, "true"),
       "Whether to use adaptive gridded SURF feature detection, which splits "
       "the image in grid of sub-images and detects features for each grid "
       "cell separately in order to ensure an evenly distributed number of "
       "features in each region of the image.")
      ("surf-adaptive-min-per-cell",
       config::value<size_t>(&surf_options.adaptive_min_per_cell)
         ->default_value(100),
       "Minimum number of features per grid cell.")
      ("surf-adaptive-max-per-cell",
       config::value<size_t>(&surf_options.adaptive_max_per_cell)
         ->default_value(300),
       "Maximum number of features per grid cell.")
      ("surf-adaptive-cell-rows",
       config::value<size_t>(&surf_options.adaptive_cell_rows)
         ->default_value(3),
       "Number of grid cells in the first dimension of the image. The total "
       "number of grid cells is defined as `surf-cell-rows` x "
       "`surf-cell-cols`.")
      ("surf-adaptive-cell-cols",
       config::value<size_t>(&surf_options.adaptive_cell_cols)
         ->default_value(3),
       "Number of grid cells in the first dimension of the image. The total "
       "number of grid cells is defined as `surf-cell-rows` x "
       "`surf-cell-cols`.")

      // Sequential mapper options
      //    General

      ("init-max-homography-inliers",
       config::value<double>(&init_mapper_options.max_homography_inliers)
         ->default_value(0.7, "0.7"),
       "Maximum allow relative number (0-1) of inliers in homography between "
       "two images in order to guarantee sufficient view-point change. "
       "Larger values result in requiring larger view-point changes.")
      ("init-min-disparity",
       config::value<double>(&init_mapper_options.min_disparity)
         ->default_value(0, "0"),
       "Minimum median feature disparity between "
       "two images in order to guarantee sufficient view-point change.")
      ("max-homography-inliers",
       config::value<double>(&mapper_options.max_homography_inliers)
         ->default_value(0.8, "0.8"),
       "Maximum allow relative number (0-1) of inliers in homography between "
       "two images in order to guarantee sufficient view-point change. "
       "Larger values result in requiring larger view-point changes.")
      ("min-disparity",
       config::value<double>(&mapper_options.min_disparity)
         ->default_value(0, "0"),
       "Minimum median feature disparity between "
       "two images in order to guarantee sufficient view-point change.")
      ("match-max-ratio",
       config::value<double>(&mapper_options.match_max_ratio)
         ->default_value(0.9, "0.9"),
       "Maximum distance ratio between first and second best matches.")
      ("match-max-distance",
       config::value<double>(&mapper_options.match_max_distance)
         ->default_value(-1, "-1"),
       "Maximum distance in pixels for valid correspondence.")
      ("min-track-len",
       config::value<size_t>(&mapper_options.min_track_len)
         ->default_value(3),
       "Minimum track length of a 3D point to be used for 2D-3D pose "
       "estimation. This threshold takes effect when the number of "
       "successfully processed images is > `2 * min-track-len`.")
      ("final-cost-threshold",
       config::value<double>(&mapper_options.final_cost_threshold)
         ->default_value(2),
       "Threshold for final cost of pose refinement.")
      ("loss-scale-factor",
       config::value<double>(&ba_options.loss_scale_factor)
         ->default_value(1),
       "Scale factor of robust Cauchy loss function for pose refinement and "
       "bundle adjustment.")
      ("tri-max-reproj-error",
       config::value<double>(&mapper_options.tri_max_reproj_error)
         ->default_value(4),
       "Maximum reprojection error for newly triangulated points to be saved.")
      ("init-tri-min-angle",
       config::value<double>(&init_mapper_options.tri_min_angle)
         ->default_value(10),
       "Minimum (or maximum as 180 - angle) angle between two rays of a newly "
       "triangulated point.")
      ("tri-min-angle",
       config::value<double>(&mapper_options.tri_min_angle)
         ->default_value(1),
       "Minimum (or maximum as 180 - angle) angle between two rays of a newly "
       "triangulated point.")
      ("ransac-min-inlier-stop",
       config::value<double>(&mapper_options.ransac_min_inlier_stop)
           ->default_value(0.6, "0.6"),
       "RANSAC algorithm for 2D-3D pose estimation stops when at least this "
       "number of inliers is found, as relative (<1) w.r.t. total number of "
       "features or absolute (>1) value.")
      ("ransac-max-reproj-error",
       config::value<double>(&mapper_options.ransac_max_reproj_error)
         ->default_value(4),
       "Maximum reprojection used for RANSAC.")
      ("ransac-min-inlier-threshold",
       config::value<double>(&mapper_options.ransac_min_inlier_threshold)
         ->default_value(30),
       "Processing of image pair fails if less than this number of inliers is "
       "found in the RANSAC 2D-3D pose estimation, as relative (<1) w.r.t. "
       "total number of features or absolute (>1) value.")

      // Processing options
      ("process-curr-prev-prev",
       config::value<bool>(&process_curr_prev_prev)
         ->default_value(true, "true"),
       "Whether to subsequently process current image not only against "
       "previous image but also against image before previous image.")

      // Failure options
      ("max-subsequent-trials",
       config::value<size_t>(&max_subsequent_trials)
         ->default_value(30),
       "Maximum number of times to skip subsequent images due to failed "
       "processing.")
      ("failure-max-image-dist",
       config::value<size_t>(&failure_max_image_dist)
         ->default_value(10),
       "If subsequent processing fails (after `max-subsequent-trials`) this "
       "routine tries to find a valid image pair by trying to process all "
       "possible combinations in the range "
       "[`last-image-idx - dist`; `last-image-idx + dist`].")
      ("failure-skip-images",
       config::value<size_t>(&failure_skip_images)
         ->default_value(1),
       "If all trials to find a valid image pair failed, a new mapper is "
       "created. This mapper starts to process images at image index "
       "`last-image-idx + failure-skip-images`.")

      // Loop detection options
      ("loop-detection",
       config::value<bool>(&loop_detection)
         ->default_value(true, "true"),
       "Whether to enable loop detection.")
      ("loop-detection-num-images",
       config::value<size_t>(&loop_detection_num_images)
         ->default_value(30),
       "Maximum number of most similar images to test for loop closure.")
      ("loop-detection-num-nh-images",
       config::value<size_t>(&loop_detection_num_nh_images)
         ->default_value(15),
       "Maximum number of most similar images in direct neighborhood of "
       "current image to test for loop closure.")
      ("loop-detection-nh-dist",
       config::value<size_t>(&loop_detection_nh_dist)
         ->default_value(30),
       "Distance which determines neighborhood of current image as "
       "[`curr-image-idx - dist`; `curr-image-idx + dist`]")
      ("loop-detection-period",
       config::value<size_t>(&loop_detection_period)
         ->default_value(20),
       "Loop detection is initiated every `loop-period` successfully "
       "processed images.")

      // Bundle adjustment options
      ("local-ba-window-size",
       config::value<size_t>(&local_ba_window_size)
         ->default_value(8),
       "Window size of the local bundle adjustment. The last N poses are "
       "adjusted after each successful pose reconstruction.")
      ("constrain-rotation",
       config::value<bool>(&ba_global_options.constrain_rotation)
         ->default_value(false, "false"),
       "Whether to constrain poses against given poses in `imagedata.txt` "
       "in the global bundle adjustment.")
      ("constrain-rotation-weight",
       config::value<double>(&ba_global_options.constrain_rotation_weight)
         ->default_value(0),
       "Weight for constraint residual of rotation.")
      ("refine-camera-params",
       config::value<bool>(&ba_global_options.refine_camera_params)
         ->default_value(true, "true"),
       "Whether to refine the camera parameters in global bundle adjustment.")
      ("local-ba-refine-camera-params",
       config::value<bool>(&ba_local_options.refine_camera_params)
         ->default_value(true, "true"),
       "Whether to refine the camera parameters in the local bundle "
       "adjustment.")


      // Merge routine options
      ("merge",
       config::value<bool>(&merge)
         ->default_value(true, "true"),
       "Whether to try to merge separate mappers.")
      ("merge-num-skip-images",
       config::value<size_t>(&merge_num_skip_images)->default_value(5),
       "Merge routine searches every `merge-num-skip-images` frames for "
       "a common image in the separate mappers.")

      // Control point options
      ("use-control-points",
       config::value<bool>(&use_control_points)
         ->default_value(false, "false"),
       "Whether to estimate GCP coordinates and to transform model to GCP system.")
      ("control-point-data-path",
       config::value<std::string>(&control_point_data_path)->default_value(""),
       "Path to GCP data.")

      // Filter options
      ("filter-max-error",
       config::value<double>(&filter_max_error)
         ->default_value(-1),
       "If >0 re-run bundle adjustment with 3D points filtered whose mean residual is larger "
       "than this threshold.");

    // Use the same loss-scale factor for global BA
    ba_local_options.loss_scale_factor = ba_options.loss_scale_factor;
    ba_global_options.loss_scale_factor = ba_options.loss_scale_factor;

    config::store(config::parse_command_line(argc, argv, options_description),
                  vmap);

    if (vmap.count("help")) {
      std::cout << options_description << std::endl;
      return 1;
    }

    vmap.notify();

  }
  catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << "Unknown error!" << std::endl;
    return 1;
  }

  if ((first_image_idx == -1) ^ (second_image_idx == -1)) {
    throw std::invalid_argument("You must specify both `first-image-idx` and "
                                "`second-image-idx`.");
  }

  input_path = ensure_trailing_slash(input_path);
  output_path = ensure_trailing_slash(output_path);


  // Check paths

  if (!boost::filesystem::exists(input_path + "imagedata.txt")) {
    std::cerr << "`imagedata.txt` does not exist." << std::endl;
    return 1;
  }
  if (!boost::filesystem::exists(output_path)) {
    std::cerr << "`output-path` does not exist." << std::endl;
    return 1;
  }
  if (!boost::filesystem::exists(cache_path)) {
    std::cerr << "`cache-path` does not exist." << std::endl;
    return 1;
  }
  if (!boost::filesystem::exists(debug_path)) {
    std::cerr << "`debug-path` does not exist." << std::endl;
    return 1;
  }
  if (use_control_points && !boost::filesystem::exists(control_point_data_path)) {
    std::cerr << "`control-point-data-path` does not exist." << std::endl;
    return 1;
  }


  // Read input data

  std::vector<Image> image_data
    = read_image_data(input_path + "imagedata.txt", input_path,
                      image_prefix, image_suffix, image_ext);

  std::vector<SequentialMapper*> mappers;
  SequentialMapper* mapper
    = new SequentialMapper(image_data,
                           cache_path,
                           voc_tree_path,
                           surf_options,
                           loop_detection,
                           debug,
                           debug_path);
  mappers.push_back(mapper);


   
//     std::vector<Image> image_data
//     = read_image_data ( input_path + "imagedata.txt", input_path,
//                         image_prefix, image_suffix, image_ext );
     Eigen::Matrix3d calib_matrix
     = read_calib_matrix ( input_path + "calibrationdata.txt" );
//     
     //Read Poses							!!!!!!!!nezd meg meg nem az outputot olvassa be
     std::vector<Image> camera_poses
     = read_image_data ( output_path + "imagedataout.txt", input_path,
                         image_prefix, image_suffix, image_ext );
//     
//   
     rectify_images(calib_matrix ,camera_poses, 33, 34); //22,23  //29,30
//     
    

    
    
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
