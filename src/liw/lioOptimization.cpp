#include "liw/lioOptimization.h"

#include <cnpy.h>
#include <algorithm>
#include "gs/loss_utils.cuh"

#define DEBUG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "output/" + name))

cloudFrame::cloudFrame(std::vector<point3D>& point_frame_, state* p_state_) {
  point_frame.insert(point_frame.end(), point_frame_.begin(), point_frame_.end());

  p_state = p_state_;
}

cloudFrame::cloudFrame(cloudFrame* p_cloud_frame) {
  time_sweep_begin = p_cloud_frame->time_sweep_begin;
  time_sweep_end = p_cloud_frame->time_sweep_end;
  time_frame_begin = p_cloud_frame->time_frame_begin;
  time_frame_end = p_cloud_frame->time_frame_end;

  id = p_cloud_frame->id;
  sub_id = p_cloud_frame->sub_id;
  frame_id = p_cloud_frame->frame_id;

  p_state = p_cloud_frame->p_state;

  point_frame.insert(point_frame.end(), p_cloud_frame->point_frame.begin(), p_cloud_frame->point_frame.end());

  offset_begin = p_cloud_frame->offset_begin;
  offset_end = p_cloud_frame->offset_end;
  dt_offset = p_cloud_frame->dt_offset;
}

void cloudFrame::release() {
  std::vector<point3D>().swap(point_frame);
  if (p_state != nullptr) {
    p_state->release();
    delete p_state;
    p_state = nullptr;
  }

  if (!rgb_image.empty()) {
    rgb_image.release();
  }

  if (!gray_image.empty()) {
    gray_image.release();
  }
}

bool cloudFrame::if2dPointsAvailable(const double& u, const double& v, const double& scale, double fov_mar) {
  double used_fov_margin = p_state->fov_margin;

  if (fov_mar > 0.0)
    used_fov_margin = fov_mar;

  if ((u / scale >= (used_fov_margin * image_cols + 1)) &&
      (std::ceil(u / scale) < ((1 - used_fov_margin) * image_cols)) &&
      (v / scale >= (used_fov_margin * image_rows + 1)) &&
      (std::ceil(v / scale) < ((1 - used_fov_margin) * image_rows)))
    return true;
  else
    return false;
}

bool cloudFrame::getRgb(const double& u, const double& v, int& r, int& g, int& b) {
  r = rgb_image.at<cv::Vec3b>(v, u)[2];
  g = rgb_image.at<cv::Vec3b>(v, u)[1];
  b = rgb_image.at<cv::Vec3b>(v, u)[0];

  return true;
}

template <typename T>
inline T getSubPixel(cv::Mat& mat, const double& row, const double& col, double pyramid_layer = 0) {
  int floor_row = floor(row);
  int floor_col = floor(col);

  double frac_row = row - floor_row;
  double frac_col = col - floor_col;

  int ceil_row = floor_row + 1;
  int ceil_col = floor_col + 1;

  if (pyramid_layer != 0) {
    int pos_bias = pow(2, pyramid_layer - 1);

    floor_row -= pos_bias;
    floor_col -= pos_bias;
    ceil_row += pos_bias;
    ceil_row += pos_bias;
  }

  return ((1.0 - frac_row) * (1.0 - frac_col) * (T)mat.ptr<T>(floor_row)[floor_col]) +
         (frac_row * (1.0 - frac_col) * (T)mat.ptr<T>(ceil_row)[floor_col]) +
         ((1.0 - frac_row) * frac_col * (T)mat.ptr<T>(floor_row)[ceil_col]) +
         (frac_row * frac_col * (T)mat.ptr<T>(ceil_row)[ceil_col]);
}

Eigen::Vector3d cloudFrame::getRgb(double& u, double& v, int layer, Eigen::Vector3d* rgb_dx, Eigen::Vector3d* rgb_dy) {
  const int ssd = 5;

  cv::Vec3b rgb = getSubPixel<cv::Vec3b>(rgb_image, v, u, layer);

  if (rgb_dx != nullptr) {
    cv::Vec3f rgb_left(0, 0, 0), rgb_right(0, 0, 0);

    float pixel_dif = 0;

    for (int bias_idx = 1; bias_idx < ssd; bias_idx++) {
      rgb_left += getSubPixel<cv::Vec3b>(rgb_image, v, u - bias_idx, layer);
      rgb_right += getSubPixel<cv::Vec3b>(rgb_image, v, u + bias_idx, layer);
      pixel_dif += 2 * bias_idx;
    }

    cv::Vec3f cv_rgb_dx = rgb_right - rgb_left;
    *rgb_dx = Eigen::Vector3d(cv_rgb_dx(0), cv_rgb_dx(1), cv_rgb_dx(2)) / pixel_dif;
  }

  if (rgb_dy != nullptr) {
    cv::Vec3f rgb_down(0, 0, 0), rgb_up(0, 0, 0);

    float pixel_dif = 0;

    for (int bias_idx = 1; bias_idx < ssd; bias_idx++) {
      rgb_down += getSubPixel<cv::Vec3b>(rgb_image, v - bias_idx, u, layer);
      rgb_up += getSubPixel<cv::Vec3b>(rgb_image, v + bias_idx, u, layer);
      pixel_dif += 2 * bias_idx;
    }

    cv::Vec3f cv_rgb_dy = rgb_up - rgb_down;
    *rgb_dy = Eigen::Vector3d(cv_rgb_dy(0), cv_rgb_dy(1), cv_rgb_dy(2)) / pixel_dif;
  }

  return Eigen::Vector3d(rgb(0), rgb(1), rgb(2));
}

bool cloudFrame::project3dTo2d(const pcl::PointXYZI& point_in, double& u, double& v, const double& scale) {
  Eigen::Vector3d point_world(point_in.x, point_in.y, point_in.z);

  Eigen::Vector3d point_camera = p_state->q_camera_world.toRotationMatrix() * point_world + p_state->t_camera_world;

  if (point_camera(2, 0) < 0.001) {
    return false;
  }

  u = (point_camera(0) * p_state->fx / point_camera(2) + p_state->cx) * scale;
  v = (point_camera(1) * p_state->fy / point_camera(2) + p_state->cy) * scale;

  return true;
}

bool cloudFrame::project3dPointInThisImage(
    const pcl::PointXYZI& point_in,
    double& u,
    double& v,
    pcl::PointXYZRGB* rgb_point,
    double intrinsic_scale) {
  if (project3dTo2d(point_in, u, v, intrinsic_scale) == false)
    return false;

  if (if2dPointsAvailable(u, v, intrinsic_scale) == false)
    return false;

  if (rgb_point != nullptr) {
    int r = 0;
    int g = 0;
    int b = 0;
    getRgb(u, v, r, g, b);
    rgb_point->x = point_in.x;
    rgb_point->y = point_in.y;
    rgb_point->z = point_in.z;
    rgb_point->r = r;
    rgb_point->g = g;
    rgb_point->b = b;
    rgb_point->a = 255;
  }

  return true;
}

bool cloudFrame::project3dPointInThisImage(
    const Eigen::Vector3d& point_in,
    double& u,
    double& v,
    pcl::PointXYZRGB* rgb_point,
    double intrinsic_scale) {
  pcl::PointXYZI temp_point;
  temp_point.x = point_in(0);
  temp_point.y = point_in(1);
  temp_point.z = point_in(2);

  return project3dPointInThisImage(temp_point, u, v, rgb_point, intrinsic_scale);
}

void cloudFrame::refreshPoseForProjection() {
  p_state->q_camera_world = p_state->q_world_camera.inverse();
  p_state->t_camera_world = -p_state->q_camera_world.toRotationMatrix() * p_state->t_world_camera;
}

estimationSummary::estimationSummary() {}

void estimationSummary::release() {}

lioOptimization::lioOptimization() {
  allocateMemory();

  readParameters();

  initialValue();

  pub_odom = nh.advertise<nav_msgs::Odometry>("/Odometry", 5);

  if (ENABLE_PUBLISH) {
    pub_cloud_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_current", 2);
    pub_cloud_world = nh.advertise<sensor_msgs::PointCloud2>("/cloud_global_map", 2);
    pub_path = nh.advertise<nav_msgs::Path>("/path", 5);
    pub_cloud_color = nh.advertise<sensor_msgs::PointCloud2>("/color_global_map", 2);
    pub_cloud_test = nh.advertise<sensor_msgs::PointCloud2>("/pub_cloud_test", 2);
    pub_cloud_color_vec.resize(1000);
  }

  if (cloud_pro->getLidarType() == LIVOX)
    sub_cloud_ori = nh.subscribe<livox_ros_driver2::CustomMsg>(lidar_topic, 20, &lioOptimization::livoxHandler, this);
  else
    sub_cloud_ori =
        nh.subscribe<sensor_msgs::PointCloud2>(lidar_topic, 20, &lioOptimization::standardCloudHandler, this);

  sub_imu_ori = nh.subscribe<sensor_msgs::Imu>(imu_topic, 500, &lioOptimization::imuHandler, this);

  if (image_type == RGB8)
    sub_img_ori = nh.subscribe(image_topic, 20, &lioOptimization::imageHandler, this);
  else if (image_type == COMPRESSED)
    sub_img_ori = nh.subscribe(image_topic, 20, &lioOptimization::compressedImageHandler, this);

  check_timer = nh.createTimer(ros::Duration(1000.0), &lioOptimization::heartHandler, this);

  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
  points_world.reset(new pcl::PointCloud<pcl::PointXYZI>());

  gpmap_pro.reset(new GSLIVM::GpMap(gp_options_));
  gpprocess_pro.reset(new gpProcess(gp_options_));
}

void lioOptimization::readParameters() {
  int para_int;
  double para_double;
  float para_float;
  bool para_bool;
  std::string str_temp;

  nh.param<std::string>("bag_path", save_bag_path, "");

  // common
  nh.param<std::string>("common/lidar_topic", lidar_topic, "/points_raw");
  nh.param<std::string>("common/imu_topic", imu_topic, "/imu_raw");
  nh.param<std::string>("common/image_topic", image_topic, "/image_raw");
  nh.param<std::string>("common/image_type", str_temp, "RGB8");
  if (str_temp == "RGB8")
    image_type = RGB8;
  else if (str_temp == "COMPRESSED")
    image_type = COMPRESSED;
  else
    std::cout << "The `image type` " << str_temp << " is not supported." << std::endl;

  nh.param<int>("common/point_filter_num", para_int, 1);
  cloud_pro->setPointFilterNum(para_int);
  nh.param<int>("common/image_filter_num", image_filter_num, 1);
  nh.param<std::vector<double>>("common/gravity_acc", v_G, std::vector<double>());
  nh.param<bool>("debug_output", debug_output, false);
  nh.param<std::string>("output_path", output_path, "");

  // LiDAR parameter
  nh.param<int>("lidar_parameter/lidar_type", para_int, LIVOX);
  cloud_pro->setLidarType(para_int);
  nh.param<int>("lidar_parameter/N_SCANS", para_int, -16);
  cloud_pro->setNumScans(para_int);
  nh.param<int>("lidar_parameter/SCAN_RATE", para_int, -10);
  cloud_pro->setScanRate(para_int);
  nh.param<int>("lidar_parameter/time_unit", para_int, US);
  cloud_pro->setTimeUnit(para_int);
  nh.param<double>("lidar_parameter/blind", para_double, -0.01);
  cloud_pro->setBlind(para_double);
  nh.param<double>("lidar_parameter/det_range", para_double, -100);
  cloud_pro->setDetRange(para_double);

  // IMU parameter
  nh.param<double>("imu_parameter/acc_cov", para_double, -0.1);
  eskf_pro->setAccCov(para_double);
  nh.param<double>("imu_parameter/gyr_cov", para_double, -0.1);
  eskf_pro->setGyrCov(para_double);
  nh.param<double>("imu_parameter/b_acc_cov", para_double, -0.0001);
  eskf_pro->setBiasAccCov(para_double);
  nh.param<double>("imu_parameter/b_gyr_cov", para_double, -0.0001);
  eskf_pro->setBiasGyrCov(para_double);

  nh.param<bool>("imu_parameter/time_diff_enable", time_diff_enable, false);

  // camera parameter
  nh.param<int>("camera_parameter/image_width", para_int, -640);
  img_pro->setImageWidth(para_int);
  image_width_verify = para_int;
  nh.param<int>("camera_parameter/image_height", para_int, -480);
  img_pro->setImageHeight(para_int);
  image_height_verify = para_int;

  nh.param<double>("camera_parameter/image_resize_ratio", para_double, -1.0);
  img_pro->setImageRatio(para_double);
  nh.param<std::vector<double>>("camera_parameter/camera_intrinsic", v_camera_intrinsic, std::vector<double>());
  nh.param<std::vector<double>>("camera_parameter/camera_dist_coeffs", v_camera_dist_coeffs, std::vector<double>());

  // extrinsic parameter
  nh.param<bool>("extrinsic_parameter/extrinsic_enable", extrin_enable, true);
  nh.param<std::vector<double>>(
      "extrinsic_parameter/extrinsic_t_imu_lidar", v_extrin_t_imu_lidar, std::vector<double>());
  nh.param<std::vector<double>>(
      "extrinsic_parameter/extrinsic_R_imu_lidar", v_extrin_R_imu_lidar, std::vector<double>());
  nh.param<std::vector<double>>(
      "extrinsic_parameter/extrinsic_t_imu_camera", v_extrin_t_imu_camera, std::vector<double>());
  nh.param<std::vector<double>>(
      "extrinsic_parameter/extrinsic_R_imu_camera", v_extrin_R_imu_camera, std::vector<double>());

  // state estimation parameters
  nh.param<double>("odometry_options/init_voxel_size", odometry_options.init_voxel_size, -0.2);
  nh.param<double>("odometry_options/init_sample_voxel_size", odometry_options.init_sample_voxel_size, -1.0);
  nh.param<int>("odometry_options/init_num_frames", odometry_options.init_num_frames, -20);
  nh.param<double>("odometry_options/voxel_size", odometry_options.voxel_size, -0.5);
  nh.param<double>("odometry_options/sample_voxel_size", odometry_options.sample_voxel_size, -1.5);
  nh.param<double>("odometry_options/max_distance", odometry_options.max_distance, -100.0);
  nh.param<int>("odometry_options/max_num_points_in_voxel", odometry_options.max_num_points_in_voxel, -20);
  nh.param<double>("odometry_options/min_distance_points", odometry_options.min_distance_points, -0.1);
  nh.param<double>("odometry_options/distance_error_threshold", odometry_options.distance_error_threshold, -5.0);

  nh.param<std::string>("odometry_options/motion_compensation", str_temp, "CONSTANT_VELOCITY");
  if (str_temp == "IMU")
    odometry_options.motion_compensation = IMU;
  else if (str_temp == "CONSTANT_VELOCITY")
    odometry_options.motion_compensation = CONSTANT_VELOCITY;
  else
    std::cout << "The `motion_compensation` " << str_temp << " is not supported." << std::endl;

  nh.param<std::string>("odometry_options/initialization", str_temp, "INIT_IMU");
  if (str_temp == "INIT_IMU")
    odometry_options.initialization = INIT_IMU;
  else if (str_temp == "INIT_CONSTANT_VELOCITY")
    odometry_options.initialization = INIT_CONSTANT_VELOCITY;
  else
    std::cout << "The `state_initialization` " << str_temp << " is not supported." << std::endl;

  icpOptions optimize_options;
  nh.param<int>(
      "icp_options/threshold_voxel_occupancy", odometry_options.optimize_options.threshold_voxel_occupancy, -1);
  nh.param<double>("icp_options/size_voxel_map", odometry_options.optimize_options.size_voxel_map, -1.0);
  nh.param<int>("icp_options/num_iters_icp", odometry_options.optimize_options.num_iters_icp, -5);
  nh.param<int>("icp_options/min_number_neighbors", odometry_options.optimize_options.min_number_neighbors, -20);
  nh.param<int>("icp_options/voxel_neighborhood", odometry_options.optimize_options.voxel_neighborhood, -1);
  nh.param<double>("icp_options/power_planarity", odometry_options.optimize_options.power_planarity, -2.0);
  nh.param<bool>(
      "icp_options/estimate_normal_from_neighborhood",
      odometry_options.optimize_options.estimate_normal_from_neighborhood,
      true);
  nh.param<int>("icp_options/max_number_neighbors", odometry_options.optimize_options.max_number_neighbors, -20);
  nh.param<double>("icp_options/max_dist_to_plane_icp", odometry_options.optimize_options.max_dist_to_plane_icp, -0.3);
  nh.param<double>(
      "icp_options/threshold_orientation_norm", odometry_options.optimize_options.threshold_orientation_norm, -0.0001);
  nh.param<double>(
      "icp_options/threshold_translation_norm", odometry_options.optimize_options.threshold_translation_norm, -0.001);
  nh.param<int>("icp_options/max_num_residuals", odometry_options.optimize_options.max_num_residuals, -1);
  nh.param<int>("icp_options/min_num_residuals", odometry_options.optimize_options.min_num_residuals, -100);
  nh.param<int>("icp_options/num_closest_neighbors", odometry_options.optimize_options.num_closest_neighbors, -1);
  nh.param<double>("icp_options/weight_alpha", odometry_options.optimize_options.weight_alpha, -0.9);
  nh.param<double>("icp_options/weight_neighborhood", odometry_options.optimize_options.weight_neighborhood, -0.1);
  nh.param<bool>("icp_options/debug_print", odometry_options.optimize_options.debug_print, true);
  nh.param<bool>("icp_options/debug_viz", odometry_options.optimize_options.debug_viz, false);

  nh.param<double>("map_options/size_voxel_map", map_options.size_voxel_map, -0.1);
  nh.param<int>("map_options/max_num_points_in_voxel", map_options.max_num_points_in_voxel, -20);
  nh.param<double>("map_options/min_distance_points", map_options.min_distance_points, -0.01);
  nh.param<int>("map_options/add_point_step", map_options.add_point_step, -4);
  nh.param<int>("map_options/pub_point_minimum_views", map_options.pub_point_minimum_views, -3);

  nh.param<double>("map_options/max_delta_trans", map_options.max_delta_trans, 0.0);
  nh.param<double>("map_options/max_delta_degree", map_options.max_delta_degree, 0.0);

  // gp options
  nh.param<bool>("gp3d/full_cover", gp_options_.full_cover, false);
  nh.param<bool>("gp3d/debug", gp_options_.debug, false);
  nh.param<bool>("gp3d/log_time", gp_options_.log_time, false);
  img_pro->setLogtime(gp_options_.log_time);

  nh.param<int>("gp3d/min_points_num_to_gp", gp_options_.min_points_num_to_gp, 0);
  nh.param<int>("gp3d/num_gp_side", gp_options_.num_gp_side, 0);
  nh.param<int>("gp3d/curr_cam_per_iter", gp_options_.curr_cam_per_iter, 0);
  nh.param<int>("gp3d/neighbour_size", gp_options_.neighbour_size, 0);
  nh.param<int>("gp3d/image_sliding_window", gp_options_.image_sliding_window, 0);

  nh.param<double>("gp3d/grid", gp_options_.grid, 0.0);
  nh.param<double>("gp3d/kernel_size", gp_options_.kernel_size, 0.0);
  nh.param<double>("gp3d/variance_sensor", gp_options_.variance_sensor, 0.0);
  nh.param<double>("gp3d/eigen_1", gp_options_.eigen_1, 0.0);
  nh.param<double>("gp3d/max_var_mean", gp_options_.max_var_mean, 0.0);
  nh.param<int>("gp3d/history_cam_per_iter", gp_options_.history_cam_per_iter, 0.0);

  // gsoptime options
  nh.param<int>("gs/empty_iterations", gsoptimParams.empty_iterations, 0);

  nh.param<float>("gs/scale_factor", gsoptimParams.scale_factor, 0.0f);
  nh.param<float>("gs/position_lr_init", gsoptimParams.position_lr_init, 0.0f);
  nh.param<float>("gs/position_lr_final", gsoptimParams.position_lr_final, 0.0f);
  nh.param<float>("gs/position_lr_delay_mult", gsoptimParams.position_lr_delay_mult, 0.0f);
  nh.param<float>("gs/feature_lr", gsoptimParams.feature_lr, 0.0f);
  nh.param<float>("gs/percent_dense", gsoptimParams.percent_dense, 0.0f);
  nh.param<float>("gs/opacity_lr", gsoptimParams.opacity_lr, 0.0f);
  nh.param<float>("gs/scaling_lr", gsoptimParams.scaling_lr, 0.0f);
  nh.param<float>("gs/rotation_lr", gsoptimParams.rotation_lr, 0.0f);
  nh.param<float>("gs/lambda_dssim", gsoptimParams.lambda_dssim, 0.0f);
  nh.param<float>("gs/lambda_depth_simi", gsoptimParams.lambda_depth_simi, 0.0f);
  nh.param<float>("gs/lambda_delta_depth_simi", gsoptimParams.lambda_delta_depth_simi, 0.0f);

  nh.param<bool>("gs/empty_gpu_cache", gsoptimParams.empty_gpu_cache, false);

  // map_options.recordParameters();
  // odometry_options.recordParameters();
}

void lioOptimization::processAndMergePointClouds(GSLIVM::GsForMaps& all_gs) {
  std::vector<GSLIVM::GsForMaps> new_gs_points_copy;
  {
    std::lock_guard<std::mutex> lock(gs_point_for_map_mutex);
    new_gs_points_copy = new_gs_for_map_points;
    new_gs_for_map_points.clear();
    new_gs_points_for_map_count = 0;
  }

  int index = 0;
  for (const auto& cloud : new_gs_points_copy) {
    if (index == 0) {
      all_gs.hash_posi_s = cloud.hash_posi_s;
      all_gs.indexes = cloud.indexes;
      all_gs.gs_xyzs = cloud.gs_xyzs;
      all_gs.gs_rgbs = cloud.gs_rgbs;
      all_gs.gs_covs = cloud.gs_covs;
    } else {
      all_gs.hash_posi_s.reserve(all_gs.hash_posi_s.size() + cloud.hash_posi_s.size());
      all_gs.hash_posi_s.insert(all_gs.hash_posi_s.end(), cloud.hash_posi_s.begin(), cloud.hash_posi_s.end());

      all_gs.indexes.reserve(all_gs.indexes.size() + cloud.indexes.size());
      all_gs.indexes.insert(all_gs.indexes.end(), cloud.indexes.begin(), cloud.indexes.end());

      all_gs.gs_xyzs = torch::cat({all_gs.gs_xyzs, cloud.gs_xyzs}, 0);
      all_gs.gs_rgbs = torch::cat({all_gs.gs_rgbs, cloud.gs_rgbs}, 0);
      all_gs.gs_covs = torch::cat({all_gs.gs_covs, cloud.gs_covs}, 0);
    }
    index++;
  }
}

void lioOptimization::processAndMergeLosses(GSLIVM::GsForLosses& all_gs_losses) {
  std::vector<GSLIVM::GsForLosses> new_gs_losses_copy;
  {
    std::lock_guard<std::mutex> lock(gs_point_for_loss_mutex);
    new_gs_losses_copy = new_gs_for_loss_points;
    new_gs_for_loss_points.clear();
  }

  for (const auto& losses_item : new_gs_losses_copy) {
    for (const auto& [key, tensor] : losses_item._losses) {
      if (all_gs_losses._losses.find(key) == all_gs_losses._losses.end()) {
        all_gs_losses._losses[key] = tensor;
      } else {
        all_gs_losses._losses[key] = torch::cat({all_gs_losses._losses[key], tensor}, 0);
      }
    }
  }
}

void lioOptimization::allocateMemory() {
  cloud_pro.reset(new cloudProcessing());
  eskf_pro.reset(new eskfEstimator());
  img_pro.reset(new imageProcessing());

  color_points_world.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

  // gs
  gaussian_pro.reset(new GaussianModel(gsmodelParams.sh_degree));

  conv_window = gaussian_splatting::create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);
  // blue sky
  // background = gsmodelParams.white_background
  //                  ? torch::tensor({0.38f, 0.482f, 0.59f})
  //                  : torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kCUDA);
  // white sky
  background = gsmodelParams.white_background
                   ? torch::tensor({1.0f, 1.0f, 1.0f})
                   : torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kCUDA);
}

void lioOptimization::initialValue() {
  laser_point_cov = 0.001;

  G = vec3FromArray(v_G);
  G_norm = G.norm();

  // Til
  R_imu_lidar = mat33FromArray(v_extrin_R_imu_lidar);
  t_imu_lidar = vec3FromArray(v_extrin_t_imu_lidar);
  Til.block<3, 3>(0, 0) = R_imu_lidar;
  Til.block<3, 1>(0, 3) = t_imu_lidar;

  // Tic
  R_imu_camera = mat33FromArray(v_extrin_R_imu_camera);
  t_imu_camera = vec3FromArray(v_extrin_t_imu_camera);
  Eigen::Matrix4d Tic = Eigen::Matrix4d::Identity();
  Tic.block<3, 3>(0, 0) = R_imu_camera;
  Tic.block<3, 1>(0, 3) = t_imu_camera;

  cloud_pro->setExtrinR(R_imu_lidar);
  cloud_pro->setExtrinT(t_imu_lidar);

  img_pro->setCameraIntrinsic(v_camera_intrinsic);
  img_pro->setCameraDistCoeffs(v_camera_dist_coeffs);
  img_pro->initCameraParams();
  img_pro->setExtrinR(R_imu_camera);
  img_pro->setExtrinT(t_imu_camera);

  // Tcl
  Tcl = Tic.inverse() * Til;
  R_camera_lidar = Tcl.block<3, 3>(0, 0);
  t_camera_lidar = Tcl.block<3, 1>(0, 3);

  dt_sum = 0;

  start_time_img = -1.0;

  last_time_lidar = -1.0;
  last_time_imu = -1.0;
  last_time_img = -1.0;
  last_get_measurement = -1.0;
  last_rendering = false;
  last_time_frame = -1.0;
  current_time = -1.0;

  index_frame = 1;

  odometry_options.optimize_options.init_num_frames = odometry_options.init_num_frames;

  img_pro->printParameter();
  // std::cout << "R_imu_lidar: \n" << std::fixed << R_imu_lidar << std::endl;
  // std::cout << "t_imu_lidar: \n" << std::fixed << t_imu_lidar.transpose() << std::endl;
  // std::cout << "R_camera_lidar: \n" << std::fixed << R_camera_lidar << std::endl;
  // std::cout << "t_camera_lidar: \n" << std::fixed << t_camera_lidar.transpose() << std::endl;
}

void lioOptimization::addPointToMap(
    voxelHashMap& map,
    rgbPoint& point,
    double voxel_size,
    int max_num_points_in_voxel,
    double min_distance_points,
    int min_num_points,
    cloudFrame* p_frame) {
  short kx = static_cast<short>(point.getPosition().x() / voxel_size);
  short ky = static_cast<short>(point.getPosition().y() / voxel_size);
  short kz = static_cast<short>(point.getPosition().z() / voxel_size);

  voxelHashMap::iterator search = map.find(voxel(kx, ky, kz));

  if (search != map.end()) {
    auto& voxel_block = (search.value());

    if (!voxel_block.IsFull()) {
      double sq_dist_min_to_points = 10 * voxel_size * voxel_size;

      for (int i(0); i < voxel_block.NumPoints(); ++i) {
        auto& _point = voxel_block.points[i];
        double sq_dist = (_point.getPosition() - point.getPosition()).squaredNorm();
        if (sq_dist < sq_dist_min_to_points) {
          sq_dist_min_to_points = sq_dist;
        }
      }

      if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
        if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
          voxel_block.AddPoint(point);
          addPointToPcl(points_world, point, p_frame);
        }
      }
    }
  } else {
    if (min_num_points <= 0) {
      voxelBlock voxel_block(max_num_points_in_voxel);
      voxel_block.AddPoint(point);
      map[voxel(kx, ky, kz)] = std::move(voxel_block);
    }
  }
}

void lioOptimization::addPointToColorMap(
    voxelHashMap& map,
    rgbPoint& point,
    double voxel_size,
    int max_num_points_in_voxel,
    double min_distance_points,
    int min_num_points,
    cloudFrame* p_frame,
    std::vector<voxelId>& voxels_recent_visited_temp) {
  bool add_point = true;

  int point_map_kx = static_cast<short>(point.getPosition().x() / min_distance_points);
  int point_map_ky = static_cast<short>(point.getPosition().y() / min_distance_points);
  int point_map_kz = static_cast<short>(point.getPosition().z() / min_distance_points);

  int kx = static_cast<short>(point.getPosition().x() / voxel_size);
  int ky = static_cast<short>(point.getPosition().y() / voxel_size);
  int kz = static_cast<short>(point.getPosition().z() / voxel_size);

  if (hashmap_3d_points.if_exist(point_map_kx, point_map_ky, point_map_kz)) {
    add_point = false;
  }

  voxelHashMap::iterator search = map.find(voxel(kx, ky, kz));

  if (search != map.end()) {
    auto& voxel_block = (search.value());

    if (!voxel_block.IsFull()) {
      if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
        voxel_block.AddPoint(point);

        if (add_point) {
          std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_rgb_points_vec);
          point.point_index = img_pro->map_tracker->rgb_points_vec.size();
          img_pro->map_tracker->rgb_points_vec.push_back(&voxel_block.points.back());
          hashmap_3d_points.insert(
              point_map_kx, point_map_ky, point_map_kz, img_pro->map_tracker->rgb_points_vec.back());
        }
      }
    }

    if (fabs(p_frame->time_sweep_end - img_pro->time_last_process) > 1e-5 &&
        fabs(voxel_block.last_visited_time - p_frame->time_sweep_end) > 1e-5) {
      voxel_block.last_visited_time = p_frame->time_sweep_end;
      voxels_recent_visited_temp.push_back(voxelId(kx, ky, kz));
    }
  } else {
    if (min_num_points <= 0) {
      voxelBlock voxel_block(max_num_points_in_voxel);
      voxel_block.AddPoint(point);
      map[voxel(kx, ky, kz)] = std::move(voxel_block);

      if (add_point) {
        std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_rgb_points_vec);
        point.point_index = img_pro->map_tracker->rgb_points_vec.size();
        img_pro->map_tracker->rgb_points_vec.push_back(&map[voxel(kx, ky, kz)].points.back());
        hashmap_3d_points.insert(point_map_kx, point_map_ky, point_map_kz, img_pro->map_tracker->rgb_points_vec.back());
      }

      if (fabs(p_frame->time_sweep_end - img_pro->time_last_process) > 1e-5 &&
          fabs(map[voxel(kx, ky, kz)].last_visited_time - p_frame->time_sweep_end) > 1e-5) {
        map[voxel(kx, ky, kz)].last_visited_time = p_frame->time_sweep_end;
        voxels_recent_visited_temp.push_back(voxelId(kx, ky, kz));
      }
    }
  }
}

void lioOptimization::addPointsToMap(
    voxelHashMap& map,
    cloudFrame* p_frame,
    double voxel_size,
    int max_num_points_in_voxel,
    double min_distance_points,
    int min_num_points,
    bool to_rendering) {
  if (to_rendering) {
    voxels_recent_visited_temp.clear();
    std::vector<voxelId>().swap(voxels_recent_visited_temp);
  }

  int number_of_voxels_before_add = voxels_recent_visited_temp.size();

  int point_idx = 0;

  for (const auto& point : p_frame->point_frame) {
    rgbPoint rgb_point(point.point);
    addPointToMap(map, rgb_point, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points, p_frame);

    if (point_idx % map_options.add_point_step == 0)
      addPointToColorMap(
          color_voxel_map,
          rgb_point,
          map_options.size_voxel_map,
          map_options.max_num_points_in_voxel,
          map_options.min_distance_points,
          0,
          p_frame,
          voxels_recent_visited_temp);

    point_idx++;
  }

  if (to_rendering) {
    img_pro->map_tracker->voxels_recent_visited.clear();
    std::vector<voxelId>().swap(img_pro->map_tracker->voxels_recent_visited);
    img_pro->map_tracker->voxels_recent_visited = voxels_recent_visited_temp;
    img_pro->map_tracker->number_of_new_visited_voxel =
        img_pro->map_tracker->voxels_recent_visited.size() - number_of_voxels_before_add;
  }

  if (ENABLE_PUBLISH) {
    publishCLoudWorld(pub_cloud_world, points_world, p_frame);
  }
  points_world->clear();
}

void lioOptimization::removePointsFarFromLocation(voxelHashMap& map, const Eigen::Vector3d& location, double distance) {
  std::vector<voxel> voxels_to_erase;

  for (auto& pair : map) {
    rgbPoint rgb_point = pair.second.points[0];
    Eigen::Vector3d pt = rgb_point.getPosition();
    if ((pt - location).squaredNorm() > (distance * distance)) {
      voxels_to_erase.push_back(pair.first);
    }
  }

  for (auto& vox : voxels_to_erase)
    map.erase(vox);

  std::vector<voxel>().swap(voxels_to_erase);
}

size_t lioOptimization::mapSize(const voxelHashMap& map) {
  size_t map_size(0);
  for (auto& itr_voxel_map : map) {
    map_size += (itr_voxel_map.second).NumPoints();
  }
  return map_size;
}

void lioOptimization::standardCloudHandler(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  assert(msg->header.stamp.toSec() > last_time_lidar);

  cloud_pro->process(msg, point_buffer);

  assert(msg->header.stamp.toSec() > last_time_lidar);
  last_time_lidar = msg->header.stamp.toSec();
}

void lioOptimization::livoxHandler(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
  assert(msg->header.stamp.toSec() > last_time_lidar);

  cloud_pro->livoxHandler(msg, point_buffer);

  assert(msg->header.stamp.toSec() > last_time_lidar);
  last_time_lidar = msg->header.stamp.toSec();
}

void lioOptimization::heartHandler(const ros::TimerEvent& event) {
  if (is_gs_started && !is_received_data) {
    stop_thread = true;
  }
  is_received_data = false;
}

void lioOptimization::imuHandler(const sensor_msgs::Imu::ConstPtr& msg) {
  is_received_data = true;
  sensor_msgs::Imu::Ptr msg_temp(new sensor_msgs::Imu(*msg));

  if (abs(time_diff) > 0.1 && time_diff_enable) {
    msg_temp->header.stamp = ros::Time().fromSec(time_diff + msg->header.stamp.toSec());
  }

  assert(msg_temp->header.stamp.toSec() > last_time_imu);

  imu_buffer.push(msg_temp);

  assert(msg_temp->header.stamp.toSec() > last_time_imu);
  last_time_imu = msg_temp->header.stamp.toSec();

  if (last_get_measurement < 0) {
    last_get_measurement = last_time_imu;
  }
}

void lioOptimization::imageHandler(const sensor_msgs::ImageConstPtr& msg) {
  if (image_filter_index % image_filter_num != 0) {
    image_filter_index++;
    return;
  }
  image_filter_index = 1;

  cv::Mat image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image.clone();

  if (image.cols != image_width_verify || image.rows != image_height_verify) {
    std::cout << image.cols << " != " << image_width_verify << " && " << image.rows << " != " << image_height_verify
              << std::endl;
    exit(-1);
  }

  assert(msg->header.stamp.toSec() > last_time_img);

  ImageTs img_ts;
  img_ts.image = image;
  img_ts.timestamp = msg->header.stamp.toSec();
  time_img_buffer.push(img_ts);

  assert(msg->header.stamp.toSec() > last_time_img);
  if (last_time_img == -1.0) {
    start_time_img = msg->header.stamp.toSec();
  }
  last_time_img = msg->header.stamp.toSec();
}

void lioOptimization::compressedImageHandler(const sensor_msgs::CompressedImageConstPtr& msg) {
  if (image_filter_index % image_filter_num != 0) {
    image_filter_index++;
    return;
  }
  image_filter_index = 1;

  cv::Mat image;

  try {
    cv_bridge::CvImagePtr cv_ptr_compressed = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    image = cv_ptr_compressed->image;
    cv_ptr_compressed->image.release();
  } catch (cv_bridge::Exception& e) {
    printf("Could not convert from '%s' to 'bgr8' !!! ", msg->format.c_str());
  }

  if (image.cols != image_width_verify || image.rows != image_height_verify) {
    std::cout << image.cols << " != " << image_width_verify << " && " << image.rows << " != " << image_height_verify
              << std::endl;
    exit(-1);
  }

  assert(msg->header.stamp.toSec() > last_time_img);

  ImageTs img_ts;
  img_ts.image = image;
  img_ts.timestamp = msg->header.stamp.toSec();
  time_img_buffer.push(img_ts);

  if (last_time_img == -1.0) {
    start_time_img = msg->header.stamp.toSec();
  }
  last_time_img = msg->header.stamp.toSec();
}

std::vector<Measurements> lioOptimization::getMeasurements() {
  std::vector<Measurements> measurements;

  while (true) {
    if (imu_buffer.empty() || time_img_buffer.empty() || point_buffer.empty()) {
      return measurements;
    }

    if (point_buffer.back().timestamp <= time_img_buffer.front().timestamp) {
      return measurements;
    }

    if (abs(point_buffer.front().timestamp - time_img_buffer.front().timestamp) > 1e3) {
      std::cout << "data time fault " << point_buffer.empty() << std::endl;
      std::cout << "points: " << point_buffer.front().timestamp << ", " << time_img_buffer.front().timestamp << " "
                << (point_buffer.front().timestamp < time_img_buffer.front().timestamp) << std::endl;

      std::cout << point_buffer.size() << " " << time_img_buffer.size() << " " << time_img_buffer.size() << std::endl;
      point_buffer.pop();
    }

    if (point_buffer.front().timestamp >= time_img_buffer.front().timestamp) {
      time_img_buffer.front().image.release();
      time_img_buffer.pop();
      continue;
    }

    // std::cout << "imu: " << imu_buffer.front()->header.stamp.toSec() << ", " << time_img_buffer.front().timestamp << std::endl;

    if (imu_buffer.back()->header.stamp.toSec() <= time_img_buffer.front().timestamp) {
      return measurements;
    }

    if (imu_buffer.front()->header.stamp.toSec() >= time_img_buffer.front().timestamp) {
      time_img_buffer.front().image.release();
      time_img_buffer.pop();
      continue;
    }

    Measurements measurement;

    if (last_get_measurement + cloud_pro->getSweepInterval() <
        time_img_buffer.front().timestamp - 1 * cloud_pro->getSweepInterval()) {
      measurement.time_image = last_get_measurement + cloud_pro->getSweepInterval();

      while (imu_buffer.front()->header.stamp.toSec() < last_get_measurement + cloud_pro->getSweepInterval()) {
        measurement.imu_measurements.emplace_back(imu_buffer.front());
        imu_buffer.pop();
      }

      measurement.imu_measurements.emplace_back(imu_buffer.front());

      while (point_buffer.front().timestamp < last_get_measurement + cloud_pro->getSweepInterval()) {
        measurement.lidar_points.push_back(point_buffer.front());
        point_buffer.pop();
      }

      measurement.time_sweep.first = last_get_measurement;
      measurement.time_sweep.second = cloud_pro->getSweepInterval();

      measurement.rendering = false;

      if (measurement.lidar_points.size() > 0) {
        measurements.emplace_back(measurement);
        last_rendering = measurement.rendering;
      }

      last_get_measurement = last_get_measurement + cloud_pro->getSweepInterval();

      break;
    } else {
      measurement.time_image = time_img_buffer.front().timestamp;
      measurement.image = time_img_buffer.front().image.clone();

      time_img_buffer.front().image.release();
      time_img_buffer.pop();

      while (imu_buffer.front()->header.stamp.toSec() < measurement.time_image) {
        measurement.imu_measurements.emplace_back(imu_buffer.front());
        imu_buffer.pop();
      }

      measurement.imu_measurements.emplace_back(imu_buffer.front());

      while (point_buffer.front().timestamp < measurement.time_image) {
        measurement.lidar_points.push_back(point_buffer.front());
        point_buffer.pop();
      }

      measurement.time_sweep.first = last_get_measurement;
      measurement.time_sweep.second = measurement.time_image - last_get_measurement;

      measurement.rendering = true;

      if (measurement.lidar_points.size() > 0) {
        measurements.emplace_back(measurement);
        last_rendering = measurement.rendering;
      }

      last_get_measurement = measurement.time_image;

      break;
    }
  }

  return measurements;
}

void lioOptimization::makePointTimestamp(std::vector<point3D>& sweep, double time_begin, double time_end) {
  if (cloud_pro->isPointTimeEnable()) {
    double delta_t = time_end - time_begin;

    for (int i = 0; i < sweep.size(); i++) {
      sweep[i].relative_time = sweep[i].timestamp - time_begin;
      sweep[i].alpha_time = sweep[i].relative_time / delta_t;
      sweep[i].relative_time = sweep[i].relative_time * 1000.0;
      if (sweep[i].alpha_time > 1.0)
        sweep[i].alpha_time = 1.0 - 1e-5;
    }
  } else {
    double delta_t = time_end - time_begin;

    std::vector<point3D>::iterator iter = sweep.begin();

    while (iter != sweep.end()) {
      if ((*iter).timestamp > time_end)
        iter = sweep.erase(iter);
      else if ((*iter).timestamp < time_begin)
        iter = sweep.erase(iter);
      else {
        (*iter).relative_time = (*iter).timestamp - time_begin;
        (*iter).alpha_time = (*iter).relative_time / delta_t;
        (*iter).relative_time = (*iter).relative_time * 1000.0;
        iter++;
      }
    }
  }
}

cloudFrame* lioOptimization::buildFrame(
    std::vector<point3D>& cut_sweep,
    state* cur_state,
    double timestamp_begin,
    double timestamp_offset) {
  std::vector<point3D> frame(cut_sweep);

  double offset_begin = 0;
  double offset_end = timestamp_offset;

  double time_sweep_begin = timestamp_begin;
  double time_frame_begin = timestamp_begin;

  makePointTimestamp(frame, time_frame_begin, timestamp_begin + timestamp_offset);

  if (odometry_options.motion_compensation == CONSTANT_VELOCITY)
    distortFrameByConstant(frame, imu_states, time_frame_begin, R_imu_lidar, t_imu_lidar);
  else if (odometry_options.motion_compensation == IMU)
    distortFrameByImu(frame, imu_states, time_frame_begin, R_imu_lidar, t_imu_lidar);

  double sample_size =
      index_frame < odometry_options.init_num_frames ? odometry_options.init_voxel_size : odometry_options.voxel_size;

  std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
  std::shuffle(frame.begin(), frame.end(), engine);

  if (odometry_options.voxel_size > 0) {
    subSampleFrame(frame, sample_size);

    std::shuffle(frame.begin(), frame.end(), engine);
  }

  transformAllImuPoint(frame, imu_states, R_imu_lidar, t_imu_lidar);

  double dt_offset = 0;

  if (index_frame > 1)
    dt_offset -= time_frame_begin - all_cloud_frame.back()->time_sweep_end;

  if (index_frame <= 2) {
    for (auto& point_temp : frame) {
      point_temp.alpha_time = 1.0;
    }
  }

  if (index_frame > 2) {
    for (auto& point_temp : frame) {
      transformPoint(point_temp, cur_state->rotation, cur_state->translation, R_imu_lidar, t_imu_lidar);
    }
  } else {
    for (auto& point_temp : frame) {
      Eigen::Quaterniond q_identity = Eigen::Quaterniond::Identity();
      Eigen::Vector3d t_zero = Eigen::Vector3d::Zero();
      transformPoint(point_temp, q_identity, t_zero, R_imu_lidar, t_imu_lidar);
    }
  }

  cloudFrame* p_frame = new cloudFrame(frame, cur_state);
  p_frame->time_sweep_begin = time_sweep_begin;
  p_frame->time_sweep_end = timestamp_begin + timestamp_offset;
  p_frame->time_frame_begin = time_frame_begin;
  p_frame->time_frame_end = p_frame->time_sweep_end;
  p_frame->offset_begin = offset_begin;
  p_frame->offset_end = offset_end;
  p_frame->dt_offset = dt_offset;
  p_frame->id = all_cloud_frame.size();
  p_frame->sub_id = 0;
  p_frame->frame_id = index_frame;

  all_cloud_frame.push_back(p_frame);

  return p_frame;
}

void lioOptimization::stateInitialization(state* cur_state) {
  if (index_frame <= 2) {
    cur_state->rotation = Eigen::Quaterniond::Identity();
    cur_state->translation = Eigen::Vector3d::Zero();
  } else if (index_frame == 3) {
    if (odometry_options.initialization == INIT_CONSTANT_VELOCITY) {
      Eigen::Quaterniond q_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                      all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                      all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation;

      Eigen::Vector3d t_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation +
                                   all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                       all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                       (all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation -
                                        all_cloud_frame[all_cloud_frame.size() - 2]->p_state->translation);

      cur_state->rotation = q_next_end;
      cur_state->translation = t_next_end;
    } else if (odometry_options.initialization == INIT_IMU) {
      if (initial_flag) {
        cur_state->rotation = eskf_pro->getRotation();
        cur_state->translation = eskf_pro->getTranslation();
      } else {
        Eigen::Quaterniond q_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                        all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                        all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation;

        Eigen::Vector3d t_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation +
                                     all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                         all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                         (all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation -
                                          all_cloud_frame[all_cloud_frame.size() - 2]->p_state->translation);

        cur_state->rotation = q_next_end;
        cur_state->translation = t_next_end;
      }
    } else {
      cur_state->rotation = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation;
      cur_state->translation = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation;
    }
  } else {
    if (odometry_options.initialization == INIT_CONSTANT_VELOCITY) {
      Eigen::Quaterniond q_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                      all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                      all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation;

      Eigen::Vector3d t_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation +
                                   all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                       all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                       (all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation -
                                        all_cloud_frame[all_cloud_frame.size() - 2]->p_state->translation);

      cur_state->rotation = q_next_end;
      cur_state->translation = t_next_end;
    } else if (odometry_options.initialization == INIT_IMU) {
      if (initial_flag) {
        cur_state->rotation = eskf_pro->getRotation();
        cur_state->translation = eskf_pro->getTranslation();
      } else {
        Eigen::Quaterniond q_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                        all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                        all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation;

        Eigen::Vector3d t_next_end = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation +
                                     all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation *
                                         all_cloud_frame[all_cloud_frame.size() - 2]->p_state->rotation.inverse() *
                                         (all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation -
                                          all_cloud_frame[all_cloud_frame.size() - 2]->p_state->translation);

        cur_state->rotation = q_next_end;
        cur_state->translation = t_next_end;
      }
    } else {
      cur_state->rotation = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->rotation;
      cur_state->translation = all_cloud_frame[all_cloud_frame.size() - 1]->p_state->translation;
    }
  }
}

optimizeSummary lioOptimization::stateEstimation(cloudFrame* p_frame, bool to_rendering) {
  icpOptions optimize_options = odometry_options.optimize_options;
  const double kSizeVoxelInitSample = odometry_options.voxel_size;

  const double kSizeVoxelMap = optimize_options.size_voxel_map;
  const double kMinDistancePoints = odometry_options.min_distance_points;
  const int kMaxNumPointsInVoxel = odometry_options.max_num_points_in_voxel;

  optimizeSummary optimize_summary;

  if (p_frame->frame_id > 1) {
    bool good_enough_registration = false;
    double sample_voxel_size = p_frame->frame_id < odometry_options.init_num_frames
                                   ? odometry_options.init_sample_voxel_size
                                   : odometry_options.sample_voxel_size;
    double min_voxel_size = std::min(odometry_options.init_voxel_size, odometry_options.voxel_size);

    optimize_summary = optimize(p_frame, optimize_options, sample_voxel_size);

    if (!optimize_summary.success) {
      return optimize_summary;
    }
  } else {
    p_frame->p_state->translation = eskf_pro->getTranslation();
    p_frame->p_state->rotation = eskf_pro->getRotation();
    p_frame->p_state->velocity = eskf_pro->getVelocity();
    p_frame->p_state->ba = eskf_pro->getBa();
    p_frame->p_state->bg = eskf_pro->getBg();
    G = eskf_pro->getGravity();
    G_norm = G.norm();
  }

  addPointsToMap(voxel_map, p_frame, kSizeVoxelMap, kMaxNumPointsInVoxel, kMinDistancePoints, 0, to_rendering);

  return optimize_summary;
}

bool lioOptimization::compareStatesImageAdd(
    const Eigen::Quaterniond& R1,
    const Eigen::Vector3d& t1,
    double& time_1,
    const Eigen::Quaterniond& R2,
    const Eigen::Vector3d& t2,
    double& time_2) {
  Eigen::Quaterniond dq = R1 * R2.conjugate();
  dq.normalize();

  auto deltaR = 2 * acos(dq.w()) * 180 / M_PI;

  auto deltaT = t1 - t2;

  auto deltaTime = time_2 - time_1;

  assert(deltaTime > 0);
  return deltaT.norm() > map_options.max_delta_trans || abs(deltaR) > map_options.max_delta_degree;
}

void lioOptimization::gsPointCloudUpdate(
    cloudFrame*& p_frame,
    int& updated_voxel_count,
    GSLIVM::GsForMaps& final_gs_sample,
    GSLIVM::GsForLosses& final_gs_calc_loss) {
  auto tmp_pc = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  {
    std::lock_guard<std::mutex> guard(color_points_mutex);
    *tmp_pc = *color_points_world;
    color_points_world->clear();
  }

  if (tmp_pc->size() == 0) {
    return;
  }

  GSLIVM::ImageRt last_image2gp;
  last_image2gp.image = p_frame->rgb_image;
  last_image2gp.R = p_frame->p_state->rotation;
  last_image2gp.t = p_frame->p_state->translation;
  last_image2gp.R_imu_lidar = R_imu_lidar;
  last_image2gp.t_imu_lidar = t_imu_lidar;
  last_image2gp.R_camera_lidar = R_camera_lidar;
  last_image2gp.t_camera_lidar = t_camera_lidar;

  std::vector<GSLIVM::needGPUdata> updataMapData;
  std::vector<GSLIVM::GsForLoss> vector_final_gs_calc_loss;

  // gp reconstruction
  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        std::size_t num_points = tmp_pc->points.size();

        Eigen::Matrix<double, 3, -1> points_curr;
        points_curr.resize(3, num_points);

        for (size_t i = 0; i < num_points; ++i) {
          const auto& point = tmp_pc->points[i];
          points_curr(0, i) = static_cast<double>(point.x);
          points_curr(1, i) = static_cast<double>(point.y);
          points_curr(2, i) = static_cast<double>(point.z);
        }

        gpmap_pro->dividePointsIntoCellInitMap(
            last_image2gp, is_gs_started, points_curr, updataMapData, vector_final_gs_calc_loss);
      },
      "dividePointsIntoCellInitMap");

  if (updataMapData.size() != 0) {
    updated_voxel_count = updataMapData.size();
    std::vector<GSLIVM::varianceUpdate> updateVas;

    gpprocess_pro->updateCamParams(img_pro->getfx(), img_pro->getfy(), img_pro->getcx(), img_pro->getcy());

    std::vector<GSLIVM::GsForMap> vector_final_gs_sample;

    gpprocess_pro->forward_gp3d(
        updataMapData, updateVas, last_image2gp, vector_final_gs_sample, vector_final_gs_calc_loss);

    if (vector_final_gs_sample.size() > 0) {
      // for map
      std::vector<torch::Tensor> tensors_to_map_xyz, tensors_to_map_rgb, tensors_to_map_cov;

      for (int count = 0; count < vector_final_gs_sample.size(); count++) {
        final_gs_sample.hash_posi_s.push_back(vector_final_gs_sample[count].hash_posi_);

        int added_count = vector_final_gs_sample[count].gs_xyz.size(0);
        if (added_count == 0) {
          continue;
        }
        std::vector<int> tmp_indices(added_count);
        std::iota(tmp_indices.begin(), tmp_indices.end(), gaussian_pro->Get_gs_cache_size());
        // total len update
        gaussian_pro->Incre_gs_cache_size(added_count);

        final_gs_sample.indexes.push_back(tmp_indices);
        tensors_to_map_xyz.push_back(vector_final_gs_sample[count].gs_xyz);
        tensors_to_map_rgb.push_back(vector_final_gs_sample[count].gs_rgb);
        tensors_to_map_cov.push_back(vector_final_gs_sample[count].gs_cov);
      }
      final_gs_sample.gs_xyzs = torch::cat(tensors_to_map_xyz, 0);
      final_gs_sample.gs_rgbs = torch::cat(tensors_to_map_rgb, 0);
      final_gs_sample.gs_covs = torch::cat(tensors_to_map_cov, 0);
    }

    // for loss
    // if (vector_final_gs_calc_loss.size() > 0) {
    //   for (int count = 0; count < vector_final_gs_calc_loss.size(); count++) {
    //     final_gs_calc_loss._losses.try_emplace(
    //         vector_final_gs_calc_loss[count].hash_posi_, vector_final_gs_calc_loss[count].gs_xyz);
    //   }
    // }

    if (vector_final_gs_calc_loss.size() > 0) {
      for (const auto& entry : vector_final_gs_calc_loss) {
        std::size_t key = entry.hash_posi_;
        const torch::Tensor& tensor = entry.gs_xyz;

        if (final_gs_calc_loss._losses.find(key) == final_gs_calc_loss._losses.end()) {
          final_gs_calc_loss._losses[key] = tensor;
        } else {
          torch::Tensor existing_tensor = final_gs_calc_loss._losses[key];
          final_gs_calc_loss._losses[key] = torch::cat({existing_tensor, tensor}, 0);
        }
      }
    }

    // update variance
    gpmap_pro->updateVariance(updateVas);

    // for verbose in every n iters
    updateVas_size_iter_debug = updateVas.size();
  }
}

void lioOptimization::process(
    std::vector<point3D>& cut_sweep,
    double timestamp_begin,
    double timestamp_offset,
    cv::Mat& cur_image,
    bool to_rendering) {
  state* cur_state = new state();

  std::vector<point3D> const_frame;

  common::Timer::Evaluate(
      gp_options_.log_time, ros::Time::now().toSec(), [&]() { stateInitialization(cur_state); }, "stateInitialization");

  const_frame.insert(const_frame.end(), cut_sweep.begin(), cut_sweep.end());

  cloudFrame* p_frame;
  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() { p_frame = buildFrame(const_frame, cur_state, timestamp_begin, timestamp_offset); },
      "buildFrame");
  dt_sum = 0;

  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        stateEstimation(p_frame, to_rendering);

        if (all_cloud_frame.size() < 3) {
          p_frame->p_state->fx = img_pro->getCameraIntrinsic()(0, 0);
          p_frame->p_state->fy = img_pro->getCameraIntrinsic()(1, 1);
          p_frame->p_state->cx = img_pro->getCameraIntrinsic()(0, 2);
          p_frame->p_state->cy = img_pro->getCameraIntrinsic()(1, 2);

          p_frame->p_state->R_imu_camera = R_imu_camera;
          p_frame->p_state->t_imu_camera = t_imu_camera;
        } else {
          p_frame->p_state->fx = all_cloud_frame[all_cloud_frame.size() - 2]->p_state->fx;
          p_frame->p_state->fy = all_cloud_frame[all_cloud_frame.size() - 2]->p_state->fy;
          p_frame->p_state->cx = all_cloud_frame[all_cloud_frame.size() - 2]->p_state->cx;
          p_frame->p_state->cy = all_cloud_frame[all_cloud_frame.size() - 2]->p_state->cy;

          p_frame->p_state->R_imu_camera = all_cloud_frame[all_cloud_frame.size() - 2]->p_state->R_imu_camera;
          p_frame->p_state->t_imu_camera = all_cloud_frame[all_cloud_frame.size() - 2]->p_state->t_imu_camera;
        }

        p_frame->p_state->q_world_camera =
            Eigen::Quaterniond(p_frame->p_state->rotation.toRotationMatrix() * p_frame->p_state->R_imu_camera);
        p_frame->p_state->t_world_camera =
            p_frame->p_state->rotation.toRotationMatrix() * p_frame->p_state->t_imu_camera +
            p_frame->p_state->translation;
        p_frame->refreshPoseForProjection();

        if (to_rendering) {
          p_frame->rgb_image = cur_image;
          img_pro->process(color_voxel_map, p_frame);
        }
      },
      "StateEstimation_verbose");
  if (to_rendering) {
    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() {
          if (image_frame_id == 0 || compareStatesImageAdd(
                                         last_image_rotation,
                                         last_image_trans,
                                         last_image_time,
                                         p_frame->p_state->rotation,
                                         p_frame->p_state->translation,
                                         p_frame->time_frame_begin)) {
            std::vector<Camera> add_cams;
            gsAddCamera(p_frame, add_cams);

            _cameras.push_back(add_cams);

            last_image_rotation = p_frame->p_state->rotation;
            last_image_trans = p_frame->p_state->translation;
            last_image_time = p_frame->time_frame_begin;
          }
        },
        "GS_CameraExpansion_verbose");

    // build map and calculate loss
    GSLIVM::GsForMaps final_gs_sample;
    GSLIVM::GsForLosses final_gs_calc_loss;
    int updated_voxel_count = 0;
    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() { gsPointCloudUpdate(p_frame, updated_voxel_count, final_gs_sample, final_gs_calc_loss); },
        "GS_MapExpansion_verbose");

    // add initial gaussians into map queue
    if (final_gs_sample.gs_xyzs.size(0) != 0) {
      std::lock_guard<std::mutex> lock(gs_point_for_map_mutex);
      new_gs_for_map_points.push_back(final_gs_sample);
      new_gs_points_for_map_count += final_gs_sample.gs_xyzs.size(0);
    }

    // add initial gaussians into loss queue
    if (final_gs_calc_loss._losses.size() != 0) {
      std::lock_guard<std::mutex> lock(gs_point_for_loss_mutex);
      new_gs_for_loss_points.push_back(final_gs_calc_loss);
    }

    if (!is_gs_started && new_gs_points_for_map_count > 1000) {
      GSLIVM::GsForMaps all_gs;
      processAndMergePointClouds(all_gs);

      gaussian_pro->Create_from_pcd(gsoptimParams, all_gs, 1.f);
      gaussian_pro->Training_setup(gsoptimParams);

      // warm up
      {
        auto [image, depth, depth_sol] = render(_cameras[0][0], gaussian_pro, background);
        auto gt_image = _cameras[0][0].Get_original_image().to(torch::kCUDA, true);
        auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);

        ssim_loss.backward();
        gaussian_pro->_optimizer->step();
        gaussian_pro->_optimizer->zero_grad(true);
      }
      is_gs_started = true;
    } else {
    }
    image_frame_id++;
  }

  if (ENABLE_PUBLISH) {
    publish_path(pub_path, p_frame);
  }

  if (debug_output) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr p_cloud_temp;
    p_cloud_temp.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    point3DtoPCL(p_frame->point_frame, p_cloud_temp);

    std::string pcd_path(output_path + "/cloud_frame/" + std::to_string(index_frame) + std::string(".pcd"));
    saveCutCloud(pcd_path, p_cloud_temp);
  }

  int num_remove = 0;

  if (initial_flag) {
    if (index_frame > 1) {
      while (all_cloud_frame.size() > 2) {
        if (gp_options_.debug) {
          recordSinglePose(all_cloud_frame[0]);
        }
        all_cloud_frame[0]->release();
        all_cloud_frame.erase(all_cloud_frame.begin());
        num_remove++;
      }
      assert(all_cloud_frame.size() == 2);
    }
  } else {
    while (all_cloud_frame.size() > odometry_options.num_for_initialization) {
      if (gp_options_.debug) {
        recordSinglePose(all_cloud_frame[0]);
      }
      all_cloud_frame[0]->release();
      all_cloud_frame.erase(all_cloud_frame.begin());
      num_remove++;
    }
  }

  for (int i = 0; i < all_cloud_frame.size(); i++) {
    all_cloud_frame[i]->id = all_cloud_frame[i]->id - num_remove;
  }
}

void lioOptimization::optimize_vis() {
  std::chrono::time_point<std::chrono::high_resolution_clock> cool_start = std::chrono::high_resolution_clock::now();
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!is_gs_started) {
      continue;
    }
    if (stop_thread) {
      return;
    }

    if (!torch::cuda::is_available()) {
      // At the moment, I want to make sure that my GPU is utilized.
      std::cout << "CUDA is not available! Training on CPU." << std::endl;
      exit(-1);
    }

    // loop map queue and add new points to the 3d gaussians
    if ((new_gs_points_for_map_count != 0 && iter % 5 == 0) || new_gs_points_for_map_count > 1000) {
      if (new_gs_points_for_map_count == 0) {
        continue;
      }

      common::Timer::Evaluate(
          gp_options_.log_time,
          ros::Time::now().toSec(),
          [&]() {
            GSLIVM::GsForMaps all_gs;
            processAndMergePointClouds(all_gs);
            gaussian_pro->addNewPointcloud(gsoptimParams, all_gs, iter, 1.f);
          },
          "optimize_vis_gsMapUpdate");
    }

    std::vector<std::vector<Camera>> optimized_cams;
    std::vector<std::vector<Camera>> optimized_cams2;

    auto camera_size = _cameras.size();
    if (camera_size <= gp_options_.image_sliding_window * 2 + gp_options_.history_cam_per_iter) {
      std::cout << "<=== wait for enough cameras... "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count()
                << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }
    // here
    {
      // std::vector<int> numbers(camera_size);
      // for (int i = 0; i < camera_size; ++i) {
      //   numbers[i] = i;
      // }

      // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

      // std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));

      // auto curr_before = findAndErase(numbers, selected_indices_hist);

      // if (curr_before.empty()) {
      //   selected_indices_hist.clear();
      //   std::cout << "<=== optimize whole history image scequence again... " << std::endl;
      //   continue;
      // }

      // auto cams = _cameras[curr_before[0]];
      // std::cout << curr_before[0] << std::endl;
      // optimized_cams.push_back(cams);
      // for (int ii = 335; ii < 340; ++ii) {
      //   if (ii >= camera_size - 10) {
      //     continue;
      //   }

      //   auto cams = _cameras[ii];
      //   optimized_cams.push_back(cams);
      // }

      // selected_indices_hist.push_back(curr_before[0]);
    }
    {
      auto [curr_indices, curr_before] = get_random_indices(
          camera_size,
          selected_indices_curr,
          selected_indices_hist,
          gp_options_.image_sliding_window,
          gp_options_.curr_cam_per_iter,
          gp_options_.history_cam_per_iter);

      curr_indices = findAndErase(curr_indices, selected_indices_curr);
      curr_before = findAndErase(curr_before, selected_indices_hist);

      if (curr_indices.empty() && gp_options_.image_sliding_window != 0) {
        selected_indices_curr.clear();
        std::cout << "<=== optimize whole current image scequence again... " << std::endl;
        continue;
      }

      if (curr_before.empty()) {
        selected_indices_hist.clear();
        std::cout << "<=== optimize whole history image scequence again... " << std::endl;
        continue;
      }

      common::Timer::Evaluate(
          gp_options_.log_time,
          ros::Time::now().toSec(),
          [&]() {
            // current
            // std::cout << "optimized camera: ";
            // for (int camera_index = curr_indices.size() - 1;
            //      camera_index >= std::max((int)curr_indices.size() - gp_options_.curr_cam_per_iter, 0);
            //      camera_index--) {

            int cccount_curr = 0;
            for (auto camera_index : curr_indices) {
              if (cccount_curr >= gp_options_.curr_cam_per_iter) {
                break;
              }
              // first
              selected_indices_curr.push_back(camera_index);

              auto cams = _cameras[camera_index];
              optimized_cams.push_back(cams);
              // std::cout << camera_index << " ";
              // second
              // auto cams_ref = _cameras[camera_index + 1];
              // optimized_cams.push_back(cams_ref);
              cccount_curr++;
            }

            // history
            int cccount_hist = 0;

            // std::cout << camera_size << " camera_size " << std::endl;
            for (auto camera_index : curr_before) {
              if (cccount_hist >= gp_options_.history_cam_per_iter) {
                break;
              }

              // first
              selected_indices_hist.push_back(camera_index);

              auto cams = _cameras[camera_index];
              optimized_cams.push_back(cams);

              // second
              auto cams_ref = _cameras[camera_index + 1];
              optimized_cams.push_back(cams_ref);
              cccount_hist++;

              // std::cout << camera_index << " outer " << std::endl;
              // for (int ii = 326; ii < 329; ++ii) {
              //   if (ii >= camera_size) {
              //     continue;
              //   }

              //   // std::cout << ii << " inter " << _cameras.size() << std::endl;
              //   auto cams = _cameras[ii];
              //   optimized_cams2.push_back(cams);
              // }
            }
            // std::cout << std::endl;
          },
          "optimize_vis_getCameraIndex");
    }

    std::vector<torch::Tensor> losses;

    GSLIVM::GsForLosses all_gs_loss;
    // for similarity loss
    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() { processAndMergeLosses(all_gs_loss); },
        "optimize_vis_merge simi points");

    torch::Tensor simi_loss = torch::zeros({});
    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() {
          if (all_gs_loss._losses.size() != 0) {
            bool re = gaussian_pro->calcSimiLoss(all_gs_loss, simi_loss, gsoptimParams.lambda_depth_simi);
            if (re) {
              losses.push_back(simi_loss);
            }
          }
        },
        "optimize_vis_calc simi loss");

    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() {
          int cam_index = 0;

          std::vector<GSLIVM::DeltaSimi> delta_infos;

          for (auto& cam : optimized_cams2) {
            // Render
            auto [image, depth, depth_sol] = render(cam[0], gaussian_pro, background);
            torch::Tensor gt_image = cam[0].Get_original_image().to(torch::kCUDA, true);
            GSLIVM::DeltaSimi _delta;
            _delta.cam_pose_R = cam[0].Get_R();
            _delta.cam_pose_t = cam[0].Get_T();
            _delta.K = cam[0].Get_K().transpose();
            _delta.inv_K = _delta.K.inverse();
            _delta.depth = depth;
            _delta.depth_sol = depth_sol;
            delta_infos.push_back(_delta);

            // Loss Computations
            auto l1l = gaussian_splatting::l1_loss(image, gt_image);

            // render this image
            auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);

            auto image_loss = (1.f - gsoptimParams.lambda_dssim) * l1l + gsoptimParams.lambda_dssim * (1.f - ssim_loss);

            losses.push_back(image_loss);
          }

          for (auto& cam : optimized_cams) {
            // Render
            auto [image, depth, depth_sol] = render(cam[0], gaussian_pro, background);
            torch::Tensor gt_image = cam[0].Get_original_image().to(torch::kCUDA, true);
            GSLIVM::DeltaSimi _delta;
            _delta.cam_pose_R = cam[0].Get_R();
            _delta.cam_pose_t = cam[0].Get_T();
            _delta.K = cam[0].Get_K().transpose();
            _delta.inv_K = _delta.K.inverse();
            _delta.depth = depth;
            _delta.depth_sol = depth_sol;
            delta_infos.push_back(_delta);

            // Loss Computations
            auto l1l = gaussian_splatting::l1_loss(image, gt_image);

            // render this image
            auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);

            auto image_loss = (1.f - gsoptimParams.lambda_dssim) * l1l + gsoptimParams.lambda_dssim * (1.f - ssim_loss);

            losses.push_back(image_loss);

            // Update status line
            if (iter % 50 == 0 && cam_index == 0) {
              auto psnr_loss = gaussian_splatting::psnr(image, gt_image);
              float psnr_value = psnr_loss.item<float>();
              float ssim_value = ssim_loss.item<float>();

              auto render_image = tensor2CvMat3X(image);
              auto gt_imagexx = tensor2CvMat3X(gt_image);

              cv::Mat mergedImage;
              cv::hconcat(render_image, gt_imagexx, mergedImage);

              std::string imagePath = gsmodelParams.output_path.string() + "/training/" + cam[0].Get_image_name();

              if (!std::filesystem::exists(imagePath)) {
                cv::imwrite(imagePath, mergedImage);
              }

              std::stringstream status_line;
              status_line.imbue(std::locale(""));
              status_line << "\rIter: " << std::setw(6) << iter;
              status_line << "  Left: " << std::setw(6) << time_img_buffer.size();
              status_line << "  UpdateGP: " << std::setw(6) << all_gs_loss._losses.size();
              status_line << "  Cams: " << std::setw(6) << optimized_cams.size();
              status_line << "  Image Loss: " << std::fixed << std::setw(9) << std::setprecision(6)
                          << image_loss.item<float>();
              status_line << "  Simi Loss: " << std::fixed << std::setw(9) << std::setprecision(6)
                          << simi_loss.item<float>();
              status_line << "  Splats: " << std::setw(10) << (int)gaussian_pro->Get_xyz().size(0);
              status_line << "  PSNR: " << std::setw(10) << (float)psnr_value;
              status_line << "  SSIM: " << std::setw(10) << (float)ssim_value;
              const int curlen = status_line.str().length();
              const int ws = last_status_len - curlen;
              if (ws > 0) {
                status_line << std::string(ws, ' ');
              }
              std::cout << status_line.str() << std::flush << std::endl;
              last_status_len = curlen;
            }
            cam_index++;
          }

          for (int tmp_i = gp_options_.curr_cam_per_iter; tmp_i < optimized_cams.size(); tmp_i += 2) {
            auto renderedimage = gaussian_pro->calcDeltaSimi(delta_infos[tmp_i], delta_infos[tmp_i + 1]);

            auto inv_renderedimage = gaussian_splatting::inv_depth(renderedimage);
            auto inv_ref_depth = gaussian_splatting::inv_depth(delta_infos[tmp_i + 1].depth);

            auto mask_src = torch::ones_like(delta_infos[tmp_i].depth_sol, torch::kCUDA);
            auto mask_condition = delta_infos[tmp_i].depth_sol < 0.5;
            mask_src.masked_fill_(mask_condition, 0);

            auto mask_ref = torch::ones_like(delta_infos[tmp_i + 1].depth_sol, torch::kCUDA);
            auto mask_condition_ref = delta_infos[tmp_i + 1].depth_sol < 0.5;
            mask_ref.masked_fill_(mask_condition_ref, 0);

            auto rendered_image_mask = inv_renderedimage * mask_src * mask_ref;
            auto rendered_ref_image_mask = inv_ref_depth * mask_ref * mask_src;

            auto gap = torch::abs((rendered_image_mask - rendered_ref_image_mask));

            auto delta_simi_loss = gsoptimParams.lambda_delta_depth_simi * gap.mean();

            losses.push_back(delta_simi_loss);
            if (iter % 200 == 0 || delta_simi_loss.item<float>() > 0.5) {
              auto rep_image = tensor2CvMat2X(renderedimage);
              auto ref_image = tensor2CvMat2X(delta_infos[tmp_i + 1].depth);
              cv::Mat mergeddepthImage;
              cv::hconcat(rep_image, ref_image, mergeddepthImage);
              cv::imwrite(gsmodelParams.output_path.string() + "/training/latest_depth.jpg", mergeddepthImage);

              // saveDepthMapAsNPY(
              //     delta_infos[tmp_i + 1].depth_sol, gsmodelParams.output_path.string() + "/training/latest_sol.npy");
              // torch::Tensor merged_depth = torch::cat({rendered_image_mask, rendered_ref_image_mask}, 1);
              // saveDepthMapAsNPY(merged_depth, gsmodelParams.output_path.string() + "/training/latest_depth.npy");
            }
          }
        },
        "GS_Render_verbose");

    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() {
          torch::Tensor loss = torch::zeros({}).to(torch::kCUDA);
          for (auto& los : losses) {
            loss += los;
          }
          if (losses.size() != 0) {
            loss.backward();
          }

          //  Optimizer step
          gaussian_pro->_optimizer->step();
          gaussian_pro->_optimizer->zero_grad(true);
        },
        "Backward_Step_verbose");

    common::Timer::Evaluate(
        gp_options_.log_time,
        ros::Time::now().toSec(),
        [&]() {
          if (gsoptimParams.empty_gpu_cache && iter % gsoptimParams.empty_iterations == 0) {
            c10::cuda::CUDACachingAllocator::emptyCache();
          }
        },
        "optimize_vis_emptyGPUCache");
    iter++;
  }
}

std::vector<int> lioOptimization::findAndErase(std::vector<int>& vec1, const std::vector<int>& vec2) {
  for (auto it = vec1.begin(); it != vec1.end();) {
    if (std::find(vec2.begin(), vec2.end(), *it) != vec2.end()) {
      vec1.erase(it);
    } else {
      ++it;
    }
  }
  return vec1;
}

std::pair<std::vector<int>, std::vector<int>> lioOptimization::get_random_indices(
    int max_size,
    std::vector<int> exist_curr,
    std::vector<int> exist_hist,
    int window_size,
    int curr_size,
    int hist_size) {

  const int split_index = std::max(0, max_size - window_size);
  std::vector<int> indices_recent_sample;
  std::vector<int> indices_before_sample;

  if (window_size > 0 && curr_size > 0) {
    std::vector<int> recent_candidates;
    recent_candidates.reserve(max_size - split_index);
    for (int i = split_index; i < max_size; ++i) {
      recent_candidates.push_back(i);
    }

    std::vector<int> valid_recent;
    for (int idx : recent_candidates) {
      if (std::find(exist_curr.begin(), exist_curr.end(), idx) == exist_curr.end()) {
        valid_recent.push_back(idx);
      }
    }

    int sample_count = std::min(curr_size, static_cast<int>(valid_recent.size()));
    indices_recent_sample = std::vector<int>(valid_recent.begin(), valid_recent.begin() + sample_count);
  }

  if (split_index > 0 && hist_size > 0) {
    std::vector<int> before_candidates;
    before_candidates.reserve(split_index);
    for (int i = 0; i < split_index; ++i) {
      before_candidates.push_back(i);
    }

    std::vector<int> valid_before;
    for (int idx : before_candidates) {
      if (std::find(exist_hist.begin(), exist_hist.end(), idx) == exist_hist.end()) {
        valid_before.push_back(idx);
      }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(valid_before.begin(), valid_before.end(), gen);

    int sample_count = std::min(hist_size, static_cast<int>(valid_before.size()));
    indices_before_sample = std::vector<int>(valid_before.begin(), valid_before.begin() + sample_count);
  }

  return std::make_pair(indices_recent_sample, indices_before_sample);
}

void lioOptimization::gsAddCamera(cloudFrame* p_frame, std::vector<Camera>& cams) {
  torch::Tensor original_image_tensor;
  cv::Mat rgbimage;
  cv::cvtColor(p_frame->rgb_image, rgbimage, cv::COLOR_BGR2RGB);
  original_image_tensor = torch::from_blob(rgbimage.data, {p_frame->image_rows, p_frame->image_cols, 3}, torch::kUInt8);
  original_image_tensor = original_image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;

  cams.push_back(Camera(
      index_frame,
      p_frame->p_state->q_world_camera.toRotationMatrix().cast<float>(),
      p_frame->p_state->t_world_camera.cast<float>(),
      static_cast<float>(p_frame->p_state->fx),
      static_cast<float>(p_frame->p_state->fy),
      static_cast<float>(p_frame->p_state->cx),
      static_cast<float>(p_frame->p_state->cy),
      focal2fov(static_cast<float>(p_frame->p_state->fx), p_frame->image_cols),
      focal2fov(static_cast<float>(p_frame->p_state->fy), p_frame->image_rows),
      original_image_tensor,
      std::to_string(index_frame) + "_" + std::to_string(0) + ".png",
      index_frame));
}

void lioOptimization::recordSinglePose(cloudFrame* p_frame) {
  std::ofstream foutC(std::string(output_path + "/pose.txt"), std::ios::app);

  foutC.setf(std::ios::scientific, std::ios::floatfield);
  foutC.precision(6);

  foutC << std::fixed << p_frame->time_sweep_end << " ";
  foutC << p_frame->p_state->translation.x() << " " << p_frame->p_state->translation.y() << " "
        << p_frame->p_state->translation.z() << " ";
  foutC << p_frame->p_state->rotation.x() << " " << p_frame->p_state->rotation.y() << " "
        << p_frame->p_state->rotation.z() << " " << p_frame->p_state->rotation.w();
  foutC << std::endl;

  foutC.close();

  if (initial_flag) {
    std::ofstream foutC2(std::string(output_path + "/velocity.txt"), std::ios::app);

    foutC2.setf(std::ios::scientific, std::ios::floatfield);
    foutC2.precision(6);

    foutC2 << std::fixed << p_frame->time_sweep_end << " ";
    foutC2 << p_frame->p_state->velocity.x() << " " << p_frame->p_state->velocity.y() << " "
           << p_frame->p_state->velocity.z();
    foutC2 << std::endl;

    foutC2.close();

    std::ofstream foutC3(std::string(output_path + "/bias.txt"), std::ios::app);

    foutC3.setf(std::ios::scientific, std::ios::floatfield);
    foutC3.precision(6);

    foutC3 << std::fixed << p_frame->time_sweep_end << " ";
    foutC3 << p_frame->p_state->ba.x() << " " << p_frame->p_state->ba.y() << " " << p_frame->p_state->ba.z() << " ";
    foutC3 << p_frame->p_state->bg.x() << " " << p_frame->p_state->bg.y() << " " << p_frame->p_state->bg.z();
    foutC3 << std::endl;

    foutC3.close();
  }
}

void lioOptimization::set_posestamp(geometry_msgs::PoseStamped& body_pose_out, cloudFrame* p_frame) {
  body_pose_out.pose.position.x = p_frame->p_state->translation.x();
  body_pose_out.pose.position.y = p_frame->p_state->translation.y();
  body_pose_out.pose.position.z = p_frame->p_state->translation.z();

  body_pose_out.pose.orientation.x = p_frame->p_state->rotation.x();
  body_pose_out.pose.orientation.y = p_frame->p_state->rotation.y();
  body_pose_out.pose.orientation.z = p_frame->p_state->rotation.z();
  body_pose_out.pose.orientation.w = p_frame->p_state->rotation.w();
}

void lioOptimization::publish_path(ros::Publisher pub_path, cloudFrame* p_frame) {
  set_posestamp(msg_body_pose, p_frame);
  msg_body_pose.header.stamp = ros::Time().fromSec(p_frame->time_sweep_end);
  msg_body_pose.header.frame_id = "camera_init";

  static int i = 0;
  i++;
  if (i % 20 == 0) {
    path.poses.push_back(msg_body_pose);
    pub_path.publish(path);
  }
}

void lioOptimization::publishCLoudWorld(
    ros::Publisher& pub_cloud_world,
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_points,
    cloudFrame* p_frame) {
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*pcl_points, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(p_frame->time_sweep_end);
  laserCloudmsg.header.frame_id = "camera_init";
  pub_cloud_world.publish(laserCloudmsg);
}

void lioOptimization::pubColorPoints(ros::Publisher& pub_cloud_rgb, cloudFrame* p_frame) {
  pcl::PointCloud<pcl::PointXYZRGB> points_rgb_vec;
  sensor_msgs::PointCloud2 ros_points_msg;

  // int num_publish = 0;

  for (int i = 0; i < img_pro->map_tracker->rgb_points_vec.size(); i++) {
    rgbPoint* p_point = img_pro->map_tracker->rgb_points_vec[i];

    if (p_point->N_rgb < map_options.pub_point_minimum_views)
      continue;

    pcl::PointXYZRGB rgb_point;

    rgb_point.x = p_point->getPosition()[0];
    rgb_point.y = p_point->getPosition()[1];
    rgb_point.z = p_point->getPosition()[2];
    rgb_point.r = p_point->getRgb()[2];
    rgb_point.g = p_point->getRgb()[1];
    rgb_point.b = p_point->getRgb()[0];

    points_rgb_vec.push_back(rgb_point);
  }

  pcl::toROSMsg(points_rgb_vec, ros_points_msg);

  ros_points_msg.header.stamp = ros::Time().fromSec(p_frame->time_sweep_end);
  ros_points_msg.header.frame_id = "camera_init";

  pub_cloud_rgb.publish(ros_points_msg);
}

void lioOptimization::threadAddColorPoints() {
  int last_pub_map_index = -1000;
  int sleep_time_after_pub = 10;

  while (1) {
    if (stop_thread) {
      std::cout << "Stop threadAddColorPoints thread.\n" << std::endl;
      break;
    }

    int points_size;
    {
      std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_rgb_points_vec);
      points_size = img_pro->map_tracker->rgb_points_vec.size();
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_rgb_vec(new pcl::PointCloud<pcl::PointXYZRGB>());
    int updated_frame_index;
    {
      std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_frame_index);
      updated_frame_index = img_pro->map_tracker->updated_frame_index;
    }

    if (last_pub_map_index == updated_frame_index) {
      continue;
    }

    last_pub_map_index = updated_frame_index;

    for (int i = 0; i < points_size - new_add_point_index; i++) {
      pcl::PointXYZRGB tmp_point;
      {
        std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_rgb_points_vec);
        tmp_point.x = img_pro->map_tracker->rgb_points_vec[new_add_point_index + i]->getPosition()[0];
        tmp_point.y = img_pro->map_tracker->rgb_points_vec[new_add_point_index + i]->getPosition()[1];
        tmp_point.z = img_pro->map_tracker->rgb_points_vec[new_add_point_index + i]->getPosition()[2];
        tmp_point.r = img_pro->map_tracker->rgb_points_vec[new_add_point_index + i]->getRgb()[2];
        tmp_point.g = img_pro->map_tracker->rgb_points_vec[new_add_point_index + i]->getRgb()[1];
        tmp_point.b = img_pro->map_tracker->rgb_points_vec[new_add_point_index + i]->getRgb()[0];
        points_rgb_vec->push_back(tmp_point);
      }
    }
    {
      std::lock_guard<std::mutex> lock(color_points_mutex);
      if (points_rgb_vec->size() > 0) {
        *color_points_world += *points_rgb_vec;
      }
    }

    new_add_point_index = points_size;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void lioOptimization::addPointToPcl(
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_points,
    rgbPoint& point,
    cloudFrame* p_frame) {
  pcl::PointXYZI cloudTemp;

  cloudTemp.x = point.getPosition().x();
  cloudTemp.y = point.getPosition().y();
  cloudTemp.z = point.getPosition().z();
  cloudTemp.intensity = 50 * (cloudTemp.z - p_frame->p_state->translation.z());
  pcl_points->points.push_back(cloudTemp);
}

cv::Mat lioOptimization::tensor2CvMat3X(torch::Tensor& tensor) {
  // torch.squeeze(input, dim=None, *, out=None) → Tensor
  // Returns a tensor with all the dimensions of input of size 1 removed.
  // tensor.detach
  // Returns a new Tensor, detached from the current graph.
  // permute dimension, 3x700x700 => 700x700x3
  tensor = tensor.detach().permute({1, 2, 0}).to(torch::kCPU);
  // float to 255 range
  tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);

  // shape of tensor
  int64_t height = tensor.size(0);
  int64_t width = tensor.size(1);

  // Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
  // so we must reshape tensor, otherwise we get a 3x3 grid
  tensor = tensor.reshape({width * height * 3});
  // CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
  cv::Mat imgbin(cv::Size(width, height), CV_8UC3, tensor.data_ptr());
  cv::Mat imgbinrgb;
  cv::cvtColor(imgbin, imgbinrgb, cv::COLOR_BGR2RGB);

  return imgbinrgb;
}

void lioOptimization::saveDepthMapAsNPY(torch::Tensor& tensor, const std::string& filename) {
  tensor = tensor.to(torch::kFloat32).detach().cpu();
  cv::Mat depthMap(tensor.size(0), tensor.size(1), CV_32F, tensor.data_ptr<float>());

  // Convert cv::Mat to std::vector<float>
  std::vector<float> depthData(depthMap.begin<float>(), depthMap.end<float>());

  // Save to npy file
  cnpy::npy_save(
      filename, &depthData[0], {static_cast<size_t>(depthMap.rows), static_cast<size_t>(depthMap.cols)}, "w");
}

cv::Mat lioOptimization::tensor2CvMat2X(torch::Tensor& tensor, float maxDepth) {
  tensor = tensor.to(torch::kFloat32).detach().cpu();
  tensor = tensor.squeeze(0);
  cv::Mat depthMap(tensor.size(0), tensor.size(1), CV_32F, tensor.data_ptr<float>());

  depthMap = depthMap * (255.0f / maxDepth);

  cv::Mat depthMap8U;
  depthMap.convertTo(depthMap8U, CV_8U);

  cv::Mat colorMap;
  cv::applyColorMap(depthMap8U, colorMap, cv::COLORMAP_JET);

  return colorMap;
}

std::string extractLastPathComponent(const std::string& path) {
  size_t last_slash = path.find_last_of('/');
  if (last_slash == std::string::npos) {
    throw std::runtime_error("Invalid path format");
  }

  std::string last_component = path.substr(last_slash + 1);

  size_t dot_pos = last_component.find_last_of('.');
  if (dot_pos != std::string::npos) {
    last_component = last_component.substr(0, dot_pos);
  }

  return last_component;
}

void lioOptimization::saveRender() {
  int count = 0;
  float psnr_value = 0;
  float ssim_value = 0;

  std::string save_dir = gsmodelParams.output_path.string() + "/" + extractLastPathComponent(save_bag_path);
  std::cout << "Results have been saved to " << save_dir << std::endl;

  gs::param::Write_model_parameters_to_file(gsmodelParams, save_dir);
  double duration = last_time_img - start_time_img;
  common::Timer::DumpIntoFile(_cameras.size(), duration, save_dir + "/log_time.txt");

  gaussian_pro->Save_ply(save_dir, 10086, false);

  std::vector<cv::Mat> frames;

  for (auto& cam : _cameras) {
    // Render
    auto gt_image = cam[0].Get_original_image().to(torch::kCUDA, true);
    auto [image, depth, depth_sol] = render(cam[0], gaussian_pro, background);

    auto _psnr = gaussian_splatting::psnr(image, gt_image);
    psnr_value += _psnr.item<float>();

    auto _ssim = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);
    ssim_value += _ssim.item<float>();

    auto render_image = tensor2CvMat3X(image);
    auto gt_imagexx = tensor2CvMat3X(gt_image);

    // 2 - dimensional
    auto depth_image = tensor2CvMat2X(depth);

    std::stringstream ss;
    // ss << "PSNR: " << _psnr.item<float>() << ", SSIM: " << _ssim.item<float>();
    // cv::putText(render_image, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

    cv::Mat mergedImage;
    cv::hconcat(render_image, gt_imagexx, mergedImage);

    if (count % 1 == 0) {
      cv::imwrite(save_dir + "/rendered_images/" + std::to_string(count) + ".png", mergedImage);
      cv::imwrite(save_dir + "/rendered_depths/" + std::to_string(count) + "_depth.png", depth_image);
    }
    frames.push_back(mergedImage);
    count++;
  }

  psnr_value /= count;
  ssim_value /= count;

  std::cout << "cameras size: " << _cameras.size() << ", mean psnr: " << psnr_value << ", mean ssim: " << ssim_value
            << std::endl;

  cv::Size frameSize = frames.front().size();
  cv::VideoWriter videoWriter;
  videoWriter.open(save_dir + "/rendered_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10, frameSize, true);

  for (const auto& frame : frames) {
    videoWriter << frame;
  }

  videoWriter.release();
}

void lioOptimization::saveColorPoints() {
  std::string pcd_path = std::string(output_path + "/rgb_map.pcd");
  std::cout << "Save colored points to " << pcd_path << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB> pcd_rgb;

  long point_size = img_pro->map_tracker->rgb_points_vec.size();
  pcd_rgb.resize(point_size);

  long point_count = 0;

  for (long i = point_size - 1; i > 0; i--) {
    int N_rgb;
    {
      std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_rgb_points_vec);
      N_rgb = img_pro->map_tracker->rgb_points_vec[i]->N_rgb;
    }

    if (N_rgb < map_options.pub_point_minimum_views) {
      continue;
    }

    pcl::PointXYZRGB point;
    {
      std::lock_guard<std::mutex> lock(*img_pro->map_tracker->mutex_rgb_points_vec);
      pcd_rgb.points[point_count].x = img_pro->map_tracker->rgb_points_vec[i]->getPosition()[0];
      pcd_rgb.points[point_count].y = img_pro->map_tracker->rgb_points_vec[i]->getPosition()[1];
      pcd_rgb.points[point_count].z = img_pro->map_tracker->rgb_points_vec[i]->getPosition()[2];
      pcd_rgb.points[point_count].r = img_pro->map_tracker->rgb_points_vec[i]->getRgb()[2];
      pcd_rgb.points[point_count].g = img_pro->map_tracker->rgb_points_vec[i]->getRgb()[1];
      pcd_rgb.points[point_count].b = img_pro->map_tracker->rgb_points_vec[i]->getRgb()[0];
    }
    point_count++;
  }

  pcd_rgb.resize(point_count);

  // std::cout << "Total have " << point_count << " points." << std::endl;
  // std::cout << "Now write to: " << pcd_path << std::endl;
  pcl::io::savePCDFileBinary(pcd_path, pcd_rgb);
}

void lioOptimization::run() {
  while (true) {
    if (stop_thread) {
      std::cout << "Stop run thread." << std::endl;
      break;
    }

    std::vector<Measurements> measurements = getMeasurements();

    for (auto& measurement : measurements) {
      // process
      double time_frame = measurement.time_image;
      double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

      if (!initial_flag) {
        for (auto& imu_msg : measurement.imu_measurements) {
          double time_imu = imu_msg->header.stamp.toSec();

          if (time_imu <= time_frame) {
            current_time = time_imu;
            dx = imu_msg->linear_acceleration.x;
            dy = imu_msg->linear_acceleration.y;
            dz = imu_msg->linear_acceleration.z;
            rx = imu_msg->angular_velocity.x;
            ry = imu_msg->angular_velocity.y;
            rz = imu_msg->angular_velocity.z;

            imu_meas.emplace_back(
                current_time, std::make_pair(Eigen::Vector3d(rx, ry, rz), Eigen::Vector3d(dx, dy, dz)));
          } else {
            double dt_1 = time_frame - current_time;
            double dt_2 = time_imu - time_frame;
            current_time = time_frame;
            assert(dt_1 >= 0);
            assert(dt_2 >= 0);
            assert(dt_1 + dt_2 > 0);
            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);
            dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
            rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
            ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
            rz = w1 * rz + w2 * imu_msg->angular_velocity.z;

            imu_meas.emplace_back(
                current_time, std::make_pair(Eigen::Vector3d(rx, ry, rz), Eigen::Vector3d(dx, dy, dz)));
          }
        }
        eskf_pro->tryInit(imu_meas);
        imu_meas.clear();

        last_time_frame = time_frame;

        std::vector<point3D>().swap(measurement.lidar_points);

        if (measurement.rendering) {
          measurement.image.release();
        }
        continue;
      }

      if (initial_flag) {
        imuState imu_state_temp;

        imu_state_temp.timestamp = current_time;

        imu_state_temp.un_acc =
            eskf_pro->getRotation().toRotationMatrix() * (eskf_pro->getLastAcc() - eskf_pro->getBa());
        imu_state_temp.un_gyr = eskf_pro->getLastGyr() - eskf_pro->getBg();
        imu_state_temp.trans = eskf_pro->getTranslation();
        imu_state_temp.quat = eskf_pro->getRotation();
        imu_state_temp.vel = eskf_pro->getVelocity();

        imu_states.push_back(imu_state_temp);
      }

      common::Timer::Evaluate(
          gp_options_.log_time,
          ros::Time::now().toSec(),
          [&]() {
            for (auto& imu_msg : measurement.imu_measurements) {
              double time_imu = imu_msg->header.stamp.toSec();

              if (time_imu <= time_frame) {
                double dt = time_imu - current_time;

                if (dt < -1e-6)
                  continue;
                assert(dt >= 0);
                current_time = time_imu;
                dx = imu_msg->linear_acceleration.x;
                dy = imu_msg->linear_acceleration.y;
                dz = imu_msg->linear_acceleration.z;
                rx = imu_msg->angular_velocity.x;
                ry = imu_msg->angular_velocity.y;
                rz = imu_msg->angular_velocity.z;

                imuState imu_state_temp;

                imu_state_temp.timestamp = current_time;

                imu_state_temp.un_acc =
                    eskf_pro->getRotation().toRotationMatrix() *
                    (0.5 * (eskf_pro->getLastAcc() + Eigen::Vector3d(dx, dy, dz)) - eskf_pro->getBa());
                imu_state_temp.un_gyr =
                    0.5 * (eskf_pro->getLastGyr() + Eigen::Vector3d(rx, ry, rz)) - eskf_pro->getBg();

                dt_sum = dt_sum + dt;
                eskf_pro->predict(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));

                imu_state_temp.trans = eskf_pro->getTranslation();
                imu_state_temp.quat = eskf_pro->getRotation();
                imu_state_temp.vel = eskf_pro->getVelocity();

                imu_states.push_back(imu_state_temp);
              } else {
                double dt_1 = time_frame - current_time;
                double dt_2 = time_imu - time_frame;
                current_time = time_frame;
                assert(dt_1 >= 0);
                assert(dt_2 >= 0);
                assert(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                rz = w1 * rz + w2 * imu_msg->angular_velocity.z;

                imuState imu_state_temp;

                imu_state_temp.timestamp = current_time;

                imu_state_temp.un_acc =
                    eskf_pro->getRotation().toRotationMatrix() *
                    (0.5 * (eskf_pro->getLastAcc() + Eigen::Vector3d(dx, dy, dz)) - eskf_pro->getBa());
                imu_state_temp.un_gyr =
                    0.5 * (eskf_pro->getLastGyr() + Eigen::Vector3d(rx, ry, rz)) - eskf_pro->getBg();

                dt_sum = dt_sum + dt_1;
                eskf_pro->predict(dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));

                imu_state_temp.trans = eskf_pro->getTranslation();
                imu_state_temp.quat = eskf_pro->getRotation();
                imu_state_temp.vel = eskf_pro->getVelocity();

                imu_states.push_back(imu_state_temp);
              }

              nav_msgs::Odometry odomAftMapped;
              odomAftMapped.header.frame_id = "camera_init";
              odomAftMapped.child_frame_id = "body";
              odomAftMapped.header.stamp = imu_msg->header.stamp;
              Eigen::Vector3d p_predict = eskf_pro->getTranslation();
              Eigen::Quaterniond q_predict = eskf_pro->getRotation();
              odomAftMapped.pose.pose.orientation.x = q_predict.x();
              odomAftMapped.pose.pose.orientation.y = q_predict.y();
              odomAftMapped.pose.pose.orientation.z = q_predict.z();
              odomAftMapped.pose.pose.orientation.w = q_predict.w();
              odomAftMapped.pose.pose.position.x = p_predict.x();
              odomAftMapped.pose.pose.position.y = p_predict.y();
              odomAftMapped.pose.pose.position.z = p_predict.z();
              pub_odom.publish(odomAftMapped);
            }
          },
          "stateSLAM");

      process(
          measurement.lidar_points,
          measurement.time_sweep.first,
          measurement.time_sweep.second,
          measurement.image,
          measurement.rendering);

      imu_states.clear();

      last_time_frame = time_frame;
      index_frame++;

      std::vector<point3D>().swap(measurement.lidar_points);

      if (measurement.rendering) {
        measurement.image.release();
      }
    }
  }
}

int main(int argc, char** argv) {

  ros::init(argc, argv, "livo_node");

  std::shared_ptr<lioOptimization> lio;
  lio.reset(new lioOptimization());

#ifdef _OPENMP
  int NUM_THREADS = omp_get_num_procs() / 4;
  omp_set_num_threads(NUM_THREADS);
  std::cout << "OpenMP available. Number of threads: " << omp_get_max_threads() << std::endl;
#else
  puts("OpenMp unavailable");
  exit(-1);
#endif

  std::thread add_color_pointcloud(&lioOptimization::threadAddColorPoints, lio);
  std::thread optimizevis(&lioOptimization::optimize_vis, lio);
  std::thread odom_low(&lioOptimization::run, lio);

  ros::MultiThreadedSpinner spinner(6);
  spinner.spin();

  lio->stop_thread = true;

  lio->saveRender();
  lio->saveColorPoints();

  sleep(1);
  return 0;
}
