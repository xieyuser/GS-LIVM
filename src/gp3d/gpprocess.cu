
#include "cuda_runtime.h"
#include "gp3d/auxiliary.h"
#include "gp3d/gpprocess.cuh"

// ALG.meshgrid
__device__ void
deviceEvenSetLinSpaced(int num_gp_side, float min_x, float min_y, float interval, float* test, bool fullcover) {
  for (int i = 0; i < num_gp_side * num_gp_side; i++) {
    test[i * 2] = min_x + interval * (fullcover ? (i / num_gp_side) : ((i / num_gp_side) + 0.5f));
    test[i * 2 + 1] = min_y + interval * (fullcover ? (i % num_gp_side) : ((i % num_gp_side) + 0.5f));
  }
}

// computeKernel for every voxel: K_\alpha = \mathcal{K}(x_\alpha, x_\alpha)
__device__ void computeKernelMatrices(
    float* d_ky,
    float* k_starm,
    float* d_train_x,
    float* d_train_y,
    float* d_points_testm,
    int num_train,
    int num_gp_side_square,
    float kernel_size,
    float* variance_sensor) {

  for (int i = 0; i < num_train; i++) {
    for (int j = i; j < num_train; j++) {
      float distance = sqrtf(powf((d_train_x[j] - d_train_x[i]), 2) + powf((d_train_y[j] - d_train_y[i]), 2));
      d_ky[i * num_train + j] = expf(-kernel_size * distance);
      if (i != j) {
        d_ky[j * num_train + i] = d_ky[i * num_train + j];
      } else {
        d_ky[i * num_train + j] += variance_sensor[i] * variance_sensor[i];
      }
    }
  }

  for (int i = 0; i < num_gp_side_square; i++) {
    for (int j = 0; j < num_train; j++) {
      float distance =
          sqrtf(powf((d_train_x[j] - d_points_testm[i * 2]), 2) + powf((d_train_y[j] - d_points_testm[i * 2 + 1]), 2));
      k_starm[i * num_train + j] = expf(-kernel_size * distance);
    }
  }
}

// calculate mean value
__device__ float calculateMean(float* data, int num_elements) {
  float sum = 0.0f;
  for (int i = 0; i < num_elements; i++) {
    sum += data[i];
  }
  float mean = sum / num_elements;

  for (int i = 0; i < num_elements; i++) {
    data[i] -= mean;
  }
  return mean;
}

// calculate variance for next 3D GS fast initialization
__global__ void processVoxelsVarianceKernel(
    DVoxel* d_voxels,
    float* output,
    int num_voxels,
    int num_gp_side_square,
    int* error_code,
    float max_var_mean) {
  int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (voxel_index < num_voxels) {
    float tmp_total = 0.0f;

    for (int ip = 0; ip < num_gp_side_square; ip++) {
      tmp_total += d_voxels[voxel_index].k_variance[ip * num_gp_side_square + ip];
    }

    float var_mean = 1.0f - tmp_total / num_gp_side_square;
    if (var_mean > 1.0f || var_mean < 0.0f) {
      error_code[voxel_index] = -404;
      return;
    }

    if (var_mean > max_var_mean) {
      int num_train = d_voxels[voxel_index].num_train;
      int direction = d_voxels[voxel_index].direction;

      for (int count = 0; count < num_train; count++) {
        int x_index, y_index;

        switch (direction) {
          case 0:
            x_index = (d_voxels[voxel_index].points[count].y - d_voxels[voxel_index].region.y_min) /
                      d_voxels[voxel_index].interval;
            y_index = (d_voxels[voxel_index].points[count].z - d_voxels[voxel_index].region.z_min) /
                      d_voxels[voxel_index].interval;
            break;
          case 1:
            x_index = (d_voxels[voxel_index].points[count].z - d_voxels[voxel_index].region.z_min) /
                      d_voxels[voxel_index].interval;
            y_index = (d_voxels[voxel_index].points[count].x - d_voxels[voxel_index].region.x_min) /
                      d_voxels[voxel_index].interval;
            break;
          case 2:
            x_index = (d_voxels[voxel_index].points[count].x - d_voxels[voxel_index].region.x_min) /
                      d_voxels[voxel_index].interval;
            y_index = (d_voxels[voxel_index].points[count].y - d_voxels[voxel_index].region.y_min) /
                      d_voxels[voxel_index].interval;
            break;
        }

        float vari = 1.0f - d_voxels[voxel_index].k_variance[x_index * num_gp_side_square + y_index];
        output[voxel_index * num_train + count] = vari;
      }
      error_code[voxel_index] = -1;
    } else {
      error_code[voxel_index] = 0;
      return;
    }
  }
}

// main function
__global__ void processVoxelsKernel(
    DVoxel* d_voxels,
    int num_gp_side,
    int num_gp_side_square,
    int num_voxels,
    bool full_cover,
    float kernel_size) {
  // Determine the voxel_index of the thread
  int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure we do not go out of bounds
  if (voxel_index >= num_voxels) {
    return;
  }

  // get train points
  for (int i = 0; i < d_voxels[voxel_index].num_train; i++) {
    switch (d_voxels[voxel_index].direction) {
      case 0:
        d_voxels[voxel_index].global_train_x_points[i] = d_voxels[voxel_index].points[i].y;
        d_voxels[voxel_index].global_train_y_points[i] = d_voxels[voxel_index].points[i].z;
        d_voxels[voxel_index].global_train_f_points[i] = d_voxels[voxel_index].points[i].x;
        break;
      case 1:
        d_voxels[voxel_index].global_train_x_points[i] = d_voxels[voxel_index].points[i].z;
        d_voxels[voxel_index].global_train_y_points[i] = d_voxels[voxel_index].points[i].x;
        d_voxels[voxel_index].global_train_f_points[i] = d_voxels[voxel_index].points[i].y;
        break;
      case 2:
        d_voxels[voxel_index].global_train_x_points[i] = d_voxels[voxel_index].points[i].x;
        d_voxels[voxel_index].global_train_y_points[i] = d_voxels[voxel_index].points[i].y;
        d_voxels[voxel_index].global_train_f_points[i] = d_voxels[voxel_index].points[i].z;
        break;
    }
  }

  // calc mean f
  d_voxels[voxel_index].mean =
      calculateMean(d_voxels[voxel_index].global_train_f_points, d_voxels[voxel_index].num_train);

  // even test point coords
  switch (d_voxels[voxel_index].direction) {
    case 0: {
      deviceEvenSetLinSpaced(
          num_gp_side,
          d_voxels[voxel_index].region.y_min,
          d_voxels[voxel_index].region.z_min,
          d_voxels[voxel_index].interval,
          d_voxels[voxel_index].global_points_testm,
          full_cover);
      break;
    }
    case 1: {
      deviceEvenSetLinSpaced(
          num_gp_side,
          d_voxels[voxel_index].region.z_min,
          d_voxels[voxel_index].region.x_min,
          d_voxels[voxel_index].interval,
          d_voxels[voxel_index].global_points_testm,
          full_cover);
      break;
    }
    case 2: {
      deviceEvenSetLinSpaced(
          num_gp_side,
          d_voxels[voxel_index].region.x_min,
          d_voxels[voxel_index].region.y_min,
          d_voxels[voxel_index].interval,
          d_voxels[voxel_index].global_points_testm,
          full_cover);
      break;
    }
  }

  computeKernelMatrices(
      d_voxels[voxel_index].ky,
      d_voxels[voxel_index].k_starm,
      d_voxels[voxel_index].global_train_x_points,
      d_voxels[voxel_index].global_train_y_points,
      d_voxels[voxel_index].global_points_testm,
      d_voxels[voxel_index].num_train,
      num_gp_side_square,
      kernel_size,
      d_voxels[voxel_index].global_train_points_variance);
}

// allocate host data and gpt memory
void allocateHostDataGP3D(
    DVoxel*& h_voxels,
    std::vector<GSLIVM::needGPUdata>& data,
    int num_gp_side,
    int num_gp_side_square,
    bool fullcover,
    int num_train) {
  int num_voxels = data.size();
  h_voxels = (DVoxel*)malloc(num_voxels * sizeof(DVoxel));

#pragma omp parallel for
  for (int voxel_index = 0; voxel_index < num_voxels; voxel_index++) {
    // main data
    auto& value = data[voxel_index];

    // region
    DRegion region{
        .x_min = static_cast<float>(value.region_.x_min),
        .x_max = static_cast<float>(value.region_.x_max),
        .y_min = static_cast<float>(value.region_.y_min),
        .y_max = static_cast<float>(value.region_.y_max),
        .z_min = static_cast<float>(value.region_.z_min),
        .z_max = static_cast<float>(value.region_.z_max)};

    h_voxels[voxel_index].region = region;
    // hash position
    h_voxels[voxel_index].hash_posi = value.hash_;
    // direction
    h_voxels[voxel_index].direction = value.direction_;
    // interval rectangle same interval
    h_voxels[voxel_index].interval =
        (value.region_.z_max - value.region_.z_min) / (fullcover ? (num_gp_side - 1) : num_gp_side);

    // points
    Eigen::Matrix<float, 3, Eigen::Dynamic> points_matrix = value.point.cast<float>();
    Eigen::Matrix<float, 1, Eigen::Dynamic> points_variance = value.variance.cast<float>();

    // num_train
    int num_points = static_cast<int>(value.num_point);
    h_voxels[voxel_index].num_train = num_train;

    h_voxels[voxel_index].points = (DPoint*)malloc(num_train * sizeof(DPoint));

    for (int tmp_i = 0; tmp_i < num_train; tmp_i++) {
      DPoint tmp_point;
      tmp_point.x = points_matrix(0, tmp_i + num_points - num_train);
      tmp_point.y = points_matrix(1, tmp_i + num_points - num_train);
      tmp_point.z = points_matrix(2, tmp_i + num_points - num_train);
      h_voxels[voxel_index].points[tmp_i] = tmp_point;
    }

    h_voxels[voxel_index].global_train_points_variance = new float[num_train];
    memset(h_voxels[voxel_index].global_train_points_variance, 0, num_train * sizeof(float));

    for (int i = 0; i < num_train; i++) {
      // std::cout << points_variance(i) << " ";
      h_voxels[voxel_index].global_train_points_variance[i] = points_variance(i);
    }
    // std::cout << std::endl;

    // for kernel function
    h_voxels[voxel_index].global_train_x_points = new float[num_train];
    h_voxels[voxel_index].global_train_y_points = new float[num_train];
    h_voxels[voxel_index].global_train_f_points = new float[num_train];
    h_voxels[voxel_index].global_points_testm = new float[2 * num_gp_side_square];

    memset(h_voxels[voxel_index].global_train_x_points, 0, num_train * sizeof(float));
    memset(h_voxels[voxel_index].global_train_y_points, 0, num_train * sizeof(float));
    memset(h_voxels[voxel_index].global_train_f_points, 0, num_train * sizeof(float));
    memset(h_voxels[voxel_index].global_points_testm, 0, 2 * num_gp_side_square * sizeof(float));

    // start K
    h_voxels[voxel_index].ky = new float[num_train * num_train];
    h_voxels[voxel_index].k_starm = new float[num_gp_side_square * num_train];
    h_voxels[voxel_index].kky = new float[num_gp_side_square * num_train];
    h_voxels[voxel_index].f_starm = new float[num_gp_side_square];
    h_voxels[voxel_index].k_variance = new float[num_gp_side_square * num_gp_side_square];

    memset(h_voxels[voxel_index].ky, 0, num_train * num_train * sizeof(float));
    memset(h_voxels[voxel_index].k_starm, 0, num_gp_side_square * num_train * sizeof(float));
    memset(h_voxels[voxel_index].kky, 0, num_gp_side_square * num_train * sizeof(float));
    memset(h_voxels[voxel_index].f_starm, 0, num_gp_side_square * sizeof(float));
    memset(h_voxels[voxel_index].k_variance, 0, num_gp_side_square * num_gp_side_square * sizeof(float));
  }
}

// copy host data to gpu
void gpProcess::copyDataArrayToGPU(DVoxel* h_voxels, DVoxel*& d_voxels, int num_voxels, int num_gp_side_square) {
  all_d_data.clear();
  size_t total_points_size = 0;
  size_t total_global_train_points_size = 0;
  size_t total_global_points_testm_size = num_voxels * 2 * num_gp_side_square * sizeof(float);
  size_t total_ky_size = 0;
  size_t total_k_starm_size = 0;
  size_t total_kky_size = 0;
  size_t total_f_starm_size = num_voxels * num_gp_side_square * sizeof(float);
  size_t total_k_variance_size = num_voxels * num_gp_side_square * num_gp_side_square * sizeof(float);
  for (int i = 0; i < num_voxels; i++) {
    int num_train = h_voxels[i].num_train;
    total_points_size += num_train * sizeof(DPoint);
    total_global_train_points_size += num_train * 4 * sizeof(float);  // x, y, f, variance
    total_ky_size += num_train * num_train * sizeof(float);
    total_k_starm_size += num_gp_side_square * num_train * sizeof(float);
    total_kky_size += num_gp_side_square * num_train * sizeof(float);
  }
  char* d_total_memory = nullptr;
  cudaMalloc(
      &d_total_memory,
      total_points_size + total_global_train_points_size + total_global_points_testm_size + total_ky_size +
          total_k_starm_size + total_kky_size + total_f_starm_size + total_k_variance_size);

  char* d_current_ptr = d_total_memory;
  DVoxel* d_tmp_voxels = new DVoxel[num_voxels];
  for (int i = 0; i < num_voxels; i++) {
    DVoxel h_data = h_voxels[i];
    DVoxel d_data;
    int num_train = h_data.num_train;
    d_data.num_train = h_data.num_train;
    d_data.direction = h_data.direction;
    d_data.hash_posi = h_data.hash_posi;
    d_data.region = h_data.region;
    d_data.interval = h_data.interval;
    d_data.points = reinterpret_cast<DPoint*>(d_current_ptr);
    cudaMemcpyAsync(d_data.points, h_data.points, num_train * sizeof(DPoint), cudaMemcpyHostToDevice);
    d_current_ptr += num_train * sizeof(DPoint);
    d_data.global_train_x_points = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.global_train_x_points, h_data.global_train_x_points, num_train * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_train * sizeof(float);
    d_data.global_train_y_points = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.global_train_y_points, h_data.global_train_y_points, num_train * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_train * sizeof(float);
    d_data.global_train_f_points = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.global_train_f_points, h_data.global_train_f_points, num_train * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_train * sizeof(float);
    d_data.global_train_points_variance = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.global_train_points_variance,
        h_data.global_train_points_variance,
        num_train * sizeof(float),
        cudaMemcpyHostToDevice);
    d_current_ptr += num_train * sizeof(float);
    d_data.global_points_testm = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.global_points_testm,
        h_data.global_points_testm,
        2 * num_gp_side_square * sizeof(float),
        cudaMemcpyHostToDevice);
    d_current_ptr += 2 * num_gp_side_square * sizeof(float);
    d_data.ky = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(d_data.ky, h_data.ky, num_train * num_train * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_train * num_train * sizeof(float);
    d_data.k_starm = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.k_starm, h_data.k_starm, num_gp_side_square * num_train * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_gp_side_square * num_train * sizeof(float);
    d_data.kky = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(d_data.kky, h_data.kky, num_gp_side_square * num_train * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_gp_side_square * num_train * sizeof(float);
    d_data.f_starm = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(d_data.f_starm, h_data.f_starm, num_gp_side_square * sizeof(float), cudaMemcpyHostToDevice);
    d_current_ptr += num_gp_side_square * sizeof(float);
    d_data.k_variance = reinterpret_cast<float*>(d_current_ptr);
    cudaMemcpyAsync(
        d_data.k_variance,
        h_data.k_variance,
        num_gp_side_square * num_gp_side_square * sizeof(float),
        cudaMemcpyHostToDevice);
    d_current_ptr += num_gp_side_square * num_gp_side_square * sizeof(float);
    d_tmp_voxels[i] = d_data;
    all_d_data.push_back(d_data);

    delete[] h_voxels[i].global_train_x_points;
    delete[] h_voxels[i].global_train_y_points;
    delete[] h_voxels[i].global_train_f_points;
    delete[] h_voxels[i].global_train_points_variance;
    delete[] h_voxels[i].ky;
    delete[] h_voxels[i].k_starm;
    delete[] h_voxels[i].kky;
  }
  cudaMemcpy(d_voxels, d_tmp_voxels, num_voxels * sizeof(DVoxel), cudaMemcpyHostToDevice);
  delete[] d_tmp_voxels;
}

// allocate pointers
__global__ void allocatePointerArrays(
    DVoxel* d_voxels,
    float** d_ky_array,
    float** d_k_starm_array,
    float** d_kky_array,
    float** d_f_starm_array,
    float** d_k_variance_array,
    float** d_global_train_f_points_array,
    int num_voxels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_voxels) {
    d_ky_array[i] = d_voxels[i].ky;
    d_k_starm_array[i] = d_voxels[i].k_starm;
    d_kky_array[i] = d_voxels[i].kky;
    d_f_starm_array[i] = d_voxels[i].f_starm;
    d_k_variance_array[i] = d_voxels[i].k_variance;
    d_global_train_f_points_array[i] = d_voxels[i].global_train_f_points;
  }
}

// fast initialization of 3DGS
std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Matrix3f>>
fastInitial3DGS(const std::vector<DPoint>& points, int num_gp_side, int neighbour_size) {
  std::vector<Eigen::Vector3f> means;
  int grid_size = num_gp_side / neighbour_size;
  std::vector<Eigen::Matrix3f> covariance_matrices;

  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      std::vector<DPoint> block_points;
      for (int di = 0; di < neighbour_size; ++di) {
        for (int dj = 0; dj < neighbour_size; ++dj) {
          int index = (i * neighbour_size + di) * num_gp_side + (j * neighbour_size + dj);
          block_points.push_back(points[index]);
        }
      }

      Eigen::MatrixXf coordinates(block_points.size(), 3);
      Eigen::VectorXf weights(block_points.size());
      for (int k = 0; k < block_points.size(); ++k) {
        coordinates(k, 0) = block_points[k].x;
        coordinates(k, 1) = block_points[k].y;
        coordinates(k, 2) = block_points[k].z;
        weights(k) = 1.0f / block_points[k].variance;
      }

      Eigen::Vector3f weighted_mean = (coordinates.array().colwise() * weights.array()).colwise().sum() / weights.sum();

      Eigen::MatrixXf centered = coordinates.rowwise() - weighted_mean.transpose();

      Eigen::Matrix3f weighted_covariance =
          (centered.array().colwise() * weights.array()).matrix().transpose() * centered / weights.sum();

      covariance_matrices.push_back(weighted_covariance);
      means.push_back(weighted_mean);
    }
  }

  return std::make_pair(means, covariance_matrices);
}

// Kernel function
__global__ void calculateDPointsKernel(
    float* output,           // Output array: shape [num_voxels * num_gp_side_square * 4]
    const DVoxel* d_voxels,  // Input voxel array
    int num_voxels,          // Number of voxels
    int num_gp_side_square)  // Square of the side length (number of points per voxel)
{
  int voxel_index = blockIdx.x;
  int point_index = threadIdx.x;

  if (voxel_index < num_voxels && point_index < num_gp_side_square) {
    int output_index = voxel_index * num_gp_side_square * 4 + point_index * 4;

    float x, y, z;

    switch (d_voxels[voxel_index].direction) {
      case 0:
        x = d_voxels[voxel_index].f_starm[point_index] + d_voxels[voxel_index].mean;
        y = d_voxels[voxel_index].global_points_testm[point_index * 2];
        z = d_voxels[voxel_index].global_points_testm[point_index * 2 + 1];
        break;
      case 1:
        x = d_voxels[voxel_index].global_points_testm[point_index * 2 + 1];
        y = d_voxels[voxel_index].f_starm[point_index] + d_voxels[voxel_index].mean;
        z = d_voxels[voxel_index].global_points_testm[point_index * 2];
        break;
      case 2:
        x = d_voxels[voxel_index].global_points_testm[point_index * 2];
        y = d_voxels[voxel_index].global_points_testm[point_index * 2 + 1];
        z = d_voxels[voxel_index].f_starm[point_index] + d_voxels[voxel_index].mean;
        break;
    }

    float variance = d_voxels[voxel_index].k_variance[point_index * num_gp_side_square + point_index];

    // Store the values in the output array
    output[output_index] = x;
    output[output_index + 1] = y;
    output[output_index + 2] = z;
    output[output_index + 3] = variance;
  }
}

// main function
void gpProcess::forward_gp3d(
    std::vector<GSLIVM::needGPUdata>& data,
    std::vector<GSLIVM::varianceUpdate>& updateVas,
    GSLIVM::ImageRt& frame,
    std::vector<GSLIVM::GsForMap>& final_gs_sample,
    std::vector<GSLIVM::GsForLoss>& final_gs_calc_loss) {
  int num_voxels = data.size();

  DVoxel* h_voxels;
  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        allocateHostDataGP3D(
            h_voxels, data, num_gp_side, num_gp_side_square, gp_options_.full_cover, gp_options_.min_points_num_to_gp);
      },
      "allocateHostDataGP3D");

  DVoxel* d_voxels;
  cudaMalloc(&d_voxels, num_voxels * sizeof(DVoxel));

  // copy data to gpu
  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() { copyDataArrayToGPU(h_voxels, d_voxels, num_voxels, num_gp_side_square); },
      "copyDataArrayToGPU");

  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        int num_blocks = (num_voxels + 255) / 256;
        processVoxelsKernel<<<num_blocks, 256>>>(
            d_voxels, num_gp_side, num_gp_side_square, num_voxels, gp_options_.full_cover, gp_options_.kernel_size);
      },
      "processGPVoxelsKernel");

  float* d_global_testm;
  cudaMalloc(&d_global_testm, num_voxels * num_gp_side_square * 4 * sizeof(float));

  float* d_update_varis;
  int* d_error_codes;

  cudaMalloc((void**)&d_update_varis, num_voxels * gp_options_.min_points_num_to_gp * sizeof(float));
  cudaMalloc((void**)&d_error_codes, num_voxels * sizeof(int));

  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        cublasHandle_t cublasHandle = nullptr;
        cublasCreate(&cublasHandle);

        cusolverDnHandle_t cusolverHandle = nullptr;
        cusolverDnCreate(&cusolverHandle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Allocate device memory for pointer arrays
        float** d_ky_array;
        float** d_k_starm_array;
        float** d_kky_array;

        float** d_f_starm_array;
        float** d_k_variance_array;

        float** d_global_train_f_points_array;

        cudaMalloc(&d_ky_array, num_voxels * sizeof(float*));
        cudaMalloc(&d_k_starm_array, num_voxels * sizeof(float*));
        cudaMalloc(&d_kky_array, num_voxels * sizeof(float*));
        cudaMalloc(&d_f_starm_array, num_voxels * sizeof(float*));
        cudaMalloc(&d_k_variance_array, num_voxels * sizeof(float*));
        cudaMalloc(&d_global_train_f_points_array, num_voxels * sizeof(float*));

        // Launch kernel to initialize pointer arrays
        int blockSize = 256;
        int numBlocks = (num_voxels + blockSize - 1) / blockSize;
        allocatePointerArrays<<<numBlocks, blockSize>>>(
            d_voxels,
            d_ky_array,
            d_k_starm_array,
            d_kky_array,
            d_f_starm_array,
            d_k_variance_array,
            d_global_train_f_points_array,
            num_voxels);

        // Prepare info array for batched Cholesky decomposition

        int NUM_TRAIN = gp_options_.min_points_num_to_gp;

        int *d_pivotArray, *d_infoArray;
        cudaMalloc(&d_pivotArray, NUM_TRAIN * num_voxels * sizeof(int));
        cudaMalloc(&d_infoArray, num_voxels * sizeof(int));

        cublasSgetrfBatched(cublasHandle, NUM_TRAIN, d_ky_array, NUM_TRAIN, d_pivotArray, d_infoArray, num_voxels);

        cublasSgetriBatched(
            cublasHandle,
            NUM_TRAIN,
            (const float**)d_ky_array,
            NUM_TRAIN,
            d_pivotArray,
            d_ky_array,
            NUM_TRAIN,
            d_infoArray,
            num_voxels);

        cudaFree(d_pivotArray);
        cudaFree(d_infoArray);

        // Batched matrix multiplication operations
        cublasSgemmBatched(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            NUM_TRAIN,
            num_gp_side_square,
            NUM_TRAIN,
            &alpha,
            (const float**)d_ky_array,
            NUM_TRAIN,
            (const float**)d_k_starm_array,
            NUM_TRAIN,
            &beta,
            d_kky_array,
            NUM_TRAIN,
            num_voxels);

        cublasSgemmBatched(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            1,
            num_gp_side_square,
            NUM_TRAIN,
            &alpha,
            (const float**)d_global_train_f_points_array,
            1,
            (const float**)d_kky_array,
            NUM_TRAIN,
            &beta,
            d_f_starm_array,
            1,
            num_voxels);

        cublasSgemmBatched(
            cublasHandle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            num_gp_side_square,
            num_gp_side_square,
            NUM_TRAIN,
            &alpha,
            (const float**)d_k_starm_array,
            NUM_TRAIN,
            (const float**)d_kky_array,
            NUM_TRAIN,
            &beta,
            d_k_variance_array,
            num_gp_side_square,
            num_voxels);

        // Free allocated memory
        cudaFree(d_ky_array);
        cudaFree(d_k_starm_array);
        cudaFree(d_kky_array);
        cudaFree(d_f_starm_array);
        cudaFree(d_k_variance_array);
        cudaFree(d_global_train_f_points_array);

        cublasDestroy(cublasHandle);
        cusolverDnDestroy(cusolverHandle);

        // result processing
        // Define grid and block sizes
        dim3 num_grid(num_voxels);
        dim3 num_block(num_gp_side_square);

        // Launch the kernel
        calculateDPointsKernel<<<num_grid, num_block>>>(d_global_testm, d_voxels, num_voxels, num_gp_side_square);

        processVoxelsVarianceKernel<<<numBlocks, blockSize>>>(
            d_voxels, d_update_varis, num_voxels, num_gp_side_square, d_error_codes, gp_options_.max_var_mean);
      },
      "gpStream");

  // process data for gmm
  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        float* h_global_testm = new float[num_voxels * num_gp_side_square * 4];
        float* h_update_varis = new float[num_voxels * gp_options_.min_points_num_to_gp];
        int* h_error_codes = new int[num_voxels];

        {
          CHECK_CUDA(
              cudaMemcpy(
                  h_global_testm,
                  d_global_testm,
                  num_voxels * num_gp_side_square * 4 * sizeof(float),
                  cudaMemcpyDeviceToHost),
              gp_options_.debug);

          CHECK_CUDA(
              cudaMemcpy(
                  h_update_varis,
                  d_update_varis,
                  num_voxels * gp_options_.min_points_num_to_gp * sizeof(float),
                  cudaMemcpyDeviceToHost),
              gp_options_.debug);

          CHECK_CUDA(
              cudaMemcpy(h_error_codes, d_error_codes, num_voxels * sizeof(int), cudaMemcpyDeviceToHost),
              gp_options_.debug);

          CHECK_CUDA(cudaFree(d_global_testm), gp_options_.debug);
          CHECK_CUDA(cudaFree(d_update_varis), gp_options_.debug);
          CHECK_CUDA(cudaFree(d_error_codes), gp_options_.debug);
        }

#pragma omp parallel for
        for (int voxel_index = 0; voxel_index < num_voxels; voxel_index++) {
          if (h_error_codes[voxel_index] == -404) {
            std::cout << " <<< ERROR in variance calculation..." << std::endl;
            exit(-1);
          }

          if (h_error_codes[voxel_index] == -1) {
            // update parameters
            GSLIVM::varianceUpdate ud_vari;
            ud_vari.num_point = h_voxels[voxel_index].num_train;
            ud_vari.hash_ = h_voxels[voxel_index].hash_posi;

            for (int count = 0; count < h_voxels[voxel_index].num_train; count++) {
              ud_vari.update_variance.push_back(
                  0.2 * h_update_varis[voxel_index * gp_options_.min_points_num_to_gp + count]);
            }

            {
              std::lock_guard<std::mutex> lock(gp_update_mutex);
              updateVas.push_back(ud_vari);
            }
          }

          std::vector<DPoint> ori_points;
          torch::Tensor ori_points_tensor = torch::empty({0, 3}, torch::kFloat32);

          {
            std::vector<DPoint> temp_points;
            int vvindex = voxel_index * num_gp_side_square * 4;
            for (int i = 0; i < num_gp_side_square; i++) {
              DPoint tmp_tensor;
              tmp_tensor.direction = h_voxels[voxel_index].direction;
              int index = vvindex + i * 4;
              tmp_tensor.x = h_global_testm[index + 0];
              tmp_tensor.y = h_global_testm[index + 1];
              tmp_tensor.z = h_global_testm[index + 2];
              tmp_tensor.variance = h_global_testm[index + 3];

              ori_points.push_back(tmp_tensor);
              temp_points.push_back(tmp_tensor);
            }

            std::vector<torch::Tensor> temp_tensors;

            for (const auto& point : temp_points) {
              torch::Tensor tmp_point = torch::tensor({point.x, point.y, point.z}, torch::kFloat32);
              temp_tensors.push_back(tmp_point);
            }

            torch::Tensor all_points_tensor = torch::stack(temp_tensors);

            if (h_error_codes[voxel_index] == -1) {
              std::vector<torch::Tensor> selected_points;
              for (int i = 0; i < temp_tensors.size(); i += 30) {
                selected_points.push_back(temp_tensors[i]);
              }
              ori_points_tensor = torch::stack(selected_points);
            } else {
              ori_points_tensor = all_points_tensor;
            }
          }

          if (h_error_codes[voxel_index] == -1) {
            GSLIVM::GsForLoss _tmp_gs_loss;
            _tmp_gs_loss.hash_posi_ = h_voxels[voxel_index].hash_posi;
            _tmp_gs_loss.gs_xyz = ori_points_tensor;
            {
              std::lock_guard<std::mutex> lock(final_gs_loss_mutex);
              final_gs_calc_loss.push_back(_tmp_gs_loss);
            }
          }

          {
            std::lock_guard<std::mutex> lock(mutex_added_final_gs_sample_insert);
            if (added_final_gs_sample.find(h_voxels[voxel_index].hash_posi) == added_final_gs_sample.end()) {
              added_final_gs_sample.insert(h_voxels[voxel_index].hash_posi);
            } else {
              continue;
            }
          }

          std::vector<DColor> colors;
          std::vector<Eigen::Vector3f> points;
          std::vector<Eigen::Matrix3f> covs;

          {
            // for map
            std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::Matrix3f>> means_covs =
                fastInitial3DGS(ori_points, num_gp_side, gp_options_.neighbour_size);

            points = means_covs.first;
            covs = means_covs.second;

            getColors(points, colors, frame);
          }

          torch::Tensor pc_tensor = torch::empty({0, 3}, torch::kFloat32);
          torch::Tensor pc_color = torch::empty({0, 3}, torch::kUInt8);
          torch::Tensor pc_variance = torch::empty({0, 3, 3}, torch::kFloat32);
          {
            std::vector<torch::Tensor> temp_points;
            std::vector<torch::Tensor> temp_colors;
            std::vector<torch::Tensor> temp_variances;

            for (int index = 0; index < colors.size(); index++) {
              if (colors[index].r + colors[index].g + colors[index].b == -3.0f) {
                continue;
              }

              torch::Tensor single_point =
                  torch::tensor({points[index].x(), points[index].y(), points[index].z()}, torch::kFloat32);
              torch::Tensor single_color = torch::tensor(
                  {{static_cast<uint8_t>(colors[index].r),
                    static_cast<uint8_t>(colors[index].g),
                    static_cast<uint8_t>(colors[index].b)}},
                  torch::kUInt8);
              torch::Tensor single_variance = torch::zeros({3, 3}, torch::kFloat32);

              for (int tmp_i = 0; tmp_i < 3; tmp_i++) {
                for (int tmp_j = 0; tmp_j < 3; tmp_j++) {
                  single_variance[tmp_i][tmp_j] = covs[index](tmp_i, tmp_j);
                }
              }

              temp_points.push_back(single_point);
              temp_colors.push_back(single_color);
              temp_variances.push_back(single_variance);
            }

            if (temp_points.size() > 0) {
              pc_tensor = torch::stack(temp_points).squeeze(1);
              pc_color = torch::stack(temp_colors).squeeze(1);
              pc_variance = torch::stack(temp_variances).squeeze(1);
            }
          }

          if (pc_tensor.size(0) > 0) {
            GSLIVM::GsForMap _tmp_gs_sample;
            _tmp_gs_sample.hash_posi_ = h_voxels[voxel_index].hash_posi;
            _tmp_gs_sample.gs_xyz = pc_tensor;
            _tmp_gs_sample.gs_rgb = pc_color;
            _tmp_gs_sample.gs_cov = pc_variance;

            {
              std::lock_guard<std::mutex> lock(final_gs_map_mutex);
              final_gs_sample.push_back(_tmp_gs_sample);
            }
          }
        }
        delete[] h_global_testm;
        delete[] h_update_varis;
        delete[] h_error_codes;
      },
      "gpResultProcessing");

  common::Timer::Evaluate(
      gp_options_.log_time,
      ros::Time::now().toSec(),
      [&]() {
        for (int voxel_index = 0; voxel_index < num_voxels; voxel_index++) {
          delete[] h_voxels[voxel_index].points;
          delete[] h_voxels[voxel_index].global_points_testm;
          delete[] h_voxels[voxel_index].f_starm;
          delete[] h_voxels[voxel_index].k_variance;

          cudaMemcpy(&all_d_data[voxel_index], &d_voxels[voxel_index], sizeof(DVoxel), cudaMemcpyDeviceToHost);

          CHECK_CUDA(cudaFree(all_d_data[voxel_index].points), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].global_train_x_points), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].global_train_y_points), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].global_train_f_points), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].global_train_points_variance), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].ky), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].k_starm), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].kky), gp_options_.debug);

          CHECK_CUDA(cudaFree(all_d_data[voxel_index].global_points_testm), gp_options_.debug);

          CHECK_CUDA(cudaFree(all_d_data[voxel_index].f_starm), gp_options_.debug);
          CHECK_CUDA(cudaFree(all_d_data[voxel_index].k_variance), gp_options_.debug);
        }
        CHECK_CUDA(cudaFree(d_voxels), gp_options_.debug);
        delete[] h_voxels;
      },
      "gpFree");

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
  
}

bool gpProcess::getColors(
    const std::vector<Eigen::Vector3f>& points,
    std::vector<DColor>& colors,
    GSLIVM::ImageRt& frame) {
  // Initialize vectors to store transformed points and colors
  std::vector<Eigen::Vector3d> points_inworld(points.size());
  std::vector<Eigen::Vector3d> points_incam(points.size());
  colors.resize(points.size(), DColor());

  // Transform all points to camera coordinates
  for (size_t i = 0; i < points.size(); ++i) {
    points_inworld[i] = points[i].cast<double>();
    points_incam[i] = transformRawPointToCamera(points_inworld[i], frame);
  }

  // Project all points to image and get colors
  return projectPointsToImage(points_incam, colors, frame.image);
}

bool gpProcess::projectPointsToImage(
    const std::vector<Eigen::Vector3d>& points_incam,
    std::vector<DColor>& colors,
    cv::Mat& frame) {
  for (size_t i = 0; i < points_incam.size(); ++i) {
    const auto& point_incam = points_incam[i];
    float Xc = point_incam(0);
    float Yc = point_incam(1);
    float Zc = point_incam(2);

    float x_prime = Xc / Zc;
    float y_prime = Yc / Zc;
    float r = std::sqrt(x_prime * x_prime + y_prime * y_prime);
    float rd = r * (1 + d0 * std::pow(r, 2) + d1 * std::pow(r, 4) + d2 * std::pow(r, 6) + d3 * std::pow(r, 8));

    float x_prime_d = x_prime * rd / r;
    float y_prime_d = y_prime * rd / r;

    int u = static_cast<int>(fx * x_prime_d + cx);
    int v = static_cast<int>(fy * y_prime_d + cy);

    if (u >= 0 && u < frame.cols && v >= 0 && v < frame.rows) {
      cv::Vec3b pixel = frame.at<cv::Vec3b>(v, u);
      colors[i].b = static_cast<float>(pixel[0]);
      colors[i].g = static_cast<float>(pixel[1]);
      colors[i].r = static_cast<float>(pixel[2]);
    } else {
      // If point is out of image bounds, set color to a default value (e.g., black)
      colors[i].b = -1;
      colors[i].g = -1;
      colors[i].r = -1;
    }
  }
  return true;
}

Eigen::Vector3d gpProcess::transformRawPointToCamera(const Eigen::Vector3d& raw_point, GSLIVM::ImageRt& frame) {
  // Step 1: Transform the raw point from the last camera pose to the IMU frame.
  Eigen::Vector3d point_in_imu = frame.R.inverse() * (raw_point - frame.t);

  // Step 2: Apply the inverse IMU-to-LiDAR transformation to move the point to the LiDAR frame.
  point_in_imu -= frame.t_imu_lidar;
  point_in_imu = frame.R_imu_lidar.inverse() * point_in_imu;

  // Step 3: Finally, apply the LiDAR-to-camera transformation to obtain the point in the camera frame.
  Eigen::Vector3d point_incam = frame.R_camera_lidar * point_in_imu + frame.t_camera_lidar;
  return point_incam;
}