#include "ATen/NativeFunctions.h"
// #include <tuple>

namespace at {
namespace native {

template <typename scalar>
__device__ __forceinline__ scalar fmin(scalar a, scalar b)
{
  return a > b ? b : a;
}

template <typename scalar>
__device__ __forceinline__ scalar fmax(scalar a, scalar b)
{
  return a > b ? a : b;
}

template <typename scalar>
__device__ __forceinline__ scalar IoU(const scalar* box_x, const scalar* box_y)
{
  // Calculate IoU between the boxes.
  scalar rightmost_l = fmax(box_x[0], box_y[0]);
  scalar leftmost_r = fmin(box_x[0] + box_x[2], box_y[0] + box_y[2]);
  scalar delta_x = fmax((scalar)0., leftmost_r - rightmost_l);

  scalar bottommost_tp = fmax(box_x[1], box_y[1]);
  scalar topmost_b = fmin(box_x[1] + box_x[3], box_y[1] + box_y[3]);
  scalar delta_y = fmax((scalar)0., topmost_b - bottommost_tp);

  scalar uni = box_x[2] * box_x[3] + box_y[2] * box_y[3];

  return delta_x * delta_y / (uni - delta_x * delta_y);

}

template <typename scalar>
__global__ void nms_kernel(unsigned char* mask,
                          const scalar* boxes,
                          const int64_t* inds,
                          const int64_t num_boxes,
                          double thresh)
{
//A pretty straightforward implementation, analogous to the standard serial
//version but with the IoUs computed and mask updated in parallel. We access
//the box data through an array of sorted indices rather than physically
//sorting it: unless one has an inordinate number of boxes (O(10^5), whereas
//for example in the faster rcnn paper they feed 6000 per batch) the
//data will fit in L2 so sorting it won't actually reduce the number of
//messy reads from global.
  int col = 0;
  while(col < num_boxes-1)
  {
    for(int i = threadIdx.x; i < num_boxes-1; i+=blockDim.x)
      if(i >= col)
      {
        scalar iou = IoU(&boxes[4*inds[i+1+num_boxes*blockIdx.x] + 4*num_boxes*blockIdx.x],
                        &boxes[4*inds[col+num_boxes*blockIdx.x] + 4*num_boxes*blockIdx.x]);
        mask[i+1+blockIdx.x*num_boxes] *= (iou>thresh) ? 0 : 1;
      }
    __syncthreads();
    ++col;
    while((col < num_boxes - 1) && (mask[col+blockIdx.x*num_boxes]==0))
      ++col;
  }
}

Tensor non_max_suppression_cuda(const Tensor& input,
                                const Tensor& scores,
                                double thresh)
{
  AT_CHECK(input.ndimension() == 3,
            "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
  AT_CHECK(scores.ndimension() == 2,
            "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
  AT_CHECK(input.size(0) == scores.size(0),
            "First and second arguments must have equal-sized first dimension");
  AT_CHECK(input.size(1) == scores.size(1),
            "First and second arguments must have equal-sized second dimension");
  AT_CHECK(input.size(2) == 4,
           "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
  AT_CHECK(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_CHECK(scores.is_contiguous(), "Second argument must be a contiguous Tensor");
  AT_CHECK(input.type().scalarType() == kFloat || input.type().scalarType() == kDouble,
      "First argument must be Float or Double Tensor")
  AT_CHECK(scores.type().scalarType() == kFloat || scores.type().scalarType() == kDouble,
      "Second argument must be Float or Double Tensor")


  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = input.type().toScalarType(kByte).tensor({batch_size, num_boxes});
  mask.fill_(1);

  //need the indices of the boxes sorted by score.
  Tensor sorted_inds = std::get<1>(scores.sort(-1, true));


  dim3 mask_block(512); //would be nice to have 1024 here for gpus that support it,
                        //but not sure how to do this cleanly without calling
                        //cudaGetDeviceProperties in the funcion body...

  dim3 mask_grid(batch_size);
  if(input.type().scalarType() == kFloat)
  {
      nms_kernel<<<mask_grid, mask_block, 0, globalContext().getCurrentCUDAStream()>>>(
                                        mask.data<unsigned char>(),
                                        input.data<float>(),
                                        sorted_inds.data<int64_t>(),
                                        num_boxes,
                                        thresh);
      AT_CHECK(cudaGetLastError() == cudaSuccess, "nms_mask_kernel failed");
  }
  else
  {
      nms_kernel<<<mask_grid, mask_block, 0, globalContext().getCurrentCUDAStream()>>>(
                                        mask.data<unsigned char>(),
                                        input.data<double>(),
                                        sorted_inds.data<int64_t>(),
                                        num_boxes,
                                        thresh);
      AT_CHECK(cudaGetLastError() == cudaSuccess, "nms_mask_kernel failed");
  }

  //It's not entirely clear what the best thing to return is here. The algorithm will
  //produce a different number of boxes for each batch, so there is no obvious way of
  //way of returning the surving boxes/indices as a tensor. Returning a mask on the
  //sorted boxes together with the sorted indices seems reasonable; that way, the user
  //can easily take the N highest-scoring surviving boxes to form a tensor if they wish.
  return sorted_inds;
}

}//native
}//at