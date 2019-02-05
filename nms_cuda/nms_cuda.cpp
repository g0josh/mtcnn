#include "ATen/NativeFunctions.h"
// #include <tuple>
#include <torch/torch.h>

// at::Tensor non_max_suppression_cuda(
//                                 const at::Tensor& input,
//                                 const at::Tensor& scores,
//                                 double thresh);
at::Tensor non_max_suppression_cuda(const at::Tensor boxes,
                                float nms_overlap_thresh);

// at::Tensor nms_cuda(const at::Tensor& input,
//                             const at::Tensor& scores,
//                             double thresh)

at::Tensor nms_cuda(const at::Tensor input,
                    float thresh)
{

    AT_CHECK(input.ndimension() == 3,
        "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
    // AT_CHECK(scores.ndimens/ion() == 2,
        // "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
    // AT_CHECK(input.size(0) == scores.size(0),
        // "First and second arguments must have equal-sized first dimensions");
    // AT_CHECK(input.size(1) == scores.size(1),
        // "First and second arguments must have equal-sized second dimensions");
    AT_CHECK(input.size(2) == 4,
        "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
    AT_CHECK(input.is_contiguous(), "First argument must be a contiguous Tensor");
    // AT_CHECK(scores.is_contiguous(), "Second argument must be a contiguous Tensor");
    AT_CHECK(input.type().scalarType() == at::kFloat || input.type().scalarType() == at::kDouble,
        "First argument must be Float or Double Tensor");
    // AT_CHECK(scores.type().scalarType() == at::kFloat || scores.type().scalarType() == at::kDouble,
        // "Second argument must be Float or Double Tensor");
    AT_CHECK(input.is_contiguous(), "First argument must be a contiguous Tensor");
    // AT_CHECK(scores.is_contiguous(), "Second argument must be a contiguous Tensor");

    return non_max_suppression_cuda(input, thresh);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_cuda", &nms_cuda, "Non max suppression on CUDA");
}