/**
 * @file   congestion_grad.cpp
 * @brief  C++/PyTorch wrapper for congestion gradient computation
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// CUDA forward declaration
extern "C" void launch_congestion_grad_kernel(
    const float* pos_x,
    const float* pos_y,
    const float* congestion_map,
    const float* field_map_x,
    const float* field_map_y,
    const int* flat_net2pin_map,
    const int* flat_net2pin_start_map,
    const int* pin2node_map,
    const float* pin_offset_x,
    const float* pin_offset_y,
    const float* net_weights,
    const float* node_size_x,
    const float* node_size_y,
    float* grad_x,
    float* grad_y,
    int num_nets,
    int num_pins,
    int num_movable_nodes,
    int num_nodes,
    int num_bins_x,
    int num_bins_y,
    float bin_size_x,
    float bin_size_y,
    float xl,
    float yl,
    float avg_pin_per_node,
    float congestion_threshold,
    cudaStream_t stream
);


/**
 * @brief Compute congestion gradient using CUDA
 * 
 * @param pos Cell positions [num_nodes * 2]: (x0, x1, ..., y0, y1, ...)
 * @param congestion_map 2D congestion map [num_bins_x, num_bins_y]
 * @param field_map_x X-component of electric field [num_bins_x, num_bins_y]
 * @param field_map_y Y-component of electric field [num_bins_x, num_bins_y]
 * @param flat_net2pin_map Flattened net to pin map
 * @param flat_net2pin_start_map Starting index for each net
 * @param pin2node_map Pin to node map
 * @param pin_offset_x Pin x offset from cell origin
 * @param pin_offset_y Pin y offset from cell origin
 * @param net_weights Weight for each net
 * @param node_size_x Cell width
 * @param node_size_y Cell height
 * @param xl, yl, xh, yh Placement region bounds
 * @param bin_size_x, bin_size_y Bin sizes
 * @param num_bins_x, num_bins_y Number of bins
 * @param num_movable_nodes Number of movable nodes
 * @param num_nodes Total number of nodes
 * @param avg_pin_per_node Average pins per node
 * @param congestion_threshold Threshold for high congestion
 * 
 * @return Gradient tensor [num_nodes * 2]
 */
torch::Tensor congestion_grad_forward(
    torch::Tensor pos,
    torch::Tensor congestion_map,
    torch::Tensor field_map_x,
    torch::Tensor field_map_y,
    torch::Tensor flat_net2pin_map,
    torch::Tensor flat_net2pin_start_map,
    torch::Tensor pin2node_map,
    torch::Tensor pin_offset_x,
    torch::Tensor pin_offset_y,
    torch::Tensor net_weights,
    torch::Tensor node_size_x,
    torch::Tensor node_size_y,
    double xl, double yl, double xh, double yh,
    double bin_size_x, double bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_movable_nodes, int num_nodes,
    double avg_pin_per_node, double congestion_threshold
) {
    // Check inputs
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(congestion_map.is_cuda(), "congestion_map must be a CUDA tensor");
    TORCH_CHECK(field_map_x.is_cuda(), "field_map_x must be a CUDA tensor");
    TORCH_CHECK(field_map_y.is_cuda(), "field_map_y must be a CUDA tensor");
    
    int num_nets = flat_net2pin_start_map.size(0) - 1;
    int num_pins = pin2node_map.size(0);
    
    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(pos.dtype())
        .device(pos.device());
    
    torch::Tensor grad = torch::zeros({num_nodes * 2}, options);
    
    // Split pos into x and y
    torch::Tensor pos_x = pos.slice(0, 0, num_nodes).contiguous();
    torch::Tensor pos_y = pos.slice(0, num_nodes, num_nodes * 2).contiguous();
    
    // Split grad into x and y
    torch::Tensor grad_x = grad.slice(0, 0, num_nodes);
    torch::Tensor grad_y = grad.slice(0, num_nodes, num_nodes * 2);
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_congestion_grad_kernel(
        pos_x.data_ptr<float>(),
        pos_y.data_ptr<float>(),
        congestion_map.contiguous().data_ptr<float>(),
        field_map_x.contiguous().data_ptr<float>(),
        field_map_y.contiguous().data_ptr<float>(),
        flat_net2pin_map.data_ptr<int>(),
        flat_net2pin_start_map.data_ptr<int>(),
        pin2node_map.data_ptr<int>(),
        pin_offset_x.data_ptr<float>(),
        pin_offset_y.data_ptr<float>(),
        net_weights.data_ptr<float>(),
        node_size_x.data_ptr<float>(),
        node_size_y.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        num_nets,
        num_pins,
        num_movable_nodes,
        num_nodes,
        num_bins_x,
        num_bins_y,
        (float)bin_size_x,
        (float)bin_size_y,
        (float)xl,
        (float)yl,
        (float)avg_pin_per_node,
        (float)congestion_threshold,
        stream
    );
    
    return grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &congestion_grad_forward, "Congestion gradient forward (CUDA)");
}
