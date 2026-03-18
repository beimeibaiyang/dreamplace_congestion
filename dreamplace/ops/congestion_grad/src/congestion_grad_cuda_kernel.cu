/**
 * @file   congestion_grad_cuda_kernel.cu
 * @brief  CUDA kernel for congestion gradient computation
 * 
 * This implements the virtual cell gradient method for congestion optimization.
 * 
 * Algorithm:
 * Part A: Direct gradient for high-fanout cells in congested regions
 * Part B: Two-pin net virtual cell gradient with perpendicular force projection
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Device helper functions
// ============================================================================

/**
 * @brief Bilinear interpolation of field gradient
 */
__device__ void bilinear_interpolate(
    float x, float y,
    const float* field_map_x,
    const float* field_map_y,
    int num_bins_x, int num_bins_y,
    float bin_size_x, float bin_size_y,
    float xl, float yl,
    float& out_gx, float& out_gy
) {
    // Normalize to bin coordinates
    float fx = (x - xl) / bin_size_x;
    float fy = (y - yl) / bin_size_y;
    
    // Get integer bin indices
    int ix = (int)fx;
    int iy = (int)fy;
    
    // Clamp to valid range
    ix = max(0, min(ix, num_bins_x - 2));
    iy = max(0, min(iy, num_bins_y - 2));
    
    // Fractional parts
    fx = fx - ix;
    fy = fy - iy;
    fx = fmaxf(0.0f, fminf(fx, 1.0f));
    fy = fmaxf(0.0f, fminf(fy, 1.0f));
    
    // Bilinear interpolation weights
    float w00 = (1 - fx) * (1 - fy);
    float w10 = fx * (1 - fy);
    float w01 = (1 - fx) * fy;
    float w11 = fx * fy;
    
    // Index calculation (row-major)
    int idx00 = ix * num_bins_y + iy;
    int idx10 = (ix + 1) * num_bins_y + iy;
    int idx01 = ix * num_bins_y + (iy + 1);
    int idx11 = (ix + 1) * num_bins_y + (iy + 1);
    
    // Interpolate
    out_gx = w00 * field_map_x[idx00] + w10 * field_map_x[idx10] +
             w01 * field_map_x[idx01] + w11 * field_map_x[idx11];
    out_gy = w00 * field_map_y[idx00] + w10 * field_map_y[idx10] +
             w01 * field_map_y[idx01] + w11 * field_map_y[idx11];
}


/**
 * @brief Get congestion value at position (x, y)
 */
__device__ float get_congestion_at(
    float x, float y,
    const float* congestion_map,
    int num_bins_x, int num_bins_y,
    float bin_size_x, float bin_size_y,
    float xl, float yl
) {
    int bin_x = (int)((x - xl) / bin_size_x);
    int bin_y = (int)((y - yl) / bin_size_y);
    bin_x = max(0, min(bin_x, num_bins_x - 1));
    bin_y = max(0, min(bin_y, num_bins_y - 1));
    return congestion_map[bin_x * num_bins_y + bin_y];
}


// ============================================================================
// Kernel: Count pins per node
// ============================================================================
__global__ void compute_pin_counts_kernel(
    const int* pin2node_map,
    int* pin_counts,
    int num_pins,
    int num_nodes
) {
    int pin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin_idx >= num_pins) return;
    
    int node_idx = pin2node_map[pin_idx];
    if (node_idx < num_nodes) {
        atomicAdd(&pin_counts[node_idx], 1);
    }
}


// ============================================================================
// Kernel: Direct gradient for high-fanout cells
// ============================================================================
__global__ void apply_direct_gradient_kernel(
    const float* pos_x,
    const float* pos_y,
    const float* congestion_map,
    const float* field_map_x,
    const float* field_map_y,
    const float* node_size_x,
    const float* node_size_y,
    const int* pin_counts,
    float* grad_x,
    float* grad_y,
    int num_movable_nodes,
    int num_bins_x,
    int num_bins_y,
    float bin_size_x,
    float bin_size_y,
    float xl,
    float yl,
    float avg_pin_per_node,
    float congestion_threshold
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_movable_nodes) return;
    
    // Check if high-fanout cell
    int pin_count = pin_counts[node_idx];
    if (pin_count <= avg_pin_per_node) return;
    
    // Get cell center
    float cx = pos_x[node_idx] + node_size_x[node_idx] / 2;
    float cy = pos_y[node_idx] + node_size_y[node_idx] / 2;
    
    // Get congestion at cell position
    float cong = get_congestion_at(cx, cy, congestion_map,
                                   num_bins_x, num_bins_y,
                                   bin_size_x, bin_size_y, xl, yl);
    
    // Only process high congestion cells
    if (cong < congestion_threshold) return;
    
    // Compute scale
    float cell_area = node_size_x[node_idx] * node_size_y[node_idx];
    float scale = cong * cell_area;
    
    // Get field gradient
    float field_gx, field_gy;
    bilinear_interpolate(cx, cy, field_map_x, field_map_y,
                         num_bins_x, num_bins_y, bin_size_x, bin_size_y,
                         xl, yl, field_gx, field_gy);
    
    // Accumulate gradient
    atomicAdd(&grad_x[node_idx], scale * field_gx);
    atomicAdd(&grad_y[node_idx], scale * field_gy);
}


// ============================================================================
// Kernel: Two-pin net virtual cell gradient
// ============================================================================
__global__ void compute_twopin_grad_kernel(
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
    float* grad_x,
    float* grad_y,
    int num_nets,
    int num_movable_nodes,
    int num_nodes,
    int num_bins_x,
    int num_bins_y,
    float bin_size_x,
    float bin_size_y,
    float xl,
    float yl,
    float congestion_threshold
) {
    int net_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_idx >= num_nets) return;
    
    int pin_start = flat_net2pin_start_map[net_idx];
    int pin_end = flat_net2pin_start_map[net_idx + 1];
    int num_pins = pin_end - pin_start;
    
    if (num_pins < 2) return;
    
    float net_weight = net_weights[net_idx];
    
    // Get first pin as hub for star decomposition
    int pin0_idx = flat_net2pin_map[pin_start];
    int node0 = pin2node_map[pin0_idx];
    
    if (node0 >= num_nodes) return;
    
    float x0 = pos_x[node0] + pin_offset_x[pin0_idx];
    float y0 = pos_y[node0] + pin_offset_y[pin0_idx];
    
    // Connect pin0 to all other pins (star decomposition)
    for (int i = 1; i < num_pins; i++) {
        int pin_i_idx = flat_net2pin_map[pin_start + i];
        int node_i = pin2node_map[pin_i_idx];
        
        if (node_i >= num_nodes) continue;
        
        float x1 = pos_x[node_i] + pin_offset_x[pin_i_idx];
        float y1 = pos_y[node_i] + pin_offset_y[pin_i_idx];
        
        // Vector from pin0 to pin1
        float dx = x1 - x0;
        float dy = y1 - y0;
        float L = sqrtf(dx * dx + dy * dy);
        
        if (L < 1e-6f) continue;
        
        // Sample points to find maximum congestion
        int num_samples = max(2, min(20, (int)fmaxf(fabsf(dx) / bin_size_x,
                                                    fabsf(dy) / bin_size_y)));
        
        float max_cong = 0.0f;
        float max_t = 0.5f;
        
        for (int s = 0; s <= num_samples; s++) {
            float t = (float)s / num_samples;
            float sx = x0 + t * dx;
            float sy = y0 + t * dy;
            
            float cong = get_congestion_at(sx, sy, congestion_map,
                                           num_bins_x, num_bins_y,
                                           bin_size_x, bin_size_y, xl, yl);
            if (cong > max_cong) {
                max_cong = cong;
                max_t = t;
            }
        }
        
        // Skip if no significant congestion
        if (max_cong < congestion_threshold) continue;
        
        // Virtual cell position
        float vx = x0 + max_t * dx;
        float vy = y0 + max_t * dy;
        
        // Get field gradient at virtual cell
        float field_gx, field_gy;
        bilinear_interpolate(vx, vy, field_map_x, field_map_y,
                             num_bins_x, num_bins_y, bin_size_x, bin_size_y,
                             xl, yl, field_gx, field_gy);
        
        // Perpendicular direction: n_perp = (-dy, dx) / L
        float perp_x = -dy / L;
        float perp_y = dx / L;
        
        // Project gradient onto perpendicular direction
        float dot = field_gx * perp_x + field_gy * perp_y;
        float grad_perp_x = dot * perp_x;
        float grad_perp_y = dot * perp_y;
        
        // Distances from virtual cell to endpoints
        float d0 = fmaxf(max_t * L, L * 0.1f);
        float d1 = fmaxf((1 - max_t) * L, L * 0.1f);
        
        // Distribute force inversely proportional to distance
        float scale0 = fminf((L / (2 * d0)) * net_weight * max_cong, 10.0f);
        float scale1 = fminf((L / (2 * d1)) * net_weight * max_cong, 10.0f);
        
        // Apply gradient to movable nodes
        if (node0 < num_movable_nodes) {
            atomicAdd(&grad_x[node0], scale0 * grad_perp_x);
            atomicAdd(&grad_y[node0], scale0 * grad_perp_y);
        }
        
        if (node_i < num_movable_nodes) {
            atomicAdd(&grad_x[node_i], scale1 * grad_perp_x);
            atomicAdd(&grad_y[node_i], scale1 * grad_perp_y);
        }
    }
}


// ============================================================================
// Host wrapper function
// ============================================================================
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
) {
    const int BLOCK_SIZE = 256;
    
    // Allocate and compute pin counts
    int* pin_counts;
    cudaMalloc(&pin_counts, num_nodes * sizeof(int));
    cudaMemset(pin_counts, 0, num_nodes * sizeof(int));
    
    int pin_count_blocks = (num_pins + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_pin_counts_kernel<<<pin_count_blocks, BLOCK_SIZE, 0, stream>>>(
        pin2node_map, pin_counts, num_pins, num_nodes
    );
    
    // Part A: Direct gradient for high-fanout cells
    int direct_blocks = (num_movable_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_direct_gradient_kernel<<<direct_blocks, BLOCK_SIZE, 0, stream>>>(
        pos_x, pos_y, congestion_map, field_map_x, field_map_y,
        node_size_x, node_size_y, pin_counts,
        grad_x, grad_y,
        num_movable_nodes, num_bins_x, num_bins_y,
        bin_size_x, bin_size_y, xl, yl,
        avg_pin_per_node, congestion_threshold
    );
    
    // Part B: Two-pin net gradient
    int net_blocks = (num_nets + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_twopin_grad_kernel<<<net_blocks, BLOCK_SIZE, 0, stream>>>(
        pos_x, pos_y, congestion_map, field_map_x, field_map_y,
        flat_net2pin_map, flat_net2pin_start_map, pin2node_map,
        pin_offset_x, pin_offset_y, net_weights,
        grad_x, grad_y,
        num_nets, num_movable_nodes, num_nodes,
        num_bins_x, num_bins_y, bin_size_x, bin_size_y, xl, yl,
        congestion_threshold
    );
    
    // Cleanup
    cudaFree(pin_counts);
}
