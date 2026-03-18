##
# @file   congestion_grad.py
# @author DREAMPlace
# @brief  Congestion gradient computation for global placement
#
# This module computes gradients based on congestion maps to help reduce
# routing congestion during global placement. It implements:
# 1. Direct gradient for high-fanout cells in congested regions
# 2. Virtual cell gradient for two-pin nets using perpendicular force projection
#

import torch
import numpy as np
import logging

# Try to import CUDA extension
try:
    import dreamplace.ops.congestion_grad.congestion_grad_cuda as congestion_grad_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    logging.warning("[CongestionGrad] CUDA extension not available, using Python fallback")


class CongestionGrad:
    """
    Congestion gradient computation operator.
    
    Algorithm:
    1. For cells with pins > average AND in high congestion region:
       - Apply direct gradient: grad += congestion * field_grad * cell_area
       
    2. For each two-pin net (star decomposition of multi-pin nets):
       - Find maximum congestion point along the connection line (virtual cell)
       - Get field gradient at virtual cell position (bilinear interpolation)
       - Compute perpendicular direction to the connection line
       - Project gradient to perpendicular direction (doesn't change wirelength)
       - Distribute force to both endpoints inversely proportional to distance
    """
    
    def __init__(self,
                 flat_net2pin_map,
                 flat_net2pin_start_map,
                 pin2node_map,
                 pin_offset_x,
                 pin_offset_y,
                 net_weights,
                 node_size_x,
                 node_size_y,
                 xl, yl, xh, yh,
                 bin_size_x, bin_size_y,
                 num_bins_x, num_bins_y,
                 num_movable_nodes,
                 num_nodes,
                 avg_pin_per_node,
                 congestion_threshold=0.7):
        """
        Initialize congestion gradient operator.
        
        Args:
            flat_net2pin_map: Flattened net to pin mapping
            flat_net2pin_start_map: Start indices for each net in flat_net2pin_map
            pin2node_map: Pin to node mapping
            pin_offset_x/y: Pin offset from cell origin
            net_weights: Weight for each net
            node_size_x/y: Cell dimensions
            xl, yl, xh, yh: Placement region bounds
            bin_size_x/y: Bin dimensions
            num_bins_x/y: Number of bins
            num_movable_nodes: Number of movable cells
            num_nodes: Total number of cells
            avg_pin_per_node: Average pins per node (for high-fanout detection)
            congestion_threshold: Threshold for high congestion (default 0.7)
        """
        self.flat_net2pin_map = flat_net2pin_map
        self.flat_net2pin_start_map = flat_net2pin_start_map
        self.pin2node_map = pin2node_map
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.net_weights = net_weights
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        
        self.num_movable_nodes = num_movable_nodes
        self.num_nodes = num_nodes
        self.avg_pin_per_node = avg_pin_per_node
        self.congestion_threshold = congestion_threshold
        
        # Cached field maps (set by set_field_maps)
        self.field_grad_x = None
        self.field_grad_y = None
        
        # Precompute pin counts per node
        self._compute_pin_counts()
        
        logging.info(f"[CongestionGrad] Initialized: bins={num_bins_x}x{num_bins_y}, "
                     f"threshold={congestion_threshold}, avg_pins={avg_pin_per_node:.2f}")
    
    def _compute_pin_counts(self):
        """Compute number of pins for each node."""
        device = self.pin2node_map.device
        self.pin_counts = torch.zeros(self.num_nodes, dtype=torch.int32, device=device)
        
        # Count pins per node
        for pin_idx in range(len(self.pin2node_map)):
            node_idx = self.pin2node_map[pin_idx].item()
            if node_idx < self.num_nodes:
                self.pin_counts[node_idx] += 1
    
    def set_field_maps(self, field_grad_x, field_grad_y):
        """
        Set the field gradient maps computed from congestion potential.
        
        Args:
            field_grad_x: X-component of field gradient [num_bins_x, num_bins_y]
            field_grad_y: Y-component of field gradient [num_bins_x, num_bins_y]
        """
        self.field_grad_x = field_grad_x
        self.field_grad_y = field_grad_y
    
    def compute_grad_direct(self, pos, congestion_map):
        """
        Compute congestion gradient.
        
        Args:
            pos: Cell positions [num_nodes * 2], format: (x0, x1, ..., xn, y0, y1, ..., yn)
            congestion_map: 2D congestion map [num_bins_x, num_bins_y]
            
        Returns:
            grad: Gradient tensor [num_nodes * 2]
        """
        if self.field_grad_x is None or self.field_grad_y is None:
            raise RuntimeError("[CongestionGrad] Field maps not set. Call set_field_maps first.")
        
        # Use CUDA if available and inputs are on GPU
        if HAS_CUDA and pos.is_cuda:
            return self._compute_grad_cuda(pos, congestion_map)
        else:
            return self._compute_grad_python(pos, congestion_map)
    
    def _compute_grad_cuda(self, pos, congestion_map):
        """CUDA implementation of gradient computation."""
        return congestion_grad_cuda.forward(
            pos,
            congestion_map,
            self.field_grad_x,
            self.field_grad_y,
            self.flat_net2pin_map,
            self.flat_net2pin_start_map,
            self.pin2node_map,
            self.pin_offset_x,
            self.pin_offset_y,
            self.net_weights,
            self.node_size_x,
            self.node_size_y,
            self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y,
            self.num_bins_x, self.num_bins_y,
            self.num_movable_nodes, self.num_nodes,
            self.avg_pin_per_node, self.congestion_threshold
        )
    
    def _compute_grad_python(self, pos, congestion_map):
        """
        Python fallback implementation.
        Exactly mirrors the CUDA algorithm for consistency.
        """
        device = pos.device
        dtype = pos.dtype
        
        # Initialize gradient
        grad = torch.zeros(self.num_nodes * 2, dtype=dtype, device=device)
        
        # Split pos into x and y
        pos_x = pos[:self.num_nodes]
        pos_y = pos[self.num_nodes:]
        
        # ========================================
        # Part A: Direct gradient for high-fanout cells
        # ========================================
        for node_idx in range(self.num_movable_nodes):
            pin_count = self.pin_counts[node_idx].item()
            
            # Only process high-fanout cells
            if pin_count <= self.avg_pin_per_node:
                continue
            
            # Get cell center position
            cx = pos_x[node_idx].item() + self.node_size_x[node_idx].item() / 2
            cy = pos_y[node_idx].item() + self.node_size_y[node_idx].item() / 2
            
            # Get bin index
            bin_x = int((cx - self.xl) / self.bin_size_x)
            bin_y = int((cy - self.yl) / self.bin_size_y)
            bin_x = max(0, min(bin_x, self.num_bins_x - 1))
            bin_y = max(0, min(bin_y, self.num_bins_y - 1))
            
            # Get congestion value
            cong = congestion_map[bin_x, bin_y].item()
            
            # Only process cells in high congestion regions
            if cong < self.congestion_threshold:
                continue
            
            # Compute gradient scale
            cell_area = self.node_size_x[node_idx].item() * self.node_size_y[node_idx].item()
            scale = cong * cell_area
            
            # Get field gradient at cell position (bilinear interpolation)
            field_gx, field_gy = self._bilinear_interpolate(cx, cy)
            
            # Accumulate gradient
            grad[node_idx] += scale * field_gx
            grad[self.num_nodes + node_idx] += scale * field_gy
        
        # ========================================
        # Part B: Two-pin net virtual cell gradient
        # ========================================
        num_nets = len(self.flat_net2pin_start_map) - 1
        
        for net_idx in range(num_nets):
            pin_start = self.flat_net2pin_start_map[net_idx].item()
            pin_end = self.flat_net2pin_start_map[net_idx + 1].item()
            num_pins = pin_end - pin_start
            
            if num_pins < 2:
                continue
            
            net_weight = self.net_weights[net_idx].item() if net_idx < len(self.net_weights) else 1.0
            
            # Star decomposition: pin0 connects to all other pins
            pin0_idx = self.flat_net2pin_map[pin_start].item()
            node0 = self.pin2node_map[pin0_idx].item()
            
            if node0 >= self.num_nodes:
                continue
                
            x0 = pos_x[node0].item() + self.pin_offset_x[pin0_idx].item()
            y0 = pos_y[node0].item() + self.pin_offset_y[pin0_idx].item()
            
            for i in range(1, num_pins):
                pin_i_idx = self.flat_net2pin_map[pin_start + i].item()
                node_i = self.pin2node_map[pin_i_idx].item()
                
                if node_i >= self.num_nodes:
                    continue
                
                x1 = pos_x[node_i].item() + self.pin_offset_x[pin_i_idx].item()
                y1 = pos_y[node_i].item() + self.pin_offset_y[pin_i_idx].item()
                
                # Compute two-pin gradient
                self._compute_twopin_grad(
                    x0, y0, node0,
                    x1, y1, node_i,
                    net_weight,
                    congestion_map,
                    grad
                )
        
        return grad
    
    def _bilinear_interpolate(self, x, y):
        """
        Bilinear interpolation of field gradient at position (x, y).
        
        Returns:
            (field_gx, field_gy): Interpolated field gradient
        """
        # Normalize to bin coordinates
        fx = (x - self.xl) / self.bin_size_x
        fy = (y - self.yl) / self.bin_size_y
        
        # Get integer bin indices
        ix = int(fx)
        iy = int(fy)
        
        # Clamp to valid range
        ix = max(0, min(ix, self.num_bins_x - 2))
        iy = max(0, min(iy, self.num_bins_y - 2))
        
        # Fractional parts
        fx = fx - ix
        fy = fy - iy
        fx = max(0.0, min(fx, 1.0))
        fy = max(0.0, min(fy, 1.0))
        
        # Bilinear interpolation weights
        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy
        
        # Interpolate field gradient
        field_gx = (w00 * self.field_grad_x[ix, iy].item() +
                    w10 * self.field_grad_x[ix + 1, iy].item() +
                    w01 * self.field_grad_x[ix, iy + 1].item() +
                    w11 * self.field_grad_x[ix + 1, iy + 1].item())
        
        field_gy = (w00 * self.field_grad_y[ix, iy].item() +
                    w10 * self.field_grad_y[ix + 1, iy].item() +
                    w01 * self.field_grad_y[ix, iy + 1].item() +
                    w11 * self.field_grad_y[ix + 1, iy + 1].item())
        
        return field_gx, field_gy
    
    def _compute_twopin_grad(self, x0, y0, node0, x1, y1, node1, 
                             net_weight, congestion_map, grad):
        """
        Compute gradient for a two-pin connection using virtual cell method.
        
        The algorithm:
        1. Sample points along the line to find maximum congestion point (virtual cell)
        2. Get field gradient at virtual cell position
        3. Compute perpendicular direction to the line
        4. Project gradient to perpendicular (doesn't change wirelength)
        5. Distribute force inversely proportional to distance from virtual cell
        """
        # Vector from pin0 to pin1
        dx = x1 - x0
        dy = y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        
        if L < 1e-6:
            return
        
        # Sample points along the line to find maximum congestion
        num_samples = max(2, int(max(abs(dx) / self.bin_size_x, 
                                      abs(dy) / self.bin_size_y)))
        num_samples = min(num_samples, 20)  # Limit samples
        
        max_cong = 0.0
        max_t = 0.5  # Default to midpoint
        
        for s in range(num_samples + 1):
            t = s / num_samples
            sx = x0 + t * dx
            sy = y0 + t * dy
            
            # Get bin index
            bin_x = int((sx - self.xl) / self.bin_size_x)
            bin_y = int((sy - self.yl) / self.bin_size_y)
            bin_x = max(0, min(bin_x, self.num_bins_x - 1))
            bin_y = max(0, min(bin_y, self.num_bins_y - 1))
            
            cong = congestion_map[bin_x, bin_y].item()
            if cong > max_cong:
                max_cong = cong
                max_t = t
        
        # Skip if no significant congestion
        if max_cong < self.congestion_threshold:
            return
        
        # Virtual cell position
        vx = x0 + max_t * dx
        vy = y0 + max_t * dy
        
        # Get field gradient at virtual cell (bilinear interpolation)
        field_gx, field_gy = self._bilinear_interpolate(vx, vy)
        
        # Perpendicular direction (rotate 90 degrees): n_perp = (-dy, dx) / L
        perp_x = -dy / L
        perp_y = dx / L
        
        # Project gradient onto perpendicular direction
        # This ensures the force doesn't change wirelength
        dot = field_gx * perp_x + field_gy * perp_y
        grad_perp_x = dot * perp_x
        grad_perp_y = dot * perp_y
        
        # Distances from virtual cell to each endpoint
        d0 = max_t * L
        d1 = (1 - max_t) * L
        
        # Distribute force inversely proportional to distance
        # F_i = (L / 2d_i) * grad_perp
        # Clamp to avoid division by zero and extreme values
        d0 = max(d0, L * 0.1)
        d1 = max(d1, L * 0.1)
        
        scale0 = min((L / (2 * d0)) * net_weight * max_cong, 10.0)
        scale1 = min((L / (2 * d1)) * net_weight * max_cong, 10.0)
        
        # Apply gradient to movable nodes only
        if node0 < self.num_movable_nodes:
            grad[node0] += scale0 * grad_perp_x
            grad[self.num_nodes + node0] += scale0 * grad_perp_y
        
        if node1 < self.num_movable_nodes:
            grad[node1] += scale1 * grad_perp_x
            grad[self.num_nodes + node1] += scale1 * grad_perp_y


def compute_congestion_grad_python(pos, congestion_map, field_grad_x, field_grad_y,
                                    flat_net2pin_map, flat_net2pin_start_map,
                                    pin2node_map, pin_offset_x, pin_offset_y,
                                    net_weights, node_size_x, node_size_y,
                                    xl, yl, xh, yh, bin_size_x, bin_size_y,
                                    num_bins_x, num_bins_y,
                                    num_movable_nodes, num_nodes,
                                    avg_pin_per_node, congestion_threshold):
    """
    Standalone Python function for congestion gradient computation.
    Can be used for testing or when CongestionGrad class is not needed.
    """
    op = CongestionGrad(
        flat_net2pin_map=flat_net2pin_map,
        flat_net2pin_start_map=flat_net2pin_start_map,
        pin2node_map=pin2node_map,
        pin_offset_x=pin_offset_x,
        pin_offset_y=pin_offset_y,
        net_weights=net_weights,
        node_size_x=node_size_x,
        node_size_y=node_size_y,
        xl=xl, yl=yl, xh=xh, yh=yh,
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        num_movable_nodes=num_movable_nodes,
        num_nodes=num_nodes,
        avg_pin_per_node=avg_pin_per_node,
        congestion_threshold=congestion_threshold
    )
    op.set_field_maps(field_grad_x, field_grad_y)
    return op.compute_grad_direct(pos, congestion_map)
