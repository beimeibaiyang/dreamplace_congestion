##
# @file   PlaceObj.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Placement model class defining the placement objective.
#

import os
import sys
import time
import numpy as np
import itertools
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength as weighted_average_wirelength
import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength as logsumexp_wirelength
import dreamplace.ops.density_overflow.density_overflow as density_overflow
import dreamplace.ops.electric_potential.electric_overflow as electric_overflow
import dreamplace.ops.electric_potential.electric_potential as electric_potential
import dreamplace.ops.density_potential.density_potential as density_potential
import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pin_utilization.pin_utilization as pin_utilization
import dreamplace.ops.nctugr_binary.nctugr_binary as nctugr_binary
import dreamplace.ops.adjust_node_area.adjust_node_area as adjust_node_area
import dreamplace.ops.gift_init.gift_init as gift_init

try:
    import dreamplace.ops.congestion_grad.congestion_grad as congestion_grad
    HAS_CONGESTION_GRAD = True
except ImportError:
    HAS_CONGESTION_GRAD = False
    import logging
    logging.warning("[PlaceObj] congestion_grad module not available")

try:
    from torch.fft import fft2, ifft2, rfft2, irfft2
    HAS_FFT = True
except ImportError:
    HAS_FFT = False
    logging.warning("[PlaceObj] torch.fft not available, using scipy fallback")


class PreconditionOp:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """
    def __init__(self, placedb, data_collections, op_collections):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []
        if len(placedb.regions) > 0:
            self.movablenode2fence_region_map_clamp = (
                data_collections.node2fence_region_map[: placedb.num_movable_nodes]
                .clamp(max=len(placedb.regions))
                .long()
            )
            self.filler2fence_region_map = torch.zeros(
                placedb.num_filler_nodes, device=data_collections.pos[0].device, dtype=torch.long
            )
            for i in range(len(placedb.regions) + 1):
                filler_beg, filler_end = self.placedb.filler_start_map[i : i + 2]
                self.filler2fence_region_map[filler_beg:filler_end] = i

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        elif self.best_overflow.mean() > overflow.mean():
            self.best_overflow = overflow

    def __call__(self, grad, density_weight, update_mask=None, fix_nodes_mask=None):
        """Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            # The preconditioning step in python is time-consuming, as in each gradient
            # pass, the total net weight should be re-calculated.
            sum_pin_weights_in_nodes = self.op_collections.pws_op(self.data_collections.net_weights)
            if density_weight.size(0) == 1:
                precond = (sum_pin_weights_in_nodes
                    + self.alpha * density_weight * self.data_collections.node_areas
                )
            else:
                ### only precondition the non fence region
                node_areas = self.data_collections.node_areas.clone()

                mask = self.data_collections.node2fence_region_map[: self.placedb.num_movable_nodes] >= len(
                    self.placedb.regions
                )
                node_areas[: self.placedb.num_movable_nodes].masked_scatter_(
                    mask, node_areas[: self.placedb.num_movable_nodes][mask] * density_weight[-1]
                )
                filler_beg, filler_end = self.placedb.filler_start_map[-2:]
                node_areas[
                    self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_beg : self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_end
                ] *= density_weight[-1]
                precond = sum_pin_weights_in_nodes + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[0 : self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)

            ### stop gradients for terminated electric field
            if update_mask is not None:
                grad = grad.view(2, -1)
                update_mask = ~update_mask
                movable_mask = update_mask[self.movablenode2fence_region_map_clamp]
                filler_mask = update_mask[self.filler2fence_region_map]
                grad[0, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[1, : self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                grad[0, self.placedb.num_nodes - self.placedb.num_filler_nodes :].masked_fill_(filler_mask, 0)
                grad[1, self.placedb.num_nodes - self.placedb.num_filler_nodes :].masked_fill_(filler_mask, 0)
                grad = grad.view(-1)
            if fix_nodes_mask is not None:
                grad = grad.view(2, -1)
                grad[0, :self.placedb.num_movable_nodes].masked_fill_(fix_nodes_mask[:self.placedb.num_movable_nodes], 0)
                grad[1, :self.placedb.num_movable_nodes].masked_fill_(fix_nodes_mask[:self.placedb.num_movable_nodes], 0)
                grad = grad.view(-1)
            self.iteration += 1

            # only work in benchmarks without fence region, assume overflow has been updated
            if len(self.placedb.regions) > 0 and self.overflows and self.overflows[-1].max() < 0.3 and self.alpha < 1024:
                if (self.iteration % 20) == 0:
                    self.alpha *= 2
                    logging.info(
                        "preconditioning alpha = %g, best_overflow %g, overflow %g"
                        % (self.alpha, self.best_overflow, self.overflows[-1])
                    )

        return grad


class PlaceObj(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    """
    def __init__(self, density_weight, params, placedb, data_collections,
                 op_collections, global_place_params):
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param op_collections a collection of all ops
        @param global_place_params global placement parameters for current global placement stage
        """
        super(PlaceObj, self).__init__()

        ### quadratic penalty
        self.density_quad_coeff = 2000
        self.init_density = None
        ### increase density penalty if slow convergence
        self.density_factor = 1

        if(len(placedb.regions) > 0):
            ### fence region will enable quadratic penalty by default
            self.quad_penalty = True
        else:
            ### non fence region will use first-order density penalty by default
            self.quad_penalty = False

        ### fence region
        ### update mask controls whether stop gradient/updating, 1 represents allow grad/update
        self.update_mask = None
        self.fix_nodes_mask = None 
        if len(placedb.regions) > 0:
            ### for subregion rough legalization, once stop updating, perform immediate greddy legalization once
            ### this is to avoid repeated legalization
            ### 1 represents already legal
            self.legal_mask = torch.zeros(len(placedb.regions) + 1)

        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.global_place_params = global_place_params

        self.gpu = params.gpu
        self.data_collections = data_collections
        self.op_collections = op_collections
        if len(placedb.regions) > 0:
            ### different fence region needs different density weights in multi-electric field algorithm
            self.density_weight = torch.tensor(
                [density_weight]*(len(placedb.regions)+1),
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)
        else:
            self.density_weight = torch.tensor(
                [density_weight],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)
        ### Note: even for multi-electric fields, they use the same gamma
        num_bins_x = global_place_params["num_bins_x"] if "num_bins_x" in global_place_params and global_place_params["num_bins_x"] > 1 else placedb.num_bins_x
        num_bins_y = global_place_params["num_bins_y"] if "num_bins_y" in global_place_params and global_place_params["num_bins_y"] > 1 else placedb.num_bins_y
        name = "Global placement: %dx%d bins by default" % (num_bins_x, num_bins_y)
        logging.info(name)
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        self.bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        self.gamma = torch.tensor(10 * self.base_gamma(params, placedb),
                                  dtype=self.data_collections.pos[0].dtype,
                                  device=self.data_collections.pos[0].device)

        # compute weighted average wirelength from position

        name = "%dx%d bins" % (num_bins_x, num_bins_y)
        self.name = name

        if global_place_params["wirelength"] == "weighted_average":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_weighted_average_wl(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op)
        elif global_place_params["wirelength"] == "logsumexp":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_logsumexp_wl(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op)
        else:
            assert 0, "unknown wirelength model %s" % (
                global_place_params["wirelength"])

        self.op_collections.density_overflow_op = self.build_electric_overflow(
            params,
            placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y)

        self.op_collections.density_op = self.build_electric_potential(
            params,
            placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=name)
        ### build multiple density op for multi-electric field
        if len(self.placedb.regions) > 0:
            self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op, self.op_collections.fence_region_density_overflow_merged_op = self.build_multi_fence_region_density_op()
        self.op_collections.update_density_weight_op = self.build_update_density_weight(
            params, placedb)
        self.op_collections.precondition_op = self.build_precondition(
            params, placedb, self.data_collections, self.op_collections)
        self.op_collections.noise_op = self.build_noise(
            params, placedb, self.data_collections)
        if params.routability_opt_flag:
            # compute congestion map, RISA/RUDY congestion map
            self.op_collections.route_utilization_map_op = self.build_route_utilization_map(
                params, placedb, self.data_collections)
            self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(
                params, placedb, self.data_collections)
            # self.op_collections.nctugr_congestion_map_op = self.build_nctugr_congestion_map(
            #     params, placedb, self.data_collections)
            # adjust instance area with congestion map
            self.op_collections.adjust_node_area_op = self.build_adjust_node_area(
                params, placedb, self.data_collections)
        if  params.enable_congestion_optimization:
            self._init_congestion_optimization(params, placedb, data_collections)

        # GiFt initialization 
        if params.global_place_flag and params.gift_init_flag: 
            self.op_collections.gift_init_op = gift_init.GiFtInit(
                    flat_netpin=self.data_collections.flat_net2pin_map, 
                    netpin_start=self.data_collections.flat_net2pin_start_map, 
                    pin2node_map=self.data_collections.pin2node_map, 
                    net_weights=self.data_collections.net_weights, 
                    net_mask=self.data_collections.net_mask_ignore_large_degrees, 
                    xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh,
                    num_nodes=placedb.num_physical_nodes,  
                    num_movable_nodes=placedb.num_movable_nodes, 
                    scale=params.gift_init_scale
                    ) 

        self.Lgamma_iteration = global_place_params["iteration"]
        if 'Llambda_density_weight_iteration' in global_place_params:
            self.Llambda_density_weight_iteration = global_place_params[
                'Llambda_density_weight_iteration']
        else:
            self.Llambda_density_weight_iteration = 1
        if 'Lsub_iteration' in global_place_params:
            self.Lsub_iteration = global_place_params['Lsub_iteration']
        else:
            self.Lsub_iteration = 1
        if 'routability_Lsub_iteration' in global_place_params:
            self.routability_Lsub_iteration = global_place_params[
                'routability_Lsub_iteration']
        else:
            self.routability_Lsub_iteration = self.Lsub_iteration
        self.start_fence_region_density = False

    def _init_congestion_optimization(self, params, placedb, data_collections):
        """
        Initialize congestion optimization operators and state.
        
        This method sets up:
        1. RUDY congestion map operator
        2. Poisson solver for computing potential field from congestion
        3. Congestion gradient operator
        """
        logging.info("[CongestionOpt] Initializing congestion optimization with Poisson solver...")
        
        # Store reference for later use
        self.params = params
        
        # Get routing grid dimensions
        self.cong_num_bins_x = placedb.num_routing_grids_x
        self.cong_num_bins_y = placedb.num_routing_grids_y
        self.cong_bin_size_x = (placedb.xh - placedb.xl) / self.cong_num_bins_x
        self.cong_bin_size_y = (placedb.yh - placedb.yl) / self.cong_num_bins_y
        
        # Build congestion map operator (uses RUDY)
        self.op_collections.congestion_map_op = self.build_congestion_map(
            params, placedb, data_collections
        )
        
        # Precompute Poisson solver eigenvalues for DCT
        self._init_poisson_solver(placedb, data_collections)
        
        # Build congestion gradient operator
        self.op_collections.congestion_grad_op = self.build_congestion_grad(
            params, placedb, data_collections
        )
        
        # Cached data for periodic updates
        self.cached_congestion_map = None
        self.cached_potential_field = None
        self.cached_field_grad_x = None
        self.cached_field_grad_y = None
        self.congestion_update_counter = 0
        
        logging.info(f"[CongestionOpt] Initialized with Poisson solver: "
                     f"bins={self.cong_num_bins_x}x{self.cong_num_bins_y}, "
                     f"weight={params.congestion_weight}, "
                     f"update_interval={params.congestion_update_interval}, "
                     f"threshold={params.congestion_threshold}")

    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        if torch.isnan(pos).any():  
            logging.error("Invalid positions detected in obj_fn")  
            return torch.tensor(float('inf'), device=pos.device, requires_grad=True)  
        
        self.wirelength = self.op_collections.wirelength_op(pos)  
        if len(self.placedb.regions) > 0:  
            self.density = self.op_collections.fence_region_density_merged_op(pos)  
        else:  
            self.density = self.op_collections.density_op(pos) 
            
        self.wirelength = self.op_collections.wirelength_op(pos)
        if len(self.placedb.regions) > 0:
            self.density = self.op_collections.fence_region_density_merged_op(pos)
        else:
            self.density = self.op_collections.density_op(pos)

        if self.init_density is None:
            ### record initial density
            self.init_density = self.density.data.clone()
            ### density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(self.init_density > 0, 1 /self.init_density[self.init_density > 0])
            self.quad_penalty_coeff = self.density_quad_coeff / 2 * self.density_weight_grad_precond
        if self.quad_penalty:
            ### quadratic density penalty
            self.density = self.density * (1 + self.quad_penalty_coeff * self.density)
        if len(self.placedb.regions) > 0:
            result = self.wirelength + self.density_weight.dot(self.density)
        else:
            result = torch.add(self.wirelength, self.density, alpha=(self.density_factor * self.density_weight).item())

        return result

    def obj_and_grad_fn_old(self, pos_w, pos_g=None, admm_multiplier=None):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        if not self.start_fence_region_density:
            obj = self.obj_fn(pos_w, pos_g, admm_multiplier)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            obj.backward()
        else:
            num_nodes = self.placedb.num_nodes
            num_movable_nodes = self.placedb.num_movable_nodes
            num_filler_nodes = self.placedb.num_filler_nodes


            wl = self.op_collections.wirelength_op(pos_w)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            wl.backward()
            wl_grad = pos_w.grad.data.clone()
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if self.init_density is None:
                self.init_density = self.op_collections.density_op(pos_w.data).data.item()

            if self.quad_penalty:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)
                inner_density = inner_density + self.density_quad_coeff / 2 / self.init_density  * inner_density**2
            else:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)

            inner_density.backward()
            inner_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map > 1e3
            inner_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes:num_nodes+num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes-num_filler_nodes:num_nodes].mul_(0.5)
            inner_density_grad[-num_filler_nodes:].mul_(0.5)
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if self.quad_penalty:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)
                outer_density = outer_density + self.density_quad_coeff / 2 / self.init_density  * outer_density ** 2
            else:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)

            outer_density.backward()
            outer_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map < 1e3
            outer_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes:num_nodes+num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes-num_filler_nodes:num_nodes].mul_(0.5)
            outer_density_grad[-num_filler_nodes:].mul_(0.5)

            if self.quad_penalty:
                density = self.op_collections.density_op(pos_w.data)
                obj = wl.data.item() + self.density_weight * (density + self.density_quad_coeff / 2 / self.init_density * density ** 2)
            else:
                obj = wl.data.item() + self.density_weight * self.op_collections.density_op(pos_w.data)

            pos_w.grad.data.copy_(wl_grad + self.density_weight * (inner_density_grad + outer_density_grad))


        self.op_collections.precondition_op(pos_w.grad, self.density_weight, 0)

        return obj, pos_w.grad

    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        #self.check_gradient(pos)
        if pos.grad is not None:
            pos.grad.zero_()
        obj = self.obj_fn(pos)

        if obj.requires_grad:
          obj.backward()

        if self.params.enable_congestion_optimization and HAS_CONGESTION_GRAD:
            self._apply_congestion_gradient(pos)

        self.op_collections.precondition_op(pos.grad, self.density_weight, self.update_mask, self.fix_nodes_mask)

        return obj, pos.grad

    def _init_poisson_solver(self, placedb, data_collections):
        """
        Initialize eigenvalues for DCT-based Poisson solver.
        
        For Poisson equation: ∇²ψ = -ρ
        Using DCT: ψ_mn = ρ_mn / (λ_m + λ_n)
        where λ_k = 2(1 - cos(πk/N)) / h² (h = bin_size)
        """
        device = data_collections.pos[0].device
        dtype = data_collections.pos[0].dtype
        
        num_bins_x = self.cong_num_bins_x
        num_bins_y = self.cong_num_bins_y
        
        # Compute eigenvalues for x and y directions
        # λ_k = 2(1 - cos(πk/N)) / h²
        m = torch.arange(num_bins_x, device=device, dtype=dtype)
        n = torch.arange(num_bins_y, device=device, dtype=dtype)
        
        # Scale by bin size for proper physical dimensions
        hx = self.cong_bin_size_x
        hy = self.cong_bin_size_y
        
        lambda_m = 2 * (1 - torch.cos(torch.pi * m / num_bins_x)) / (hx * hx)
        lambda_n = 2 * (1 - torch.cos(torch.pi * n / num_bins_y)) / (hy * hy)
        
        # Create 2D eigenvalue matrix [num_bins_x, num_bins_y]
        self.poisson_eigenvalues = lambda_m.unsqueeze(1) + lambda_n.unsqueeze(0)
        
        # Handle DC component (λ_00 = 0) to avoid division by zero
        # Set to 1 since we'll set ψ_00 = 0 explicitly
        self.poisson_eigenvalues[0, 0] = 1.0
        
        logging.info(f"[CongestionOpt] Poisson solver eigenvalues initialized: "
                     f"max={self.poisson_eigenvalues.max():.4f}, "
                     f"min(non-zero)={self.poisson_eigenvalues[1:,1:].min():.4f}")

    def _apply_congestion_gradient(self, pos):
        """
        Apply congestion gradient to pos.grad using Poisson-based potential field.
        
        Pipeline:
        1. Compute RUDY congestion map (periodic update)
        2. Solve Poisson equation: ∇²ψ = -ρ (congestion as charge density)
        3. Compute electric field: E = -∇ψ
        4. Use E as field gradient for congestion_grad_op
        5. Compute and apply congestion gradient
        """
        self.congestion_update_counter += 1
        
        # Update congestion map and potential field periodically
        if (self.congestion_update_counter >= self.params.congestion_update_interval or
            self.cached_congestion_map is None):
            
            # Step 1: Compute RUDY congestion map
            with torch.no_grad():
                self.cached_congestion_map = self.op_collections.congestion_map_op(pos)
            
            # Step 2: Solve Poisson equation for potential field
            self.cached_potential_field = self._solve_poisson_equation(
                self.cached_congestion_map
            )
            
            # Step 3: Compute electric field (negative gradient of potential)
            self.cached_field_grad_x, self.cached_field_grad_y = \
                self._compute_electric_field(self.cached_potential_field)
            
            # Update field maps in congestion_grad_op
            if self.op_collections.congestion_grad_op is not None:
                self.op_collections.congestion_grad_op.set_field_maps(
                    self.cached_field_grad_x, 
                    self.cached_field_grad_y
                )
            
            self.congestion_update_counter = 0
            
            # Debug logging
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[CongestionOpt] Updated potential field: "
                             f"max_cong={self.cached_congestion_map.max():.4f}, "
                             f"max_potential={self.cached_potential_field.max():.4f}, "
                             f"max_field={max(self.cached_field_grad_x.abs().max(), self.cached_field_grad_y.abs().max()):.4f}")
        
        # Step 4: Compute congestion gradient using virtual cell method
        if self.op_collections.congestion_grad_op is not None:
            with torch.no_grad():
                cong_grad = self.op_collections.congestion_grad_op.compute_grad_direct(
                    pos, self.cached_congestion_map
                )
            
            # Step 5: Compute effective weight and apply gradient
            effective_weight = self._compute_effective_congestion_weight(pos)
            pos.grad.data.add_(effective_weight * cong_grad)

    def _solve_poisson_equation(self, congestion_map):
        """
        Solve Poisson equation ∇²ψ = -ρ using DCT.
        
        Args:
            congestion_map: [num_bins_x, num_bins_y] congestion values (ρ)
            
        Returns:
            potential_field: [num_bins_x, num_bins_y] potential values (ψ)
            
        Algorithm:
        1. DCT-II transform of ρ
        2. Divide by eigenvalues: ψ_mn = ρ_mn / (λ_m + λ_n)
        3. Set DC component to 0
        4. IDCT-II to get ψ
        """
        device = congestion_map.device
        dtype = congestion_map.dtype
        
        # Ensure correct shape
        if congestion_map.dim() != 2:
            raise ValueError(f"[CongestionOpt] congestion_map must be 2D, got shape {congestion_map.shape}")
        
        # Use negative ρ as source term (∇²ψ = -ρ)
        rho = -congestion_map
        
        # Apply 2D DCT using custom implementation
        # DCT-II: X_k = Σ x_n * cos(π*k*(2n+1)/(2N))
        rho_dct = self._dct2d(rho)
        
        # Solve in frequency domain: ψ_mn = ρ_mn / (λ_m + λ_n)
        psi_dct = rho_dct / self.poisson_eigenvalues
        
        # Set DC component to 0 (mean potential is arbitrary)
        psi_dct[0, 0] = 0.0
        
        # Apply inverse 2D DCT
        potential_field = self._idct2d(psi_dct)
        
        return potential_field

    def _dct2d(self, x):
        """
        2D Type-II DCT using FFT.
        
        DCT-II is related to FFT by:
        DCT(x) = Re(FFT(x_extended) * exp(-i*π*k/(2N)))
        
        For efficiency, we use the relationship:
        DCT-II = Re(FFT([x_0, x_1, ..., x_{N-1}, x_{N-1}, ..., x_1]))
        with appropriate normalization.
        """
        N1, N2 = x.shape
        device = x.device
        dtype = x.dtype
        
        # Method: Use FFT on mirrored sequence
        # Create symmetric extension: [x_0, ..., x_{N-1}, x_{N-1}, ..., x_1]
        
        # For x direction
        x_ext = torch.zeros(2 * N1, N2, dtype=dtype, device=device)
        x_ext[:N1, :] = x
        x_ext[N1:, :] = torch.flip(x, dims=[0])
        
        # FFT along first dimension
        x_fft = torch.fft.fft(x_ext, dim=0)
        
        # Extract DCT coefficients (real part with phase shift)
        k1 = torch.arange(N1, device=device, dtype=dtype)
        phase1 = torch.exp(-1j * torch.pi * k1 / (2 * N1)).unsqueeze(1)
        x_dct_1d = (x_fft[:N1, :] * phase1).real
        
        # For y direction on the result
        x_ext2 = torch.zeros(N1, 2 * N2, dtype=dtype, device=device)
        x_ext2[:, :N2] = x_dct_1d
        x_ext2[:, N2:] = torch.flip(x_dct_1d, dims=[1])
        
        # FFT along second dimension
        x_fft2 = torch.fft.fft(x_ext2, dim=1)
        
        # Extract DCT coefficients
        k2 = torch.arange(N2, device=device, dtype=dtype)
        phase2 = torch.exp(-1j * torch.pi * k2 / (2 * N2)).unsqueeze(0)
        x_dct = (x_fft2[:, :N2] * phase2).real
        
        # Normalize (ortho normalization)
        x_dct = x_dct * 2 / (N1 * N2)
        x_dct[0, :] = x_dct[0, :] / torch.sqrt(torch.tensor(2.0, device=device))
        x_dct[:, 0] = x_dct[:, 0] / torch.sqrt(torch.tensor(2.0, device=device))
        
        return x_dct

    def _idct2d(self, X):
        """
        2D Type-II Inverse DCT using FFT.
        
        IDCT-II is the inverse of DCT-II.
        """
        N1, N2 = X.shape
        device = X.device
        dtype = X.dtype
        
        # Undo normalization
        X_scaled = X.clone()
        X_scaled[0, :] = X_scaled[0, :] * torch.sqrt(torch.tensor(2.0, device=device))
        X_scaled[:, 0] = X_scaled[:, 0] * torch.sqrt(torch.tensor(2.0, device=device))
        X_scaled = X_scaled * (N1 * N2) / 2
        
        # For x direction: apply phase and IFFT
        k1 = torch.arange(N1, device=device, dtype=dtype)
        phase1 = torch.exp(1j * torch.pi * k1 / (2 * N1)).unsqueeze(1)
        
        # Extend to full FFT size
        X_ext = torch.zeros(2 * N1, N2, dtype=torch.complex64, device=device)
        X_ext[:N1, :] = X_scaled * phase1
        X_ext[N1 + 1:, :] = torch.flip(X_ext[1:N1, :], dims=[0]).conj()
        
        x_ifft1 = torch.fft.ifft(X_ext, dim=0).real[:N1, :]
        
        # For y direction
        k2 = torch.arange(N2, device=device, dtype=dtype)
        phase2 = torch.exp(1j * torch.pi * k2 / (2 * N2)).unsqueeze(0)
        
        X_ext2 = torch.zeros(N1, 2 * N2, dtype=torch.complex64, device=device)
        X_ext2[:, :N2] = x_ifft1.to(torch.complex64) * phase2
        X_ext2[:, N2 + 1:] = torch.flip(X_ext2[:, 1:N2], dims=[1]).conj()
        
        x_idct = torch.fft.ifft(X_ext2, dim=1).real[:, :N2]
        
        return x_idct.to(dtype)

    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.data_collections.pos[0])

    def check_gradient(self, pos):
        """
        @brief check gradient for debug
        @param pos locations of cells
        """
        wirelength = self.op_collections.wirelength_op(pos)

        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        pos.grad.zero_()
        density = self.density_weight * self.op_collections.density_op(pos)
        density.backward()
        density_grad = pos.grad.clone()

        wirelength_grad_norm = wirelength_grad.norm(p=1)
        density_grad_norm = density_grad.norm(p=1)

        logging.info("wirelength_grad norm = %.6E" % (wirelength_grad_norm))
        logging.info("density_grad norm    = %.6E" % (density_grad_norm))
        pos.grad.zero_()

    def estimate_initial_learning_rate(self, x_k, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        obj_k, g_k = self.obj_and_grad_fn(x_k)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1 = self.obj_and_grad_fn(x_k_1)

        return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)

    def build_weighted_average_wl(self, params, placedb, data_collections,
                                  pin_pos_op):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        # use WeightedAverageWirelength atomic
        wirelength_for_pin_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_logsumexp_wl(self, params, placedb, data_collections,
                           pin_pos_op):
        """
        @brief build the op to compute log-sum-exp wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        wirelength_for_pin_op = logsumexp_wirelength.LogSumExpWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_density_overflow(self, params, placedb, data_collections,
                               num_bins_x, num_bins_y):
        """
        @brief compute density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return density_overflow.DensityOverflow(
            data_collections.node_size_x,
            data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0)

    def build_electric_overflow(self, params, placedb, data_collections,
                                num_bins_x, num_bins_y):
        """
        @brief compute electric density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return electric_overflow.ElectricOverflow(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask)

    def build_density_potential(self, params, placedb, data_collections,
                                num_bins_x, num_bins_y, padding, name):
        """
        @brief NTUPlace3 density potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param padding number of padding bins to left, right, bottom, top of the placement region
        @param name string for printing
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        xl = placedb.xl - padding * bin_size_x
        xh = placedb.xh + padding * bin_size_x
        yl = placedb.yl - padding * bin_size_y
        yh = placedb.yh + padding * bin_size_y
        local_num_bins_x = num_bins_x + 2 * padding
        local_num_bins_y = num_bins_y + 2 * padding
        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x) + 4 * bin_size_x) / bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y) + 4 * bin_size_y) / bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, local_num_bins_x, local_num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, padding))
        if local_num_bins_x < max_num_bins:
            logging.warning("local_num_bins_x (%d) < max_num_bins (%d)" %
                            (local_num_bins_x, max_num_bins))
        if local_num_bins_y < max_num_bins:
            logging.warning("local_num_bins_y (%d) < max_num_bins (%d)" %
                            (local_num_bins_y, max_num_bins))

        node_size_x = placedb.node_size_x
        node_size_y = placedb.node_size_y

        # coefficients
        ax = (4 / (node_size_x + 2 * bin_size_x) /
              (node_size_x + 4 * bin_size_x)).astype(placedb.dtype).reshape(
                  [placedb.num_nodes, 1])
        bx = (2 / bin_size_x / (node_size_x + 4 * bin_size_x)).astype(
            placedb.dtype).reshape([placedb.num_nodes, 1])
        ay = (4 / (node_size_y + 2 * bin_size_y) /
              (node_size_y + 4 * bin_size_y)).astype(placedb.dtype).reshape(
                  [placedb.num_nodes, 1])
        by = (2 / bin_size_y / (node_size_y + 4 * bin_size_y)).astype(
            placedb.dtype).reshape([placedb.num_nodes, 1])

        # bell shape overlap function
        def npfx1(dist):
            # ax will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return 1.0 - ax.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfx2(dist):
            # bx will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return bx.reshape([
                placedb.num_nodes, 1
            ]) * np.square(dist - node_size_x / 2 - 2 * bin_size_x).reshape(
                [placedb.num_nodes, 1])

        def npfy1(dist):
            # ay will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return 1.0 - ay.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfy2(dist):
            # by will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return by.reshape([
                placedb.num_nodes, 1
            ]) * np.square(dist - node_size_y / 2 - 2 * bin_size_y).reshape(
                [placedb.num_nodes, 1])

        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_x = npfx1(0) + 2 * npfx1(bin_size_x) + 2 * npfx2(
            2 * bin_size_x)
        cx = (node_size_x.reshape([placedb.num_nodes, 1]) /
              integral_potential_x).reshape([placedb.num_nodes, 1])
        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_y = npfy1(0) + 2 * npfy1(bin_size_y) + 2 * npfy2(
            2 * bin_size_y)
        cy = (node_size_y.reshape([placedb.num_nodes, 1]) /
              integral_potential_y).reshape([placedb.num_nodes, 1])

        return density_potential.DensityPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            ax=torch.tensor(ax.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            bx=torch.tensor(bx.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            cx=torch.tensor(cx.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            ay=torch.tensor(ay.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            by=torch.tensor(by.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            cy=torch.tensor(cy.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            bin_center_x=data_collections.bin_center_x_padded(placedb, padding, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, padding, num_bins_y),
            target_density=data_collections.target_density,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            padding=padding,
            sigma=(1.0 / 16) * placedb.width / bin_size_x,
            delta=2.0)

    def build_electric_potential(self, params, placedb, data_collections,
                                 num_bins_x, num_bins_y, name, region_id=None, fence_regions=None):
        """
        @brief e-place electrostatic potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param name string for printing
        @param fence_regions a [n_subregions, 4] tensor for fence regions potential penalty
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x[0:placedb.num_movable_nodes]) +
             2 * bin_size_x) / bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y[0:placedb.num_movable_nodes]) +
             2 * bin_size_y) / bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, num_bins_x, num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, 0))
        if num_bins_x < max_num_bins:
            logging.warning("num_bins_x (%d) < max_num_bins (%d)" %
                            (num_bins_x, max_num_bins))
        if num_bins_y < max_num_bins:
            logging.warning("num_bins_y (%d) < max_num_bins (%d)" %
                            (num_bins_y, max_num_bins))
        #### for fence region, the target density is different from different regions
        target_density = data_collections.target_density.item() if fence_regions is None else placedb.target_density_fence_region[region_id]
        return electric_potential.ElectricPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask,
            fast_mode=params.RePlAce_skip_energy_flag,
            region_id=region_id,
            fence_regions=fence_regions,
            node2fence_region_map=data_collections.node2fence_region_map,
            placedb=placedb)

    def initialize_density_weight(self, params, placedb):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        """
        wirelength = self.op_collections.wirelength_op(
            self.data_collections.pos[0])
        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        self.data_collections.pos[0].grad.zero_()

        if len(self.placedb.regions) > 0:
            density_list = []
            density_grad_list = []
            for density_op in self.op_collections.fence_region_density_ops:
                density_i = density_op(self.data_collections.pos[0])
                density_list.append(density_i.data.clone())
                density_i.backward()
                density_grad_list.append(self.data_collections.pos[0].grad.data.clone())
                self.data_collections.pos[0].grad.zero_()

            ### record initial density
            self.init_density = torch.stack(density_list)
            ### density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(self.init_density > 0, 1/self.init_density[self.init_density > 0])
            ### compute u
            self.density_weight_u = self.init_density * self.density_weight_grad_precond
            self.density_weight_u += 0.5 * self.density_quad_coeff * self.density_weight_u ** 2
            ### compute s
            density_weight_s = 1 + self.density_quad_coeff * self.init_density * self.density_weight_grad_precond
            ### compute density grad L1 norm
            density_grad_norm = sum(self.density_weight_u[i] * density_weight_s[i] * density_grad_list[i].norm(p=1) for i in range(density_weight_s.size(0)))

            self.density_weight_u *= params.density_weight * wirelength_grad_norm / density_grad_norm
            ### set initial step size for density weight update
            self.density_weight_step_size_inc_low = 1.03
            self.density_weight_step_size_inc_high = 1.04
            self.density_weight_step_size = (self.density_weight_step_size_inc_low - 1) * self.density_weight_u.norm(p=2)
            ### commit initial density weight
            self.density_weight = self.density_weight_u * density_weight_s

        else:
            density = self.op_collections.density_op(self.data_collections.pos[0])
            ### record initial density
            self.init_density = density.data.clone()
            density.backward()
            density_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

            grad_norm_ratio = wirelength_grad_norm / density_grad_norm
            self.density_weight = torch.tensor(
                [params.density_weight * grad_norm_ratio],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)

        return self.density_weight

    def build_update_density_weight(self, params, placedb, algo="overflow"):
        """
        @brief update density weight
        @param params parameters
        @param placedb placement database
        """
        ### params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF
        ### params for overflow mode from elfPlace
        assert algo in {"hpwl", "overflow"}, logging.error("density weight update not supports hpwl mode or overflow mode")

        def update_density_weight_op_hpwl(cur_metric, prev_metric, iteration):
            ### based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(
                        np.power(0.9999, float(iteration)), 0.98)
                else:
                    mu = UPPER_PCOF * torch.pow(
                        UPPER_PCOF, -delta_hpwl / ref_hpwl).clamp(
                            min=LOWER_PCOF, max=UPPER_PCOF)
                self.density_weight *= mu

        def update_density_weight_op_overflow(cur_metric, prev_metric, iteration):
            assert self.quad_penalty == True, logging.error("density weight update based on overflow only works for quadratic density penalty")
            ### based on overflow
            ### stop updating if a region has lower overflow than stop overflow
            with torch.no_grad():
                density_norm = cur_metric.density * self.density_weight_grad_precond
                density_weight_grad = density_norm + self.density_quad_coeff / 2 * density_norm ** 2
                density_weight_grad /= density_weight_grad.norm(p=2)

                self.density_weight_u += self.density_weight_step_size * density_weight_grad
                density_weight_s = 1 + self.density_quad_coeff * density_norm

                density_weight_new = (self.density_weight_u * density_weight_s).clamp(max=10)

                ### conditional update if this region's overflow is higher than stop overflow
                if(self.update_mask is None):
                    self.update_mask = cur_metric.overflow >= self.params.stop_overflow
                else:
                    ### restart updating is not allowed
                    self.update_mask &= cur_metric.overflow >= self.params.stop_overflow
                self.density_weight.masked_scatter_(self.update_mask, density_weight_new[self.update_mask])

                ### update density weight step size
                rate = torch.log(self.density_quad_coeff * density_norm.norm(p=2)).clamp(min=0)
                rate = rate / (1 + rate)
                rate = rate * (self.density_weight_step_size_inc_high - self.density_weight_step_size_inc_low) + self.density_weight_step_size_inc_low
                self.density_weight_step_size *= rate

        if not self.quad_penalty and algo == "overflow":
            logging.warning("quadratic density penalty is disabled, density weight update is forced to be based on HPWL")
            algo = "hpwl"
        if len(self.placedb.regions) == 0 and algo == "overflow":
            logging.warning("for benchmark without fence region, density weight update is forced to be based on HPWL")
            algo = "hpwl"

        update_density_weight_op = {"hpwl":update_density_weight_op_hpwl,
                                    "overflow": update_density_weight_op_overflow}[algo]

        return update_density_weight_op

    def base_gamma(self, params, placedb):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        return params.gamma * (self.bin_size_x + self.bin_size_y)

    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        ### overflow can have multiple values for fence regions, use their weighted average based on movable node number
        if overflow.numel() == 1:
            overflow_avg = overflow
        else:
            overflow_avg = overflow
        coef = torch.pow(10, (overflow_avg - 0.1) * 20 / 9 - 1)
        self.gamma.data.fill_((base_gamma * coef).item())
        return True

    def build_noise(self, params, placedb, data_collections):
        """
        @brief add noise to cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        node_size = torch.cat(
            [data_collections.node_size_x, data_collections.node_size_y],
            dim=0).to(data_collections.pos[0].device)

        def noise_op(pos, noise_ratio):
            with torch.no_grad():
                noise = torch.rand_like(pos)
                noise.sub_(0.5).mul_(node_size).mul_(noise_ratio)
                # no noise to fixed cells
                if self.fix_nodes_mask is not None:
                    noise = noise.view(2, -1)
                    noise[0, :placedb.num_movable_nodes].masked_fill_(self.fix_nodes_mask[:placedb.num_movable_nodes], 0)
                    noise[1, :placedb.num_movable_nodes].masked_fill_(self.fix_nodes_mask[:placedb.num_movable_nodes], 0)
                    noise = noise.view(-1)
                noise[placedb.num_movable_nodes:placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                noise[placedb.num_nodes +
                      placedb.num_movable_nodes:2 * placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                return pos.add_(noise)

        return noise_op

    def build_precondition(self, params, placedb,
                           data_collections, op_collections):
        """
        @brief preconditioning to gradient
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param op_collections a collection of all ops
        """
        return PreconditionOp(placedb, data_collections, op_collections)

    def build_route_utilization_map(self, params, placedb, data_collections):
        """
        @brief routing congestion map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        congestion_op = rudy.Rudy(
            netpin_start=data_collections.flat_net2pin_start_map,
            flat_netpin=data_collections.flat_net2pin_map,
            net_weights=data_collections.net_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
            unit_vertical_capacity=placedb.unit_vertical_capacity,
            initial_horizontal_utilization_map=data_collections.
            initial_horizontal_utilization_map,
            initial_vertical_utilization_map=data_collections.
            initial_vertical_utilization_map,
            deterministic_flag=params.deterministic_flag)

        def route_utilization_map_op(pos):
            pin_pos = self.op_collections.pin_pos_op(pos)
            return congestion_op(pin_pos)

        return route_utilization_map_op

    def build_pin_utilization_map(self, params, placedb, data_collections):
        """
        @brief pin density map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        return pin_utilization.PinUtilization(
            pin_weights=data_collections.pin_weights,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            pin_stretch_ratio=params.pin_stretch_ratio,
            deterministic_flag=params.deterministic_flag)

    def build_nctugr_congestion_map(self, params, placedb, data_collections):
        """
        @brief call NCTUgr for congestion estimation
        """
        path = "%s/%s" % (params.result_dir, params.design_name())
        return nctugr_binary.NCTUgr(
            aux_input_file=os.path.realpath(params.aux_input),
            param_setting_file="%s/../thirdparty/NCTUgr.ICCAD2012/DAC12.set" %
            (os.path.dirname(os.path.realpath(__file__))),
            tmp_pl_file="%s/%s.NCTUgr.pl" %
            (os.path.realpath(path), params.design_name()),
            tmp_output_file="%s/%s.NCTUgr" %
            (os.path.realpath(path), params.design_name()),
            horizontal_routing_capacities=torch.from_numpy(
                placedb.unit_horizontal_capacities *
                placedb.routing_grid_size_y),
            vertical_routing_capacities=torch.from_numpy(
                placedb.unit_vertical_capacities *
                placedb.routing_grid_size_x),
            params=params,
            placedb=placedb)

    def build_adjust_node_area(self, params, placedb, data_collections):
        """
        @brief adjust cell area according to routing congestion and pin utilization map
        """
        total_movable_area = (
            data_collections.node_size_x[:placedb.num_movable_nodes] *
            data_collections.node_size_y[:placedb.num_movable_nodes]).sum()
        total_filler_area = (
            data_collections.node_size_x[-placedb.num_filler_nodes:] *
            data_collections.node_size_y[-placedb.num_filler_nodes:]).sum()
        total_place_area = (total_movable_area + total_filler_area
                            ) / data_collections.target_density
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin_weights=data_collections.pin_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            route_num_bins_x=placedb.num_routing_grids_x,
            route_num_bins_y=placedb.num_routing_grids_y,
            pin_num_bins_x=placedb.num_routing_grids_x,
            pin_num_bins_y=placedb.num_routing_grids_y,
            total_place_area=total_place_area,
            total_whitespace_area=total_place_area - total_movable_area,
            max_route_opt_adjust_rate=params.max_route_opt_adjust_rate,
            route_opt_adjust_exponent=params.route_opt_adjust_exponent,
            max_pin_opt_adjust_rate=params.max_pin_opt_adjust_rate,
            area_adjust_stop_ratio=params.area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=params.route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=params.pin_area_adjust_stop_ratio,
            unit_pin_capacity=data_collections.unit_pin_capacity)

        def build_adjust_node_area_op(pos, route_utilization_map,
                                      pin_utilization_map):
            return adjust_node_area_op(
                pos, data_collections.node_size_x,
                data_collections.node_size_y, data_collections.pin_offset_x,
                data_collections.pin_offset_y, data_collections.target_density,
                route_utilization_map, pin_utilization_map)

        return build_adjust_node_area_op

    def build_fence_region_density_op(self, fence_region_list, node2fence_region_map):
        assert type(fence_region_list) == list and len(fence_region_list) == 2, "Unsupported fence region list"
        self.data_collections.node2fence_region_map = torch.from_numpy(self.placedb.node2fence_region_map[:self.placedb.num_movable_nodes]).to(fence_region_list[0].device)
        self.op_collections.inner_fence_region_density_op = self.build_electric_potential(
            self.params,
            self.placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=self.name,
            fence_regions=fence_region_list[0],
            fence_region_mask=self.data_collections.node2fence_region_map>1e3) # density penalty for inner cells
        self.op_collections.outer_fence_region_density_op = self.build_electric_potential(
            self.params,
            self.placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=self.name,
            fence_regions = fence_region_list[1],
            fence_region_mask=self.data_collections.node2fence_region_map<1e3) # density penalty for outer cells

    def build_multi_fence_region_density_op(self):
        # region 0, ..., region n, non_fence_region
        self.op_collections.fence_region_density_ops = []

        for i, fence_region in enumerate(self.data_collections.virtual_macro_fence_region[:-1]):
            self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        name=self.name,
                        region_id=i,
                        fence_regions=fence_region)
            )

        self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        name=self.name,
                        region_id=len(self.placedb.regions),
                        fence_regions=self.data_collections.virtual_macro_fence_region[-1])
        )
        def merged_density_op(pos):
            ### stop mask is to stop forward of density
            ### 1 represents stop flag
            res = torch.stack([density_op(pos, mode="density") for density_op in self.op_collections.fence_region_density_ops])
            return res

        def merged_density_overflow_op(pos):
            ### stop mask is to stop forward of density
            ### 1 represents stop flag
            overflow_list, max_density_list = [], []
            for density_op in self.op_collections.fence_region_density_ops:
                overflow, max_density = density_op(pos, mode="overflow")
                overflow_list.append(overflow)
                max_density_list.append(max_density)
            overflow_list, max_density_list = torch.stack(overflow_list), torch.stack(max_density_list)

            return overflow_list, max_density_list

        self.op_collections.fence_region_density_merged_op = merged_density_op

        self.op_collections.fence_region_density_overflow_merged_op = merged_density_overflow_op
        return self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op, self.op_collections.fence_region_density_overflow_merged_op

    def build_congestion_map(self, params, placedb, data_collections):
        """
        Build RUDY congestion map operator.
        
        Returns a callable that takes pos and returns congestion_map [num_bins_x, num_bins_y]
        
        IMPORTANT: RUDY returns flattened tensors, must reshape to 2D!
        """
        num_bins_x = placedb.num_routing_grids_x
        num_bins_y = placedb.num_routing_grids_y
        
        # Use RUDY for fast congestion estimation
        rudy_op = rudy.Rudy(
            netpin_start=data_collections.flat_net2pin_start_map,
            flat_netpin=data_collections.flat_net2pin_map,
            net_weights=data_collections.net_weights,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
            unit_vertical_capacity=placedb.unit_vertical_capacity,
            deterministic_flag=params.deterministic_flag
        )
        
        logging.info(f"[CongestionOpt] RUDY congestion map: {num_bins_x}x{num_bins_y} bins")
        
        # def congestion_map_fn(pos):
        #     """Compute congestion map from positions."""
        #     pin_pos = self.op_collections.pin_pos_op(pos)
        #     # RUDY returns utilization map (flattened 1D tensors!)
        #     horizontal_utilization, vertical_utilization = rudy_op(pin_pos)
            
        #     # *** CRITICAL FIX: Reshape from 1D to 2D ***
        #     horizontal_utilization = horizontal_utilization.view(num_bins_x, num_bins_y)
        #     vertical_utilization = vertical_utilization.view(num_bins_x, num_bins_y)
            
        #     # Combine horizontal and vertical, compute congestion as max(util - 1, 0)
        #     utilization = (horizontal_utilization + vertical_utilization) / 2
        #     congestion = torch.clamp(utilization - 1.0, min=0.0)
        #     return congestion
        def congestion_map_fn(pos):
            """Compute congestion map from positions with robust unpacking."""
            pin_pos = self.op_collections.pin_pos_op(pos)
            
            # 1. 获取 RUDY 输出
            rudy_out = rudy_op(pin_pos)
            
            # 2. 鲁棒性解包：无论返回 1个、2个还是3个值都能处理
            if isinstance(rudy_out, (list, tuple)):
                if len(rudy_out) >= 2:
                    # 拿到前两个：水平和垂直利用率
                    h_util = rudy_out[0]
                    v_util = rudy_out[1]
                else:
                    h_util = rudy_out[0]
                    v_util = rudy_out[0]
            else:
                # 只有一个 Tensor 返回
                h_util = rudy_out
                v_util = rudy_out

            # 3. 核心修复：从 1D 展平张量恢复为 2D 形状 [num_bins_x, num_bins_y]
            # 注意：确保 num_bins_x 和 num_bins_y 在此作用域内可用
            if h_util.dim() == 1:
                h_util = h_util.view(self.params.num_bins_x, self.params.num_bins_y)
            if v_util.dim() == 1:
                v_util = v_util.view(self.params.num_bins_x, self.params.num_bins_y)

            # 4. 计算拥塞图
            # 这里的逻辑建议采用 (H + V) / 2 或 直接相加，取决于你的阈值设置
            utilization = (h_util + v_util) / 2.0
            
            # 计算超出容量的部分 (util - 1.0)，最小为 0
            congestion = torch.clamp(utilization - 1.0, min=0.0)
            
            return congestion
        
        return congestion_map_fn

    def build_congestion_grad(self, params, placedb, data_collections):
        """Build congestion gradient operator."""
        if not HAS_CONGESTION_GRAD:
            logging.warning("[CongestionOpt] congestion_grad module not available")
            return None
        
        # Compute average pins per node
        num_pins = len(data_collections.pin2node_map)
        avg_pin_per_node = num_pins / placedb.num_movable_nodes
        
        return congestion_grad.CongestionGrad(
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_weights=data_collections.net_weights,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=self.cong_bin_size_x,
            bin_size_y=self.cong_bin_size_y,
            num_bins_x=self.cong_num_bins_x,
            num_bins_y=self.cong_num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_nodes=placedb.num_nodes,
            avg_pin_per_node=avg_pin_per_node,
            congestion_threshold=params.congestion_threshold
        )
    
    def compute_field_gradient(self, congestion_map):
        """
        Compute field gradient from congestion map using central difference.
        
        Args:
            congestion_map: [num_bins_x, num_bins_y] congestion values
            
        Returns:
            field_grad_x, field_grad_y: Field gradient components
        """
        # Validate input shape
        if congestion_map.dim() != 2:
            raise ValueError(f"[CongestionOpt] congestion_map must be 2D, got shape {congestion_map.shape}")
        
        num_bins_x, num_bins_y = congestion_map.shape
        
        # Initialize gradients
        field_grad_x = torch.zeros_like(congestion_map)
        field_grad_y = torch.zeros_like(congestion_map)
        
        # Central difference for interior points
        # grad_x[i,j] = (map[i+1,j] - map[i-1,j]) / 2
        field_grad_x[1:-1, :] = (congestion_map[2:, :] - congestion_map[:-2, :]) / 2
        field_grad_y[:, 1:-1] = (congestion_map[:, 2:] - congestion_map[:, :-2]) / 2
        
        # Forward/backward difference for boundaries
        field_grad_x[0, :] = congestion_map[1, :] - congestion_map[0, :]
        field_grad_x[-1, :] = congestion_map[-1, :] - congestion_map[-2, :]
        field_grad_y[:, 0] = congestion_map[:, 1] - congestion_map[:, 0]
        field_grad_y[:, -1] = congestion_map[:, -1] - congestion_map[:, -2]
        
        # Normalize to prevent numerical instability
        max_grad = max(field_grad_x.abs().max().item(), 
                       field_grad_y.abs().max().item(), 1e-6)
        field_grad_x = field_grad_x / max_grad
        field_grad_y = field_grad_y / max_grad
        
        return field_grad_x, field_grad_y
    
    def _compute_effective_congestion_weight(self, pos):
        """
        Compute effective congestion weight based on current congestion state.
        
        Formula: λ₂ = (2 * Nc / N) * base_weight
        where Nc = number of cells in congested regions
              N = total movable cells
        """
        if self.cached_congestion_map is None:
            return self.params.congestion_weight
        
        # Count cells in high congestion regions
        congestion_map = self.cached_congestion_map
        high_cong_mask = congestion_map > self.params.congestion_threshold
        
        # Estimate number of congested cells (approximate)
        num_congested_bins = high_cong_mask.sum().item()
        total_bins = congestion_map.numel()
        congestion_ratio = num_congested_bins / max(total_bins, 1)
        
        # Adaptive weight: more weight when more congestion
        # Scale factor between 0.5 and 2.0
        scale = max(0.5, min(2.0, 2 * congestion_ratio + 0.5))
        
        effective_weight = self.params.congestion_weight * scale
        
        # Clamp to prevent extreme values
        effective_weight = min(effective_weight, 10.0)
        
        return effective_weight

    def _compute_electric_field(self, potential_field):
        """
        Compute electric field from potential field using central difference.
        
        E = -∇ψ
        
        Args:
            potential_field: [num_bins_x, num_bins_y] potential values
            
        Returns:
            field_x, field_y: Electric field components
        """
        num_bins_x, num_bins_y = potential_field.shape
        
        # Initialize field components
        field_x = torch.zeros_like(potential_field)
        field_y = torch.zeros_like(potential_field)
        
        # Central difference for interior points
        # E_x = -∂ψ/∂x ≈ -(ψ[i+1,j] - ψ[i-1,j]) / (2*hx)
        field_x[1:-1, :] = -(potential_field[2:, :] - potential_field[:-2, :]) / (2 * self.cong_bin_size_x)
        field_y[:, 1:-1] = -(potential_field[:, 2:] - potential_field[:, :-2]) / (2 * self.cong_bin_size_y)
        
        # Forward/backward difference for boundaries
        field_x[0, :] = -(potential_field[1, :] - potential_field[0, :]) / self.cong_bin_size_x
        field_x[-1, :] = -(potential_field[-1, :] - potential_field[-2, :]) / self.cong_bin_size_x
        field_y[:, 0] = -(potential_field[:, 1] - potential_field[:, 0]) / self.cong_bin_size_y
        field_y[:, -1] = -(potential_field[:, -1] - potential_field[:, -2]) / self.cong_bin_size_y
        
        # Normalize to prevent numerical instability
        max_field = max(field_x.abs().max().item(), field_y.abs().max().item(), 1e-6)
        field_x = field_x / max_field
        field_y = field_y / max_field
        
        return field_x, field_y


