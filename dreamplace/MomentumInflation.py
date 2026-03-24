# dreamplace/MomentumInflation.py
import logging
import torch


class MomentumInflation(object):
    """
    Momentum-based cell inflation for local routing congestion mitigation.

    r_t is interpreted as AREA inflation ratio.
    Therefore width/height are scaled by sqrt(r_t) to preserve aspect ratio.
    """

    def __init__(self, params, placedb, data_collections):
        self.params = params
        self.placedb = placedb
        self.data = data_collections
        self.device = data_collections.pos[0].device
        self.dtype = data_collections.pos[0].dtype

        self.num_movable = placedb.num_movable_nodes

        self.r_min = float(params.mci_r_min)
        self.r_max = float(params.mci_r_max)
        self.alpha = float(params.mci_alpha)
        self.apply_threshold = float(params.mci_apply_threshold)
        self.min_rounds_between_updates = int(params.mci_min_rounds_between_updates)
        self.enable_debug = bool(getattr(params, "mci_debug", 0))

        self.iteration = 0
        self.last_apply_round = -10**9

        # current area inflation ratio
        self.r = torch.ones(self.num_movable, dtype=self.dtype, device=self.device)

        # previous delta_r
        self.delta_r_prev = torch.zeros(self.num_movable, dtype=self.dtype, device=self.device)

        # previous congestion state
        self.prev_cell_congestion = None
        self.prev_avg_congestion = None

        self.eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)

    def _cell_centers(self, pos):
        x = pos[:self.placedb.num_nodes][:self.num_movable]
        y = pos[self.placedb.num_nodes:self.placedb.num_nodes * 2][:self.num_movable]

        cx = x + self.data.node_size_x[:self.num_movable] * 0.5
        cy = y + self.data.node_size_y[:self.num_movable] * 0.5
        return cx, cy

    def _routing_bin_index(self, center_x, center_y):
        gx = ((center_x - self.placedb.routing_grid_xl) / self.placedb.routing_grid_size_x).floor().long()
        gy = ((center_y - self.placedb.routing_grid_yl) / self.placedb.routing_grid_size_y).floor().long()

        gx = gx.clamp_(min=0, max=self.placedb.num_routing_grids_x - 1)
        gy = gy.clamp_(min=0, max=self.placedb.num_routing_grids_y - 1)
        return gx, gy

    def sample_cell_congestion(self, pos, route_utilization_map):
        cx, cy = self._cell_centers(pos)
        gx, gy = self._routing_bin_index(cx, cy)
        return route_utilization_map[gx, gy]

    def _refresh_cached_data(self):
        """
        refresh node_areas and sorted_node_map after node sizes are changed
        """
        self.data.update_node_areas()
        self.data.update_movable_sorted_node_map(self.placedb)

    def apply_current_inflation(self, anchor_centers=True):
        """
        Apply current r to movable node sizes only.
        Keep centers unchanged so that inflation itself does not create artificial jumps.
        """
        with torch.no_grad():
            old_w = self.data.node_size_x[:self.num_movable].clone()
            old_h = self.data.node_size_y[:self.num_movable].clone()

            scale = self.r.clamp(min=self.r_min, max=self.r_max).sqrt()
            new_w = self.data.original_node_size_x[:self.num_movable] * scale
            new_h = self.data.original_node_size_y[:self.num_movable] * scale

            pos = self.data.pos[0]

            if anchor_centers:
                # lower-left -> center
                pos[:self.num_movable].add_(old_w * 0.5)
                pos[self.placedb.num_nodes:self.placedb.num_nodes + self.num_movable].add_(old_h * 0.5)

            # overwrite movable node sizes
            self.data.node_size_x[:self.num_movable].copy_(new_w)
            self.data.node_size_y[:self.num_movable].copy_(new_h)

            self._refresh_cached_data()

            if anchor_centers:
                # center -> lower-left
                pos[:self.num_movable].sub_(new_w * 0.5)
                pos[self.placedb.num_nodes:self.placedb.num_nodes + self.num_movable].sub_(new_h * 0.5)

    def restore_original_sizes(self, anchor_centers=True):
        """
        Restore original movable-node sizes.
        """
        with torch.no_grad():
            old_w = self.data.node_size_x[:self.num_movable].clone()
            old_h = self.data.node_size_y[:self.num_movable].clone()

            pos = self.data.pos[0]

            if anchor_centers:
                pos[:self.num_movable].add_(old_w * 0.5)
                pos[self.placedb.num_nodes:self.placedb.num_nodes + self.num_movable].add_(old_h * 0.5)

            self.data.node_size_x[:self.num_movable].copy_(self.data.original_node_size_x[:self.num_movable])
            self.data.node_size_y[:self.num_movable].copy_(self.data.original_node_size_y[:self.num_movable])

            self._refresh_cached_data()

            if anchor_centers:
                pos[:self.num_movable].sub_(self.data.node_size_x[:self.num_movable] * 0.5)
                pos[self.placedb.num_nodes:self.placedb.num_nodes + self.num_movable].sub_(
                    self.data.node_size_y[:self.num_movable] * 0.5
                )

    def update(self, pos, route_utilization_map, global_round):
        """
        Update momentum-based inflation ratio and apply it if needed.

        return:
            inflation_applied (bool)
        """
        if route_utilization_map is None:
            return False

        if global_round - self.last_apply_round < self.min_rounds_between_updates:
            return False

        with torch.no_grad():
            Ct = self.sample_cell_congestion(pos, route_utilization_map).detach()
            Cbar_t = route_utilization_map.mean().detach()

            # first time: directly use current congestion
            # if self.prev_cell_congestion is None:
            #     s_t = Ct
            #     delta_r_t = (1.0 - self.alpha) * s_t
            if self.prev_cell_congestion is None:
                s_t = torch.clamp(Ct - Cbar_t, min=0.0)
                delta_r_t = (1.0 - self.alpha) * s_t
            else:
                # when a cell moves from above-average congestion to below-average congestion,
                # use negative delta for deflation
                transitioned = (self.prev_cell_congestion > self.prev_avg_congestion) & (Ct < Cbar_t)

                denom = (self.prev_avg_congestion * Cbar_t).clamp(min=self.eps)
                deflation_strength = torch.abs(
                    (self.prev_cell_congestion * Cbar_t - Ct * self.prev_avg_congestion) / denom
                )

                delta_t = torch.where(
                    transitioned,
                    -deflation_strength,
                    torch.ones_like(Ct),
                )

                s_t = delta_t * Ct
                delta_r_t = self.alpha * self.delta_r_prev + (1.0 - self.alpha) * s_t

            r_new = (self.r + delta_r_t).clamp(min=self.r_min, max=self.r_max)

            max_change = (r_new - self.r).abs().max()

            # update history regardless
            self.delta_r_prev.copy_(delta_r_t)
            self.prev_cell_congestion = Ct
            self.prev_avg_congestion = Cbar_t
            self.iteration += 1

            if max_change < self.apply_threshold:
                return False

            self.r.copy_(r_new)
            self.apply_current_inflation(anchor_centers=True)

            # ===== 验证 2：尺寸张量是否真的变化 =====
            # 就放在 apply_current_inflation() 之后，因为这里尺寸已经被实际写回 node_size_x/node_size_y
            # add modify_2
            if self.enable_debug:
                gt_avg = (Ct > Cbar_t).sum()
                lt_avg = (Ct < Cbar_t).sum()
                eq_avg = Ct.numel() - gt_avg - lt_avg

                logging.info(
                    "MCI congestion stats: avgC=%.6E, >avg=%d, <avg=%d, =avg=%d"
                    % (
                        Cbar_t.item(),
                        int(gt_avg.item()),
                        int(lt_avg.item()),
                        int(eq_avg.item()),
                    )
                )
            # modify_1    
            if self.enable_debug:
                size_diff_x = (
                    self.data.node_size_x[:self.num_movable]
                    - self.data.original_node_size_x[:self.num_movable]
                ).abs().max()

                size_diff_y = (
                    self.data.node_size_y[:self.num_movable]
                    - self.data.original_node_size_y[:self.num_movable]
                ).abs().max()

                num_changed = (
                    (
                        self.data.node_size_x[:self.num_movable]
                        - self.data.original_node_size_x[:self.num_movable]
                    ).abs() > 1e-12
                ).sum()

                logging.info(
                    "MCI debug: max node_size_x diff = %.6E, max node_size_y diff = %.6E, changed_nodes = %d"
                    % (size_diff_x.item(), size_diff_y.item(), int(num_changed.item()))
                )
            # =====================================

            self.last_apply_round = global_round
            return True

    def stats(self):
        with torch.no_grad():
            return {
                "avg_inflation_ratio": self.r.mean().detach(),
                "max_inflation_ratio": self.r.max().detach(),
                "min_inflation_ratio": self.r.min().detach(),
                "num_inflated_nodes": (self.r > 1.0 + 1e-6).sum().detach(),
            }