import math
import torch


class TargetGenerator:

    def __init__(self):


        self.level_config = {
            8: {
                "range": (0, 64),
            },
            16: {
                "range": (64, 128),
            },
            32: {
                "range": (128, 256),
            },
            64: {
                "range": (256, 512),
            },
            128: {
                "range": (512, float("inf")),
            },
        }

    # =========================================================
    # Feature-map location -> image coordinates
    # =========================================================

    def _convert_location(self, x, y, stride):

        mapped_x = (x + 0.5) * stride
        mapped_y = (y + 0.5) * stride

        return mapped_x, mapped_y

    # =========================================================
    # Check whether a location is inside a GT box
    # =========================================================

    def _check_location(self, box, location):

        x1, y1, x2, y2 = box
        x, y = location

        return (
            x >= x1
            and x <= x2
            and y >= y1
            and y <= y2
        )

    # =========================================================
    # Calculate LTRB distances
    # =========================================================

    def _calculate_ltrb(self, box, location):

        x1, y1, x2, y2 = box
        x, y = location

        l = x - x1
        t = y - y1
        r = x2 - x
        b = y2 - y

        return l, t, r, b

    # =========================================================
    # Check regression range
    # =========================================================

    def _check_scale(self, stride, ltrb):

        lower, upper = self.level_config[stride]["range"]

        # FCOS assegna il box al livello in base alla
        # massima distanza LTRB.
        max_distance = max(ltrb)

        if stride == 128:
            return max_distance >= lower

        return (
            max_distance >= lower
            and max_distance < upper
        )

    # =========================================================
    # Find all GT boxes compatible with this location
    # =========================================================

    def _find_matching_boxes(
        self,
        boxes,
        location,
        stride,
    ):

        matching_boxes = []

        for box in boxes:

            # 1. location inside box
            if not self._check_location(
                box,
                location,
            ):
                continue

            # 2. LTRB
            ltrb = self._calculate_ltrb(
                box,
                location,
            )

            # 3. box inside regression range
            if not self._check_scale(
                stride,
                ltrb,
            ):
                continue

            matching_boxes.append(box)

        return matching_boxes

    # =========================================================
    # Select GT box when multiple boxes match
    # =========================================================

    def _select_box(self, matching_boxes):

        if len(matching_boxes) == 0:
            return None

        selected_box = None
        min_area = float("inf")

        for box in matching_boxes:

            x1, y1, x2, y2 = box

            area = (
                (x2 - x1)
                *
                (y2 - y1)
            )

            if area < min_area:

                min_area = area
                selected_box = box

        return selected_box

    # =========================================================
    # Calculate center-ness target
    # =========================================================

    def _calculate_centerness(self, ltrb):

        l, t, r, b = ltrb

        lr = min(l, r) / max(l, r)
        tb = min(t, b) / max(t, b)

        centerness = math.sqrt(
            lr * tb
        )

        return centerness

    # =========================================================
    # Generate targets for one FPN level
    # =========================================================


    def generate_targets(
        self,
        label_boxes,
        feature_shape,
        stride,
        device=None,
    ):
        """
        Generate FCOS targets for one FPN level.

        The implementation is vectorized:
            - no Python loop over feature-map locations
            - no Python loop over GT boxes

        This should be substantially faster than the original
        location-by-location implementation.
        """

        height, width = feature_shape

        # ---------------------------------------------------------
        # Empty GT case
        # ---------------------------------------------------------

        if label_boxes.numel() == 0:

            return {
                "positive": torch.zeros(
                    (height, width),
                    dtype=torch.bool,
                    device=device,
                ),
                "ltrb": torch.zeros(
                    (height, width, 4),
                    dtype=torch.float32,
                    device=device,
                ),
                "centerness": torch.zeros(
                    (height, width),
                    dtype=torch.float32,
                    device=device,
                ),
            }

        # ---------------------------------------------------------
        # Make sure boxes are on the correct device.
        # ---------------------------------------------------------

        boxes = label_boxes.to(
            device=device,
            dtype=torch.float32,
        )

        num_boxes = boxes.shape[0]

        # ---------------------------------------------------------
        # Create all feature-map locations.
        #
        # Shape:
        #     [H * W, 2]
        #
        # Each row is:
        #     [x_image, y_image]
        # ---------------------------------------------------------

        ys = torch.arange(
            height,
            device=device,
            dtype=torch.float32,
        )

        xs = torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        )

        grid_y, grid_x = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        locations = torch.stack(
            (
                (grid_x + 0.5) * stride,
                (grid_y + 0.5) * stride,
            ),
            dim=-1,
        ).reshape(
            -1,
            2,
        )

        # ---------------------------------------------------------
        # Compute LTRB distances for every:
        #
        #     location x GT box
        #
        # Shape:
        #     [H*W, N, 4]
        #
        # where N = number of GT boxes.
        # ---------------------------------------------------------

        x = locations[:, 0:1]
        y = locations[:, 1:2]

        x1 = boxes[:, 0].unsqueeze(0)
        y1 = boxes[:, 1].unsqueeze(0)
        x2 = boxes[:, 2].unsqueeze(0)
        y2 = boxes[:, 3].unsqueeze(0)

        left = x - x1
        top = y - y1
        right = x2 - x
        bottom = y2 - y

        ltrb_all = torch.stack(
            (
                left,
                top,
                right,
                bottom,
            ),
            dim=-1,
        )

        # ---------------------------------------------------------
        # Check whether each location is inside each box.
        #
        # Shape:
        #     [H*W, N]
        # ---------------------------------------------------------

        inside = (
            (locations[:, 0:1] >= x1)
            & (locations[:, 0:1] <= x2)
            & (locations[:, 1:2] >= y1)
            & (locations[:, 1:2] <= y2)
        )

        # ---------------------------------------------------------
        # FCOS regression range.
        #
        # max(L, T, R, B)
        # ---------------------------------------------------------

        max_distance = ltrb_all.max(
            dim=-1
        ).values

        lower, upper = self.level_config[
            stride
        ]["range"]

        scale_mask = (
            max_distance >= lower
        )

        if stride != 128:
            scale_mask = (
                scale_mask
                & (max_distance < upper)
            )

        # ---------------------------------------------------------
        # Final matching mask.
        #
        # A location can use a box only if:
        #
        #     1. it is inside the box
        #     2. the box satisfies the level range
        # ---------------------------------------------------------

        matching = (
            inside
            & scale_mask
        )

        # ---------------------------------------------------------
        # Select the smallest-area matching box.
        #
        # This reproduces the original _select_box() rule.
        # ---------------------------------------------------------

        box_area = (
            (boxes[:, 2] - boxes[:, 0])
            *
            (boxes[:, 3] - boxes[:, 1])
        )

        # [1, N] -> [H*W, N]
        area_matrix = box_area.unsqueeze(0).expand(
            locations.shape[0],
            num_boxes,
        )

        # Invalid matches are assigned +inf.
        masked_area = torch.where(
            matching,
            area_matrix,
            torch.full_like(
                area_matrix,
                float("inf"),
            ),
        )

        # Smallest-area matching box.
        min_area, selected_indices = (
            masked_area.min(
                dim=1
            )
        )

        # A location is positive iff at least one
        # GT box matched it.
        positive_flat = torch.isfinite(
            min_area
        )

        # ---------------------------------------------------------
        # Select LTRB for the chosen box.
        # ---------------------------------------------------------

        num_locations = locations.shape[0]

        location_indices = torch.arange(
            num_locations,
            device=device,
        )

        selected_ltrb = ltrb_all[
            location_indices,
            selected_indices,
        ]

        # ---------------------------------------------------------
        # Locations without a valid GT box do not matter.
        #
        # Set their regression target to zero.
        # ---------------------------------------------------------

        selected_ltrb = torch.where(
            positive_flat.unsqueeze(-1),
            selected_ltrb,
            torch.zeros_like(selected_ltrb),
        )

        # ---------------------------------------------------------
        # Centerness
        #
        # sqrt(
        #     min(l,r) / max(l,r)
        #     *
        #     min(t,b) / max(t,b)
        # )
        # ---------------------------------------------------------

        left_target = selected_ltrb[:, 0]
        top_target = selected_ltrb[:, 1]
        right_target = selected_ltrb[:, 2]
        bottom_target = selected_ltrb[:, 3]

        lr_min = torch.minimum(
            left_target,
            right_target,
        )

        lr_max = torch.maximum(
            left_target,
            right_target,
        )

        tb_min = torch.minimum(
            top_target,
            bottom_target,
        )

        tb_max = torch.maximum(
            top_target,
            bottom_target,
        )

        # Avoid division by zero.
        eps = torch.finfo(
            selected_ltrb.dtype
        ).eps

        lr_ratio = (
            lr_min
            / lr_max.clamp_min(eps)
        )

        tb_ratio = (
            tb_min
            / tb_max.clamp_min(eps)
        )

        centerness_flat = torch.sqrt(
            lr_ratio * tb_ratio
        )

        # Background locations -> 0.
        centerness_flat = torch.where(
            positive_flat,
            centerness_flat,
            torch.zeros_like(
                centerness_flat
            ),
        )

        # ---------------------------------------------------------
        # Restore feature-map shapes.
        # ---------------------------------------------------------

        positive = positive_flat.reshape(
            height,
            width,
        )

        ltrb = selected_ltrb.reshape(
            height,
            width,
            4,
        )

        centerness = centerness_flat.reshape(
            height,
            width,
        )

        # ---------------------------------------------------------
        # Return targets.
        # ---------------------------------------------------------

        return {
            "positive": positive,
            "ltrb": ltrb,
            "centerness": centerness,
        }


    """def generate_targets(
        self,
        label_boxes,
        feature_shape,
        stride,
        device=None,
    ):

        height, width = feature_shape

        # -----------------------------------------------------
        # Target: positive / negative
        # -----------------------------------------------------

        positive = torch.zeros(
            (height, width),
            dtype=torch.bool,
            device=device,
        )

        # -----------------------------------------------------
        # Target: LTRB
        #
        # [H, W, 4]
        # -----------------------------------------------------

        ltrb = torch.zeros(
            (height, width, 4),
            dtype=torch.float32,
            device=device,
        )

        # -----------------------------------------------------
        # Target: center-ness
        #
        # [H, W]
        # -----------------------------------------------------

        centerness = torch.zeros(
            (height, width),
            dtype=torch.float32,
            device=device,
        )

        # -----------------------------------------------------
        # Iterate over every location
        # -----------------------------------------------------

        for y in range(height):

            for x in range(width):

                # ---------------------------------------------
                # Feature-map location -> image coordinates
                # ---------------------------------------------

                location = self._convert_location(
                    x,
                    y,
                    stride,
                )

                # ---------------------------------------------
                # Find compatible GT boxes
                # ---------------------------------------------

                matching_boxes = self._find_matching_boxes(
                    label_boxes,
                    location,
                    stride,
                )

                # ---------------------------------------------
                # No compatible GT box
                # ---------------------------------------------

                if len(matching_boxes) == 0:
                    continue

                # ---------------------------------------------
                # Multiple compatible boxes:
                # choose smallest area
                # ---------------------------------------------

                selected_box = self._select_box(
                    matching_boxes
                )

                # ---------------------------------------------
                # Calculate LTRB target
                # ---------------------------------------------

                location_ltrb = self._calculate_ltrb(
                    selected_box,
                    location,
                )

                # ---------------------------------------------
                # Calculate center-ness target
                # ---------------------------------------------

                location_centerness = self._calculate_centerness(
                    location_ltrb
                )

                # ---------------------------------------------
                # Store targets
                # ---------------------------------------------

                positive[y, x] = True

                ltrb[y, x] = torch.tensor(
                    location_ltrb,
                    dtype=torch.float32,
                    device=device,
                )

                centerness[y, x] = location_centerness

        # -----------------------------------------------------
        # Return all targets for this FPN level
        # -----------------------------------------------------

        return {
            "positive": positive,
            "ltrb": ltrb,
            "centerness": centerness,
        }"""
    
        