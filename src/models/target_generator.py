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
        }