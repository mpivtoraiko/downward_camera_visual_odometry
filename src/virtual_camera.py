import numpy as np
from os import path
import cv2

from utils import Dims

CAMERA_HEIGHT = 1.0  # meters above ground
CAMERA_RESOLUTION = (1920, 1080)
FOCAL_LENGTH = 1000  # in pixels


BASE_IMAGE_FILENAME = path.join(
    path.dirname(path.abspath(__file__)), "../images/gravel.png"
)


class VirtualCamera:
    def __init__(self):
        if not path.exists(BASE_IMAGE_FILENAME):
            raise IOError(f"Error: base image file {BASE_IMAGE_FILENAME} doesn't exist")

        print(f"Loading base image from {path.basename(BASE_IMAGE_FILENAME)}...")
        self.base_image = cv2.imread(BASE_IMAGE_FILENAME, cv2.IMREAD_COLOR)
        if self.base_image is None:
            raise IOError(
                f"Error: could not read base image file {BASE_IMAGE_FILENAME}"
            )

        self.half_frame_w = CAMERA_RESOLUTION[0] / 2 - 1
        self.half_frame_h = CAMERA_RESOLUTION[1] / 2 - 1

        base_img_res = self.base_image.shape
        print(f"Base image resolution: {base_img_res[1]}x{base_img_res[0]}")
        self.half_base_frame_w = base_img_res[1] / 2 - 1
        self.half_base_frame_h = base_img_res[0] / 2 - 1

        self.K = np.matrix(
            [
                [FOCAL_LENGTH, 0, self.half_frame_w + 0.5],
                [0, FOCAL_LENGTH, self.half_frame_h + 0.5],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        self.px_coords_b = None  # to be set by capture()
        self.track = []  # ground truth track

    def draw_vehicle_frame(self):
        if self.px_coords_b is None:
            print("Warning: vehicle frame coordinates not set")
            return

        out_img = cv2.line(
            self.base_image.copy(),
            (self.px_coords_b[Dims.X, 0], self.px_coords_b[Dims.Y, 0]),
            (self.px_coords_b[Dims.X, 1], self.px_coords_b[Dims.Y, 1]),
            (0, 255, 0),
            10,
        )
        out_img = cv2.line(
            out_img,
            (self.px_coords_b[Dims.X, 1], self.px_coords_b[Dims.Y, 1]),
            (self.px_coords_b[Dims.X, 2], self.px_coords_b[Dims.Y, 2]),
            (0, 255, 0),
            10,
        )
        out_img = cv2.line(
            out_img,
            (self.px_coords_b[Dims.X, 2], self.px_coords_b[Dims.Y, 2]),
            (self.px_coords_b[Dims.X, 3], self.px_coords_b[Dims.Y, 3]),
            (0, 255, 0),
            10,
        )
        out_img = cv2.line(
            out_img,
            (self.px_coords_b[Dims.X, 3], self.px_coords_b[Dims.Y, 3]),
            (self.px_coords_b[Dims.X, 0], self.px_coords_b[Dims.Y, 0]),
            (0, 255, 0),
            10,
        )

        # overlay the ground truth track
        if len(self.track) > 1:
            for i in range(1, len(self.track)):
                out_img = cv2.line(
                    out_img, self.track[i - 1], self.track[i], (255, 0, 0), 20
                )
                out_img = cv2.circle(out_img, self.track[i], 40, (255, 0, 0), -1)

        return out_img

    def capture(self, transform_2d):
        self.px_coords_b = None  # reset for this run
        # Generate a new image by clipping the appropriate subset of the base image
        if self.base_image is None:
            raise ValueError("Error: base image is not loaded")

        px_coords_o = np.array(
            [
                [0, 0, 1],
                [CAMERA_RESOLUTION[0] - 1, 0, 1],
                [CAMERA_RESOLUTION[0] - 1, CAMERA_RESOLUTION[1] - 1, 1],
                [0, CAMERA_RESOLUTION[1] - 1, 1],
            ],
            dtype=np.float64,
        )

        w_coords_o = np.linalg.inv(self.K) @ px_coords_o.T  # * CAMERA_HEIGHT
        w_coords_o /= w_coords_o[2, :]
        w_coords_b = transform_2d @ w_coords_o
        self.px_coords_b = self.K @ w_coords_b / CAMERA_HEIGHT
        self.px_coords_b /= self.px_coords_b[2, :]  # normalize
        self.px_coords_b += np.array(
            [
                [
                    self.half_base_frame_w - self.half_frame_w,
                    self.half_base_frame_h - self.half_frame_h,
                    0,
                ]
            ]
        ).T
        self.px_coords_b = np.round(self.px_coords_b).astype(np.int32)

        clip_px_coords = np.array(
            [np.min(self.px_coords_b, axis=1), np.max(self.px_coords_b, axis=1)]
        ).squeeze()

        centroid_x = int(
            (clip_px_coords[1, 0] - clip_px_coords[0, 0]) / 2 + clip_px_coords[0, 0]
        )
        centroid_y = int(
            (clip_px_coords[1, 1] - clip_px_coords[0, 1]) / 2 + clip_px_coords[0, 1]
        )
        if (
            (centroid_x - 1100) < 0
            or (centroid_x + 1100) >= self.base_image.shape[1]
            or (centroid_y - 1100) < 0
            or (centroid_y + 1100) >= self.base_image.shape[0]
        ):
            print("Warning: image clipping is out of bounds")
            return None

        self.track.append((centroid_x, centroid_y))

        crop_img = self.base_image[
            centroid_y - 1100 : centroid_y + 1100, centroid_x - 1100 : centroid_x + 1100
        ]

        # TODO: implement a more direct rotation mx estimation
        # print(transform_2d[1, 0], transform_2d[0, 0])
        rot_ang_deg = np.rad2deg(np.arctan2(transform_2d[1, 0], transform_2d[0, 0]))
        # print(transform_2d)
        # print(rot_ang_deg)
        warp_xform = cv2.getRotationMatrix2D(
            (crop_img.shape[0] / 2, crop_img.shape[1] / 2), rot_ang_deg, 1.0
        )

        warp_img = cv2.warpAffine(crop_img, warp_xform, (2200, 2200))
        margins = np.array([1100 - self.half_frame_w, 1100 - self.half_frame_h]).astype(
            np.int32
        )
        out_img = warp_img[
            margins[Dims.Y] : margins[Dims.Y] + CAMERA_RESOLUTION[1],
            margins[Dims.X] : margins[Dims.X] + CAMERA_RESOLUTION[0],
        ]

        return out_img


if __name__ == "__main__":
    vc = VirtualCamera()
    Tstep = np.eye(3)
    theta = np.pi / 4
    Tstep[:2, :2] = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    print(Tstep)
    T = np.eye(3)
    while True:
        img = vc.capture(T)
        if img is None:
            break
        T @= Tstep
