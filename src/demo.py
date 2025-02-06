import cv2
import numpy as np

from virtual_camera import VirtualCamera
from visual_odometry import compute_vo

STEP_AMOUNT = 0.03  #  meters
TURN_AMOUNT = np.deg2rad(3)  # enter degrees


def main():
    vc = VirtualCamera()

    T = np.eye(3)
    cam_img = vc.capture(T)
    Tvo = np.eye(3)
    while True:
        base_img = vc.draw_vehicle_frame()
        cv2.imshow("Demo", base_img)
        cv2.imshow("Camera", cam_img)
        key = cv2.waitKey()
        Tstep = None
        if key == ord("q"):
            break
        elif key == 0:
            Tstep = np.eye(3)
            Tstep[:2, 2] = np.array([STEP_AMOUNT, 0])
        elif key == 1:
            Tstep = np.eye(3)
            Tstep[:2, 2] = np.array([-STEP_AMOUNT, 0])
        elif key == 2:
            Tstep = np.eye(3)
            Tstep[:2, :2] = np.array(
                [
                    [np.cos(-TURN_AMOUNT), -np.sin(-TURN_AMOUNT)],
                    [np.sin(-TURN_AMOUNT), np.cos(-TURN_AMOUNT)],
                ]
            )
        elif key == 3:
            Tstep = np.eye(3)
            Tstep[:2, :2] = np.array(
                [
                    [np.cos(TURN_AMOUNT), -np.sin(TURN_AMOUNT)],
                    [np.sin(TURN_AMOUNT), np.cos(TURN_AMOUNT)],
                ]
            )

        if Tstep is not None:
            T_last = T.copy()
            T @= Tstep
            last_img = cam_img.copy()
            cam_img = vc.capture(T)
            if cam_img is None:
                # couldn't move the camera, maybe out of bounds, revert to the last pose
                T = T_last.copy()
                cam_img = vc.capture(
                    T
                )  # re-capture to reset the camera to the last pose
                continue

            # run VO
            vo_xform = compute_vo(last_img, cam_img, vc.K)
            print(vo_xform)
            if vo_xform is not None:
                Tvo = vo_xform @ Tvo
                # print(Tvo)


if __name__ == "__main__":
    print(
        "Welcome to our VO demo! Press 'q' to quit, or use the arrow keys to move the camera. Enjoy!"
    )
    main()
