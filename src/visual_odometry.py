import cv2
import numpy as np


MAX_FEATURES = 5000
visualize = True


def compute_vo(img1_raw, img2_raw, K):
    img1 = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_raw, cv2.COLOR_BGR2GRAY)

    # params for corner detection
    feature_params = dict(
        maxCorners=MAX_FEATURES, qualityLevel=0.3, minDistance=7, blockSize=7
    )
    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

    # parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(45, 45),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

    # take good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    num_feat = len(good_old)
    if len(good_new) != num_feat:
        print(
            f"Warning: number of new features ({len(good_new)}) does not match number of old features ({num_feat})"
        )
        return None

    if len(good_new) < 5 or len(good_old) < 5:
        print(
            f"Error: not enough features to estimate: {len(good_old)}, {len(good_new)}"
        )
        return None

    # optionally, prep to visualize
    if visualize:
        color = np.random.randint(0, 255, (MAX_FEATURES, 3))
        for idx, (cur_feat, new_feat) in enumerate(zip(good_old, good_new)):
            a, b = cur_feat.astype(int).ravel()
            c, d = new_feat.astype(int).ravel()
            img2_raw = cv2.circle(img2_raw, (a, b), 5, color[idx].tolist(), -1)
            img2_raw = cv2.line(img2_raw, (a, b), (c, d), color[idx].tolist(), 2)

    good_old = np.vstack([good_old.T, np.ones(len(good_old))])
    p_w_1 = np.linalg.inv(K).dot(good_old)
    p_w_1 /= p_w_1[2, :]

    good_new = np.vstack([good_new.T, np.ones(len(good_new))])
    p_w_2 = np.linalg.inv(K).dot(good_new)
    p_w_2 /= p_w_2[2, :]

    estimated_xform, _ = cv2.estimateAffinePartial2D(p_w_1[:2, :].T, p_w_2[:2, :].T)
    if estimated_xform is None:
        return None
    estimated_xform = np.vstack([estimated_xform, [0, 0, 1]])

    if visualize:
        cv2.imwrite("out.png", img2_raw)

    return estimated_xform
