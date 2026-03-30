import cv2
import numpy as np
from collections import deque

VIDEO_PATH  = "/OPTICAL_FLOW.mp4" #path for video
START_FRAME = 900 #starting frame
END_FRAME   = 1500 #ending frame
OUTPUT_PATH = "tracked_output.mp4" 

FEATURE_PARAMS = dict(
    maxCorners   = 300,
    qualityLevel = 0.1,
    minDistance  = 20,
    blockSize    = 7,
)
#here I set a maximum limit to number of features
MAX_FEATURES = 200

#after every 3 intervals I have again looked for features to track 
DETECT_INTERVAL = 3

# Pyramidal Lukas Kanade parameters
NUM_LEVELS    = 3      
LK_WIN_SIZE   = 5      
LK_ITERATIONS = 20     
LK_EPSILON    = 0.01   
MAX_FLOW      = 50.0   

# Trails
TRAIL_LENGTH = 25

# Background subtractor (foreground mask for feature detection)
BG_HISTORY    = 200
BG_THRESHOLD  = 25
BG_MORPH_SIZE = 3


def frame_generator(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")
    
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total - 1)
    print(f"[stream] {total} frames  —  streaming {start_frame} to {end_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
    cap.release()


def convolve2d_stride(image, kernel):
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    padded  = np.pad(image, ((pH, pH), (pW, pW)), mode='reflect')
    H, W    = image.shape
    shape   = (H, W, kH, kW)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])

    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', patches, kernel).astype(np.float32)


def pyr_down(image):
    kernel = np.array([
        [1,  4,  6,  4,  1],
        [4, 16, 24, 16,  4],
        [6, 24, 36, 24,  6],
        [4, 16, 24, 16,  4],
        [1,  4,  6,  4,  1]
    ], dtype=np.float32) / 256.0

    if image.ndim == 2:
        blurred = convolve2d_stride(image, kernel)
    else:
        blurred = np.stack([
            convolve2d_stride(image[:, :, c], kernel)
            for c in range(image.shape[2])
        ], axis=-1)

    return blurred[::2, ::2]

def build_pyramid(gray, levels):
    pyr = [gray.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(pyr_down(pyr[-1]))
    pyr.reverse()
    return pyr

def _gaussian_weights(win_size):
#hard code for gaussian transformation
    sigma = win_size / 2.0
    ax    = np.arange(-win_size, win_size + 1, dtype=np.float32)
    g1d   = np.exp(-ax ** 2 / (2 * sigma ** 2))
    g2d   = np.outer(g1d, g1d)
    g2d  /= g2d.sum()
    return g2d.ravel().astype(np.float32) 


def _patch_batch(img, cx, cy, dx_off, dy_off, h, w):
    px = cx[:, None] + dx_off[None, :]
    py = cy[:, None] + dy_off[None, :]

    oob = (
        (px.min(axis=1) < 0) | (px.max(axis=1) >= w - 1) |
        (py.min(axis=1) < 0) | (py.max(axis=1) >= h - 1)
    )

    x0 = np.clip(np.floor(px).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(py).astype(np.int32), 0, h - 2)
    x1, y1 = x0 + 1, y0 + 1
    dx = (px - x0).astype(np.float32)
    dy = (py - y0).astype(np.float32)

    patches = (
        (1 - dy) * (1 - dx) * img[y0, x0] +
        (1 - dy) *      dx  * img[y0, x1] +
             dy  * (1 - dx) * img[y1, x0] +
             dy  *      dx  * img[y1, x1]
    ).astype(np.float32)

    return patches, oob

def lk_single_level(I1, I2, src_pts, init_pts,
                    win=LK_WIN_SIZE,
                    max_iter=LK_ITERATIONS,
                    eps=LK_EPSILON):

    h, w = I1.shape
    N    = len(src_pts)
    if N == 0:
        return init_pts.copy(), np.ones(0, dtype=bool)

   
    Ix = (cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3) +
          cv2.Sobel(I2, cv2.CV_32F, 1, 0, ksize=3)) / 2.0
    Iy = (cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3) +
          cv2.Sobel(I2, cv2.CV_32F, 0, 1, ksize=3)) / 2.0

    ax             = np.arange(-win, win + 1, dtype=np.float32)
    dx_off, dy_off = np.meshgrid(ax, ax)
    dx_off         = dx_off.ravel()
    dy_off         = dy_off.ravel()

    W = _gaussian_weights(win)   # (P,)

    cx0  = np.round(src_pts[:, 0]).astype(np.int32)
    cy0  = np.round(src_pts[:, 1]).astype(np.int32)
    sx   = cx0[:, None] + dx_off[None, :]
    sy   = cy0[:, None] + dy_off[None, :]
    sx_c = np.clip(sx, 0, w - 1).astype(np.int32)
    sy_c = np.clip(sy, 0, h - 1).astype(np.int32)

    valid = (
        (sx.min(axis=1) >= 0) & (sx.max(axis=1) < w) &
        (sy.min(axis=1) >= 0) & (sy.max(axis=1) < h)
    )

    Ix_win = Ix[sy_c, sx_c]   # (N, P)
    Iy_win = Iy[sy_c, sx_c]   # (N, P)

    Ixx = (W * Ix_win * Ix_win).sum(axis=1)
    Iyy = (W * Iy_win * Iy_win).sum(axis=1)
    Ixy = (W * Ix_win * Iy_win).sum(axis=1)
    det      = Ixx * Iyy - Ixy ** 2
    valid   &= (np.abs(det) > 1e-6)
    safe_det = np.where(np.abs(det) > 1e-10, det, 1.0)

    p1, oob_p1 = _patch_batch(
        I1,
        cx0.astype(np.float32), cy0.astype(np.float32),
        dx_off, dy_off, h, w
    )
    valid &= ~oob_p1 

    flow   = (init_pts - src_pts).astype(np.float32)
    active = valid.copy()

    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break

        cx1    = src_pts[idx, 0] + flow[idx, 0]
        cy1    = src_pts[idx, 1] + flow[idx, 1]

        p2_sub, oob = _patch_batch(I2, cx1, cy1, dx_off, dy_off, h, w)
        active[idx[oob]] = False
        alive = ~oob
        if not alive.any():
            break

        a_idx = idx[alive]
        It    = p2_sub[alive] - p1[a_idx] 

        # Gaussian-weighted  A^T W b
        bx = -(W * Ix_win[a_idx] * It).sum(axis=1)
        by = -(W * Iy_win[a_idx] * It).sum(axis=1)

        dvx = ( Iyy[a_idx] * bx - Ixy[a_idx] * by) / safe_det[a_idx]
        dvy = (-Ixy[a_idx] * bx + Ixx[a_idx] * by) / safe_det[a_idx]

        flow[a_idx, 0] += dvx
        flow[a_idx, 1] += dvy
        active[a_idx]  &= (dvx ** 2 + dvy ** 2) >= eps ** 2

    large_flow = (flow[:, 0] ** 2 + flow[:, 1] ** 2) > MAX_FLOW ** 2
    valid &= ~large_flow

    new_pts = src_pts.copy().astype(np.float32)
    new_pts[:, 0] += flow[:, 0]
    new_pts[:, 1] += flow[:, 1]
    return new_pts, valid


def pyramidal_lk(gray1, gray2, pts, num_levels=NUM_LEVELS):

    pyr1   = build_pyramid(gray1, num_levels)
    pyr2   = build_pyramid(gray2, num_levels)
    N      = len(pts)
    flow   = np.zeros((N, 2), dtype=np.float32)
    status = np.ones(N, dtype=bool)

    for lev in range(num_levels):                    
        scale = 2 ** (num_levels - 1 - lev)

        
        src_pts  = (pts / scale).astype(np.float32)
        init_pts = (src_pts + flow).astype(np.float32)

        new_l, st = lk_single_level(
            pyr1[lev], pyr2[lev],
            src_pts, init_pts            
        )

        flow    = new_l - src_pts        
        status &= st

        if lev < num_levels - 1:
            flow *= 2                    

    new_pts         = pts.copy().astype(np.float32)
    new_pts[status] = (pts + flow)[status]
    return new_pts, status



def _fg_mask(bg_sub, frame):
    """
    MOG2 foreground mask, cleaned with morphological ops.
    Returns uint8: 255 = moving subject, 0 = static background.
    """
    raw = bg_sub.apply(frame)
    _, fg = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (BG_MORPH_SIZE, BG_MORPH_SIZE))
    fg = cv2.erode(fg,  k, iterations=1)
    fg = cv2.dilate(fg, k, iterations=3)
    return fg


def detect_features(gray, fg):

    mask    = fg if cv2.countNonZero(fg) > 300 else None
    corners = cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)
    if corners is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    pts    = corners.reshape(-1, 2).astype(np.float32)
    colors = np.random.randint(80, 256, size=(len(pts), 3), dtype=np.uint8)
    return pts, colors


def _merge_features(pts, colors, trails, new_pts_det, new_colors_det):

    if len(new_pts_det) == 0:
        return pts, colors, trails

    slots_free = MAX_FEATURES - len(pts)
    if slots_free <= 0:
        return pts, colors, trails

    take       = min(slots_free, len(new_pts_det))
    add_pts    = new_pts_det[:take]
    add_colors = new_colors_det[:take]

    pts    = np.vstack([pts,    add_pts])    if len(pts)    > 0 else add_pts
    colors = np.vstack([colors, add_colors]) if len(colors) > 0 else add_colors
    for p in add_pts:
        trails.append(deque([(p[0], p[1])], maxlen=TRAIL_LENGTH))

    return pts, colors, trails


def draw_tracks(frame, pts, trails, colors):
    vis = frame.copy()
    for trail, color in zip(trails, colors):
        pts_trail = [p for p in trail if p is not None]
        if len(pts_trail) < 2:
            continue
        bgr  = (int(color[0]), int(color[1]), int(color[2]))
        poly = np.array(pts_trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], False, bgr, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Features: {len(pts)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _in_bounds(pts, shape):
    h, w = shape[:2]
    return (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )


def run_tracker(video_path=VIDEO_PATH,
                start_frame=START_FRAME,
                end_frame=END_FRAME):

    gen = frame_generator(video_path, start_frame, end_frame)
    try:
        idx0, frame_prev = next(gen)
    except StopIteration:
        print("Empty video.")
        return

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_THRESHOLD, detectShadows=False
    )

    writer = None
    if OUTPUT_PATH:
        h, w   = frame_prev.shape[:2]
        writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30, (w, h)
        )

    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    fg0       = _fg_mask(bg_sub, frame_prev)

    pts, colors = detect_features(gray_prev, fg0)
    trails      = [deque([(p[0], p[1])], maxlen=TRAIL_LENGTH) for p in pts]
    print(f"[frame {idx0:4d}] initial features: {len(pts)}")

    vis = draw_tracks(frame_prev, pts, trails, colors)
    if writer:
        writer.write(vis)
    cv2.imshow("LK Tracker — Fixed", vis)
    cv2.waitKey(1)

    frames_since_detect = 0

    for abs_idx, frame_curr in gen:
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        frames_since_detect += 1
        do_detect = (frames_since_detect >= DETECT_INTERVAL)

        if len(pts) > 0:
            new_pts, status = pyramidal_lk(gray_prev, gray_curr, pts)
            valid           = status & _in_bounds(new_pts, gray_curr.shape)
            new_pts         = new_pts[valid]

            survived_trails, survived_colors = [], []
            t = 0
            for j, ok in enumerate(valid):
                if ok:
                    trails[j].append((new_pts[t][0], new_pts[t][1]))
                    survived_trails.append(trails[j])
                    survived_colors.append(colors[j])
                    t += 1
            trails = survived_trails
            colors = (np.array(survived_colors, dtype=np.uint8)
                      if survived_colors else np.empty((0, 3), dtype=np.uint8))
            pts    = new_pts

        if do_detect:
            fg_curr = _fg_mask(bg_sub, frame_curr)
            new_pts_det, new_colors_det = detect_features(gray_curr, fg_curr)
            pts, colors, trails = _merge_features(
                pts, colors, trails, new_pts_det, new_colors_det
            )
            frames_since_detect = 0
        else:
            _ = bg_sub.apply(frame_curr)   
        gray_prev = gray_curr

        vis = draw_tracks(frame_curr, pts, trails, colors)
        print(f"[frame {abs_idx:4d}] tracking {len(pts):4d} features")
        if writer:
            writer.write(vis)
        cv2.imshow("LK Tracker — Fixed", vis)
        if cv2.waitKey(1) == ord("q"):
            break

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run_tracker()
