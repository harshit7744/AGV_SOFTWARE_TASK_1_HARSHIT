
import math, time
import cv2
import numpy as np
import pybullet as p
from collections import deque
from simulation_setup import setup_simulation

ROAD_GAIN     = 120
OBSTACLE_GAIN = 2



DT           = 1.0 / 60.0
TRACK_END_X  = 31.66
TRACK_HALF_W = 1.16

CAM_W, CAM_H = 320, 240
CAM_FOV      = 60
CAM_FWD      = 0.3
CAM_UP       = 0.5

LK_MAX_CORNERS = 100
LK_QUALITY     = 0.05
LK_MIN_DIST    = 5
LK_WIN         = 7
LK_MAX_ITER    = 15
LK_EPS         = 0.05
LK_LEVELS      = 3
LK_MIN_FLOW    = 0.1

TTC_MAX        = 100.0
TTC_CAP        = 50.0
ALPHA          = 1.0
D_MIN          = 0.05

BASE_VEL       = 18.0
MIN_VEL        = 7.0
MOTOR_FORCE    = 800
MAX_STEER      = 0.60
KP_STEER       = 2.0
MAX_STEER_RATE = 0.03

FREP_SMOOTH    = 5

STRIP_BRIGHT   = 200
STRIP_MAX_PX   = 60

_proj        = None
frep_history = deque(maxlen=FREP_SMOOTH)
prev_steer   = 0.0



def get_camera_frame(car_id):
#setting up the camera
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot      = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    fwd      = rot @ np.array([1, 0, 0])
    up       = rot @ np.array([0, 0, 1])
    eye      = np.array(pos) + fwd * CAM_FWD + up * CAM_UP
    view     = p.computeViewMatrix(
        eye.tolist(), (eye + fwd).tolist(), up.tolist()
    )
    _, _, px, _, _ = p.getCameraImage(
        CAM_W, CAM_H,
        viewMatrix=view, projectionMatrix=_proj,
        renderer=p.ER_TINY_RENDERER
    )
    rgba = np.array(px, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)
    bgr  = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr,  cv2.COLOR_BGR2GRAY)
    return bgr, gray

def get_camera_axes(car_id):
#return camera and car position
    car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
    rot  = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)
    fwd  = rot @ np.array([1, 0, 0])
    left = rot @ np.array([0, 1, 0])
    up   = rot @ np.array([0, 0, 1])

    cam_pos   = np.array(car_pos) + fwd * CAM_FWD + up * CAM_UP
    cam_fwd   = fwd
    cam_right = -left    
    cam_down  = -up     

    return car_pos, cam_pos, cam_fwd, cam_right, cam_down


def world_to_image(world_pt, cam_pos, cam_fwd, cam_right, cam_down):

    fx = (CAM_W / 2.0) / math.tan(math.radians(CAM_FOV / 2.0))
    fy = fx
    d     = world_pt - cam_pos
    depth = float(np.dot(d, cam_fwd))
    if depth <= 0.05:
        return None
    lat  = float(np.dot(d, cam_right))
    vert = float(np.dot(d, cam_down))
    u = int(CAM_W/2.0 + fx * (lat  / depth))
    v = int(CAM_H/2.0 + fy * (vert / depth))
    if 0 <= u < CAM_W and 0 <= v < CAM_H:
        return (u, v)
    return None


def draw_road_boundary(vis, car_id):
    car_pos, cam_pos, cam_fwd, cam_right, cam_down = get_camera_axes(car_id)
    for bnd_y in [+TRACK_HALF_W, -TRACK_HALF_W]:
        pts = []
        for dist in np.linspace(1.0, 35.0, 70):
            wp  = np.array([car_pos[0] + dist, bnd_y, 0.0])
            uv  = world_to_image(wp, cam_pos, cam_fwd, cam_right, cam_down)
            if uv: pts.append(uv)
        for k in range(1, len(pts)):
            cv2.line(vis, pts[k-1], pts[k], (0, 0, 255), 2)
    return vis


def get_strip_mask(gray):
    _, bright = cv2.threshold(gray, STRIP_BRIGHT, 1, cv2.THRESH_BINARY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright.astype(np.uint8), connectivity=8
    )
    mask = np.zeros_like(bright, dtype=np.uint8)
    for lab in range(1, len(stats)):
        if stats[lab, cv2.CC_STAT_AREA] <= STRIP_MAX_PX:
            mask[labels == lab] = 1
    return mask



def _sobel(f64):
    return (cv2.Sobel(f64, cv2.CV_64F, 1, 0, ksize=3),
            cv2.Sobel(f64, cv2.CV_64F, 0, 1, ksize=3))


def _build_pyr(gray):
    #here is the function for building the pyramids
    pyr = [gray.astype(np.float64)]
    for _ in range(LK_LEVELS - 1):
        b = cv2.GaussianBlur(
            pyr[-1].astype(np.float32), (5, 5), 0
        ).astype(np.float64)
        h, w = b.shape
        pyr.append(cv2.resize(b, (w//2, h//2),
                               interpolation=cv2.INTER_LINEAR))
    return pyr


def _lk_level(pf, cf, pts, u0, v0, W=LK_WIN):
    #this applies Lucas kanade from scratch
    Ix, Iy = _sobel(pf)
    H, Wi  = pf.shape
    N      = len(pts)
    u, v   = u0.copy(), v0.copy()
    st     = np.ones(N, dtype=np.uint8)

    for i in range(N):
        if math.isnan(u0[i]) or math.isnan(v0[i]):
            st[i]=0; u[i]=v[i]=np.nan; continue
        x0=int(round(pts[i,0])); y0=int(round(pts[i,1]))
        if x0<W or x0>=Wi-W or y0<W or y0>=H-W:
            st[i]=0; continue
        fIx=Ix[y0-W:y0+W+1, x0-W:x0+W+1].flatten()
        fIy=Iy[y0-W:y0+W+1, x0-W:x0+W+1].flatten()
        A  =np.concatenate([fIx.reshape(-1,1), fIy.reshape(-1,1)], axis=1)
        ATA=A.T@A
        if np.min(np.abs(np.linalg.eigvals(ATA)))<1e-3:
            st[i]=0; continue
        pp=pf[y0-W:y0+W+1, x0-W:x0+W+1].flatten()
        ui, vi = u[i], v[i]
        for _ in range(LK_MAX_ITER):
            cx=int(round(x0+ui)); cy=int(round(y0+vi))
            if cx<W or cx>=Wi-W or cy<W or cy>=H-W:
                st[i]=0; ui=vi=np.nan; break
            b=-(cf[cy-W:cy+W+1, cx-W:cx+W+1].flatten()-pp)
            try:    d=np.linalg.solve(ATA, A.T@b)
            except: st[i]=0; ui=vi=np.nan; break
            ui+=d[0]; vi+=d[1]
            if math.sqrt(d[0]**2+d[1]**2)<LK_EPS: break
        if not(math.isnan(ui) or math.isnan(vi)):
            u[i],v[i]=ui,vi
        else:
            st[i]=0; u[i]=v[i]=np.nan
    return u, v, st


def pyramidal_lk(prev_gray, curr_gray, points):
    #3 layered pyramid
    N    = len(points)
    ppyr = _build_pyr(prev_gray)
    cpyr = _build_pyr(curr_gray)
    pts  = points[:,0,:].astype(np.float64)
    u    = np.zeros(N); v = np.zeros(N)
    ost  = np.ones(N, dtype=np.uint8)

    for lv in range(LK_LEVELS-1, -1, -1):
        sc = 2**lv
        ur, vr, lst = _lk_level(ppyr[lv], cpyr[lv], pts/sc, u/sc, v/sc)
        ost[lst==0] = 0
        if lv > 0:
            ur[np.isnan(ur)]=0.0; vr[np.isnan(vr)]=0.0
            u=ur*sc; v=vr*sc
        else:
            u, v = ur, vr

    npts=np.empty((N,1,2),dtype=np.float64)
    fv  =np.full((N,2),np.nan)
    for i in range(N):
        if ost[i]==1 and not(math.isnan(u[i]) or math.isnan(v[i])):
            npts[i]=[[pts[i,0]+u[i], pts[i,1]+v[i]]]; fv[i]=[u[i],v[i]]
        else:
            npts[i]=[[np.nan,np.nan]]; ost[i]=0
    return npts.astype(np.float32), fv, ost



def compute_foe(points, fv, status):
    #this evaluates the FOE function by solving linear equation
    rA, rb = [], []
    for i in range(len(status)):
        if status[i]==0: continue
        x,y=points[i][0]; vx,vy=fv[i]
        if math.isnan(vx) or math.isnan(vy): continue
        if math.sqrt(vx**2+vy**2)<LK_MIN_FLOW: continue
        rA.append([vy,-vx]); rb.append(x*vy-y*vx)
    if len(rA)<4: return np.array([CAM_W/2.0, CAM_H/2.0])
    A=np.array(rA); b=np.array(rb)
    try:    foe=np.linalg.solve(A.T@A, A.T@b)
    except: return np.array([CAM_W/2.0, CAM_H/2.0])
    return np.clip(foe,[0,0],[CAM_W-1,CAM_H-1])


def compute_ttc(points, fv, status, foe, strip_mask):
    ttc = np.full(len(points), TTC_MAX)
    
    for i in range(len(status)):
        if status[i] == 0: continue
            
        x, y = points[i][0]
        vx, vy = fv[i]
        
        if math.isnan(vx) or math.isnan(vy): continue
            
        px = int(np.clip(round(x), 0, CAM_W-1))
        py = int(np.clip(round(y), 0, CAM_H-1))
        
        if strip_mask[py, px] == 1: continue
            

        dx = x - foe[0]
        dy = y - foe[1]
        
        df_squared = dx**2 + dy**2
        
        if df_squared < 25.0: 
            continue
            

        dot_product = dx*vx + dy*vy
        

        if dot_product <= 0:
            continue
            
        computed_ttc = df_squared / dot_product
        ttc[i] = min(computed_ttc, TTC_MAX)
        
    return ttc

def repulsive_force(points, fv, status, ttc_vals):

    min_ttc = TTC_CAP
    min_x   = None
    min_y   = None

    for i in range(len(status)):
        if status[i]==0: continue
        vx,vy=fv[i]
        if math.isnan(vx) or math.isnan(vy): continue
        ttc=ttc_vals[i]
        if ttc >= TTC_CAP: continue
        if ttc < min_ttc:
            min_ttc  = ttc
            min_x, min_y = points[i][0]

    if min_x is None:
        return 0.0, 0.0

    icx = CAM_W / 2.0
    icy = CAM_H / 2.0
    dx  = icx - min_x
    dy  = icy - min_y
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 1e-3:
        return 0.0, 0.0
    
    # Force magnitude inversely proportional to TTC
    magnitude = OBSTACLE_GAIN / max(min_ttc, 0.5)
    urgency = 10/(min(min_ttc, 0.5))
    return float(magnitude * (dx / dist) * urgency ), float(magnitude * (dy / dist) * urgency)


def attractive_force(car_pos):
    x,y=car_pos[0],car_pos[1]
    dist=math.sqrt((TRACK_END_X-x)**2+y**2)
    if dist<1e-3: return 0.0,0.0,0.0
    th=math.atan2(-y, TRACK_END_X-x)
    mag=ALPHA*dist
    return float(mag*math.cos(th)), float(mag*math.sin(th)), float(th)


def road_force(car_pos, car_id):
    y = car_pos[1]

    d_left  = max(TRACK_HALF_W - y,  D_MIN)
    d_right = max(y + TRACK_HALF_W,  D_MIN)
    F_world = -ROAD_GAIN/(d_left**2) + ROAD_GAIN/(d_right**2)

    car_pos_arr, cam_pos, cam_fwd, cam_right, cam_down = get_camera_axes(car_id)

    F_image = 0.0
    for bnd_y, sign in [(+TRACK_HALF_W, -1.0), (-TRACK_HALF_W, +1.0)]:
        # Sample boundary point ~3m ahead of car
        for dist in [3.0, 5.0, 8.0]:
            wp = np.array([car_pos_arr[0] + dist, bnd_y, 0.0])
            uv = world_to_image(wp, cam_pos, cam_fwd, cam_right, cam_down)
            if uv is None: continue
            # Pixel offset of this boundary from image centre
            px_offset = uv[0] - CAM_W/2.0
            # Closer boundary appears nearer to image centre (smaller |offset|)
            # The closer the boundary, the more we must push away
            if abs(px_offset) < 1e-3: continue
            # Force: push AWAY from this boundary
            d_px = abs(px_offset) / (CAM_W/2.0)  # normalised 0-1
            F_image += sign * ROAD_GAIN * (1.0 - d_px) / max(d_px, 0.01)
            break  # use nearest sample only

    return 0.0, float(F_world + F_image * 0.3)  # image force scaled down (supplementary)


def desired_heading(Fatt_x, Fatt_y, Frep_x, Frep_y, Frd_x, Frd_y, car_yaw):
    
    FXT = Fatt_x - Frd_x + Frep_x
    FYT = Fatt_y + Frep_y + Frd_y

    FX  =  math.cos(car_yaw)*FXT + math.sin(car_yaw)*FYT
    FY  = -math.sin(car_yaw)*FXT + math.cos(car_yaw)*FYT
    
    if abs(FX)<1e-6 and abs(FY)<1e-6: return float(car_yaw)
    return float(math.atan2(FY, FX))


def compute_steer(psi_d, car_yaw):
    global prev_steer
    psi_e = (psi_d-car_yaw+math.pi) % (2*math.pi) - math.pi
    raw   = float(np.clip(KP_STEER*psi_e, -MAX_STEER, MAX_STEER))
    delta = float(np.clip(raw-prev_steer, -MAX_STEER_RATE, MAX_STEER_RATE))
    prev_steer += delta
    return prev_steer


def speed_control(fv, status):
    mags=[math.sqrt(fv[i][0]**2+fv[i][1]**2)
          for i in range(len(status))
          if status[i]==1 and not math.isnan(fv[i][0])]
    if not mags: return BASE_VEL
    avg=float(np.mean(mags))
    if avg<=1.2: return BASE_VEL
    r=float(np.clip((avg-1.2)/2.4,0.0,1.0))
    return BASE_VEL-r*(BASE_VEL-MIN_VEL)


def apply_control(car_id, sj, mj, steer, vel):
    for j in sj:
        p.setJointMotorControl2(car_id,j,p.POSITION_CONTROL,
                                targetPosition=float(steer),force=10)
    for j in mj:
        p.setJointMotorControl2(car_id,j,p.VELOCITY_CONTROL,
                                targetVelocity=float(vel),force=MOTOR_FORCE)


def car_state(car_id):
    pos,_=p.getBasePositionAndOrientation(car_id)
    if pos[0]>=TRACK_END_X-1.5: return 'goal'
    if abs(pos[1])>TRACK_HALF_W*0.95: return 'offtrack'
    return 'ok'



def draw_debug(bgr, p0, fv, st, ttc_vals, strip_mask,
               foe, psi_d, steer, Fatt_x, Fatt_y,
               Frep_x, Frep_y, Frd_y, car_x, car_id,
               min_ttc_idx):
    vis = bgr.copy()
    vis = draw_road_boundary(vis, car_id)

    ov=vis.copy(); ov[strip_mask>0]=[180,80,0]
    cv2.addWeighted(ov,0.20,vis,0.80,0,vis)

    for i in range(len(st)):
        if st[i]==0: continue
        x,y=int(p0[i][0][0]),int(p0[i][0][1])
        vx,vy=fv[i]
        if math.isnan(vx): continue
        px=int(np.clip(x,0,CAM_W-1)); py=int(np.clip(y,0,CAM_H-1))
        if strip_mask[py,px]==1:
            color=(180,60,0)
        elif i==min_ttc_idx:
            color=(0,0,255)   
        elif ttc_vals[i]<TTC_CAP:
            color=(0,100,200) 
        else:
            color=(0,200,0)
        cv2.arrowedLine(vis,(x,y),(int(x+vx*4),int(y+vy*4)),color,1,tipLength=0.4)
        if ttc_vals[i]<TTC_CAP and strip_mask[py,px]==0:
            cv2.putText(vis,f"{ttc_vals[i]:.1f}",(x+4,y-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.27,(0,60,255),1)

    
    if min_ttc_idx is not None and st[min_ttc_idx]==1:
        mx=int(p0[min_ttc_idx][0][0]); my=int(p0[min_ttc_idx][0][1])
        cv2.circle(vis,(mx,my),10,(0,0,255),2)
        cv2.putText(vis,"NEAREST",(mx+12,my),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)

    cv2.drawMarker(vis,(int(foe[0]),int(foe[1])),(0,255,255),cv2.MARKER_CROSS,16,2)
    cx,cy=CAM_W//2,CAM_H//2
    cv2.arrowedLine(vis,(cx,cy),
                    (int(cx+40*math.cos(psi_d)),int(cy-40*math.sin(psi_d))),
                    (255,128,0),2,tipLength=0.3)

    near=sum(1 for i in range(len(st)) if st[i]==1 and ttc_vals[i]<TTC_CAP)
    prog=min(car_x/TRACK_END_X*100,100)

    cv2.putText(vis,f"Steer:{math.degrees(steer):+5.1f}d  ψd:{math.degrees(psi_d):+5.1f}d",
                (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1)
    cv2.putText(vis,f"Fatt:({Fatt_x:.1f},{Fatt_y:.1f}) Frep:({Frep_x:.2f},{Frep_y:.2f}) Frd:{Frd_y:.2f}",
                (5,28),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,220,220),1)
    cv2.putText(vis,f"ROAD:{ROAD_GAIN}  OBS:{OBSTACLE_GAIN}  near:{near}  single-obs",
                (5,41),cv2.FONT_HERSHEY_SIMPLEX,0.35,(180,180,100),1)
    cv2.putText(vis,f"Progress:{prog:.1f}%",
                (5,54),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1)
    bw=int((CAM_W-10)*prog/100)
    cv2.rectangle(vis,(5,CAM_H-7),(CAM_W-5,CAM_H-3),(50,50,50),-1)
    cv2.rectangle(vis,(5,CAM_H-7),(5+bw,CAM_H-3),(0,200,0),-1)
    return vis



def main():
    global _proj

    print("[Nav] Initialising ...")
    car_id,sj,mj=setup_simulation(dt=DT,settle_frames=60,gui=True)
    print(f"[Nav] car={car_id}  steer={sj}  motor={mj}")
    print(f"[Nav] Pyramidal LK: {LK_LEVELS} levels (quarter->half->full)")
    print(f"[Nav] ROAD_GAIN={ROAD_GAIN}  OBSTACLE_GAIN={OBSTACLE_GAIN}")
    print(f"[Nav] Obstacle: single nearest point only (min TTC)")
    print(f"[Nav] Boundary: world-space + image-space (double)")
    print("[Nav] q=quit  d=debug  t=forces")

    _proj=p.computeProjectionMatrixFOV(
        fov=CAM_FOV,aspect=CAM_W/CAM_H,nearVal=0.1,farVal=50.0)

    bgr_prev,gray_prev=get_camera_frame(car_id)
    p0=cv2.goodFeaturesToTrack(gray_prev,LK_MAX_CORNERS,LK_QUALITY,LK_MIN_DIST)
    if p0 is None: p0=np.empty((0,1,2),dtype=np.float32)

    frame_idx=0; show_debug=True

    try:
        while True:
            state=car_state(car_id)
            if state=='goal':
                print("\n[Nav] GOAL REACHED!")
                apply_control(car_id,sj,mj,0.0,0.0); time.sleep(2.0); break
            if state=='offtrack':
                pos,_=p.getBasePositionAndOrientation(car_id)
                print(f"\n[Nav] Off track y={pos[1]:.2f}m")
                apply_control(car_id,sj,mj,0.0,0.0); break

            p.stepSimulation(); time.sleep(DT); frame_idx+=1

            bgr_curr,gray_curr=get_camera_frame(car_id)
            car_pos,car_orn=p.getBasePositionAndOrientation(car_id)
            _,_,car_yaw=p.getEulerFromQuaternion(car_orn)

            if frame_idx%20==0 or p0 is None or len(p0)<10:
                p0=cv2.goodFeaturesToTrack(gray_prev,LK_MAX_CORNERS,
                                            LK_QUALITY,LK_MIN_DIST)
                if p0 is None: p0=np.empty((0,1,2),dtype=np.float32)

            strip_mask=get_strip_mask(gray_curr)

            if len(p0)>0:
                p1,fv,st=pyramidal_lk(gray_prev,gray_curr,p0)
            else:
                p1=np.empty((0,1,2),dtype=np.float32)
                fv=np.empty((0,2)); st=np.empty((0,),dtype=np.uint8)

            foe=(compute_foe(p0,fv,st) if len(p0)>0
                 else np.array([CAM_W/2.0,CAM_H/2.0]))

            ttc_vals=(compute_ttc(p0,fv,st,foe,strip_mask)
                      if len(p0)>0 else np.array([]))

            # Find index of minimum TTC point (for debug display)
            min_ttc_idx = None
            if len(ttc_vals)>0:
                valid = [(ttc_vals[i], i) for i in range(len(st))
                         if st[i]==1 and ttc_vals[i]<TTC_CAP]
                if valid:
                    _, min_ttc_idx = min(valid)

            raw_fx,raw_fy=(repulsive_force(p0,fv,st,ttc_vals)
                           if len(p0)>0 else (0.0,0.0))

            frep_history.append((raw_fx,raw_fy))
            Frep_x=float(np.mean([f[0] for f in frep_history]))
            Frep_y=float(np.mean([f[1] for f in frep_history]))

            Fatt_x,Fatt_y,_=attractive_force(car_pos)
            Frd_x,Frd_y=road_force(car_pos, car_id)

            psi_d=desired_heading(Fatt_x,Fatt_y,Frep_x,Frep_y,
                                   Frd_x,Frd_y,car_yaw)
            steer=compute_steer(psi_d,car_yaw)
            vel=speed_control(fv,st)
            apply_control(car_id,sj,mj,steer,vel)

            if frame_idx%60==0:
                near=[ttc_vals[i] for i in range(len(st))
                      if st[i]==1 and ttc_vals[i]<TTC_CAP] if len(ttc_vals)>0 else []
                print(f"[{frame_idx:04d}] x={car_pos[0]:.1f}m y={car_pos[1]:.2f}m "
                      f"yaw={math.degrees(car_yaw):+.1f}d ψd={math.degrees(psi_d):+.1f}d "
                      f"steer={math.degrees(steer):+.1f}d "
                      f"Frep=({Frep_x:.2f},{Frep_y:.2f}) Frd_y={Frd_y:.2f} "
                      f"near:{len(near)} "
                      f"minTTC={'none' if not near else f'{min(near):.1f}'}")

            if show_debug:
                vis=draw_debug(bgr_curr,p0,fv,st,
                               ttc_vals if len(ttc_vals)>0
                               else np.full(max(len(st),1),TTC_MAX),
                               strip_mask,foe,psi_d,steer,
                               Fatt_x,Fatt_y,Frep_x,Frep_y,
                               Frd_y,car_pos[0],car_id,min_ttc_idx)
                cv2.imshow("AGV — VPF Navigation",vis)

            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            elif key==ord('d'):
                show_debug=not show_debug
                if not show_debug: cv2.destroyAllWindows()
            elif key==ord('t'):
                print(f"[Forces] Fatt=({Fatt_x:.3f},{Fatt_y:.3f}) "
                      f"Frep_raw=({raw_fx:.3f},{raw_fy:.3f}) "
                      f"Frep_smooth=({Frep_x:.3f},{Frep_y:.3f}) "
                      f"Froad=(0,{Frd_y:.3f}) "
                      f"car_y={car_pos[1]:.3f} psi_d={math.degrees(psi_d):+.2f}d")

            if len(st)>0: p0=p1[st==1]
            gray_prev=gray_curr.copy()

    except KeyboardInterrupt:
        print("\n[Nav] Ctrl+C.")
    finally:
        cv2.destroyAllWindows()
        try: p.disconnect()
        except: pass
        print("[Nav] Done.")


if __name__=="__main__":
    main()
