"""
pipeline_inspector.py — TAC Challenge 2026
Pipeline Inspection görevi için tam çözüm.

Kullanım:
  python pipeline_inspector.py --video videos/video1.mp4 --hmin 15 --hmax 82 --smin 101 --vmin 143
  python pipeline_inspector.py --video videos/video1.mp4 --tune
"""

import cv2
import numpy as np
import argparse
from collections import deque

# ─── AYARLAR ──────────────────────────────────────────────────
FRAME_W    = 640
FRAME_H    = 360
MIN_AREA   = 400       # minimum boru kontur alanı px²
DEAD_ZONE  = 0.10      # ±%10 → DUZELT
TURN_ZONE  = 0.28      # ±%28 → DUZELT_SAG/SOL
YAW_THR    = 20.0      # derece eşiği
SMOOTH_N   = 8         # komut smoothing — son N frame ortalaması

# ─── ArUco ────────────────────────────────────────────────────
_adict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
_aparams = cv2.aruco.DetectorParameters()
_aparams.adaptiveThreshWinSizeMin  = 3
_aparams.adaptiveThreshWinSizeMax  = 33
_aparams.adaptiveThreshWinSizeStep = 4
_aparams.minMarkerPerimeterRate    = 0.01
_aparams.errorCorrectionRate       = 0.8
_detector = cv2.aruco.ArucoDetector(_adict, _aparams)

# ─── RENK PALETİ ──────────────────────────────────────────────
COL = {
    "DUZELT":     ( 80, 220,  80),
    "DUZELT_SOL": (100, 200, 255),
    "DUZELT_SAG": (255, 190,  60),
    "SOL":        ( 40,  80, 255),
    "SAG":        (255,  70,  40),
    "YAW_SOL":    (140, 200, 255),
    "YAW_SAG":    (255, 190, 100),
    "YAW_YOK":    (150, 150, 150),
    "BORU_YOK":   (  0,   0, 200),
}


# ─── BORU TESPİT ──────────────────────────────────────────────
def detect_pipe(frame, lo, hi):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.dilate(mask, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None, None, None, None

    # En büyük kontur = boru
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < MIN_AREA:
        return mask, None, None, None, None

    rect = cv2.minAreaRect(best)
    (cx, cy), (w, h), raw = rect
    angle = (raw + 90 if w < h else raw) % 180
    return mask, best, (int(cx), int(cy)), rect, angle


# ─── YÖN KOMUTU ───────────────────────────────────────────────
def compute_cmd(cx, fw, angle):
    err  = cx - fw // 2
    dead = int(fw * DEAD_ZONE)
    turn = int(fw * TURN_ZONE)
    if   abs(err) <= dead: lat = "DUZELT"
    elif abs(err) <= turn: lat = "DUZELT_SAG" if err > 0 else "DUZELT_SOL"
    else:                  lat = "SAG"         if err > 0 else "SOL"

    ah = min(angle, 180 - angle)
    if   ah < YAW_THR: yaw = "YAW_YOK"
    elif angle < 90:   yaw = "YAW_SOL"
    else:              yaw = "YAW_SAG"
    return lat, err, yaw


# ─── ArUco OKUMA ──────────────────────────────────────────────
def read_aruco(frame, mask):
    """
    Sadece boru maskesi üzerinde ara — yanlış pozitifi önler.
    Maske yoksa hiç arama.
    """
    if mask is None or cv2.countNonZero(mask) < 100:
        return [], None

    # Maske etrafına biraz genişlet (marker boru kenarında olabilir)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask_big = cv2.dilate(mask, k, iterations=1)

    # Maske dışını siyah yap
    roi  = cv2.bitwise_and(frame, frame, mask=mask_big)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    corners, ids, _ = _detector.detectMarkers(gray)
    return corners, ids


# ─── SMOOTHING ────────────────────────────────────────────────
class Smoother:
    """Son N frame'in ortalamasını al — komut titremeyi önler."""
    def __init__(self, n=SMOOTH_N):
        self.buf = deque(maxlen=n)

    def update(self, err):
        self.buf.append(err)
        return int(np.mean(self.buf))


# ─── GÖRSELLEŞTİRME ──────────────────────────────────────────
def draw_frame(frame, mask, contour, rect, center,
               lat, err, yaw, angle, ordered_ids, corners, ids):
    fw, fh = frame.shape[1], frame.shape[0]
    vis = frame.copy()

    # Yeşil maske overlay
    ch = np.zeros_like(vis)
    ch[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 0.6, ch, 0.4, 0)

    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 60), 2)
    if rect is not None:
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(vis, [box], -1, (0, 210, 255), 2)

    if center:
        cx, cy = center
        fc = (fw // 2, fh // 2)

        # Boru merkezi ve frame merkezi
        cv2.circle(vis, (cx, cy), 8, (0, 210, 255), -1)
        cv2.circle(vis, fc, 6, (255, 255, 255), -1)

        # Hata çizgisi
        cv2.line(vis, (fc[0], cy), (cx, cy), (50, 80, 255), 2)

        # Dead / turn zone dikdörtgenleri
        dz = int(fw * DEAD_ZONE)
        tz = int(fw * TURN_ZONE)
        cv2.rectangle(vis, (fc[0]-dz, fh//5), (fc[0]+dz, 4*fh//5), ( 80,220, 80), 1)
        cv2.rectangle(vis, (fc[0]-tz, fh//5), (fc[0]+tz, 4*fh//5), ( 50,140, 50), 1)

        # Boru ekseni oku
        rad = np.deg2rad(angle)
        dx, dy = int(80*np.cos(rad)), int(80*np.sin(rad))
        cv2.arrowedLine(vis, (cx-dx, cy-dy), (cx+dx, cy+dy),
                        (255, 100, 0), 2, tipLength=0.2)

    # ArUco markerları çiz
    if ids is not None and len(corners) > 0:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 200), 2)
            mcx = int(pts[:,0].mean())
            mcy = int(pts[:,1].mean())
            mid = ids[i][0]
            # ID etiketi — siyah arka plan üzerine
            label = f"ID:{mid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (mcx-tw//2-4, mcy-th-6), (mcx+tw//2+4, mcy+2),
                          (0, 0, 0), -1)
            cv2.putText(vis, label, (mcx-tw//2, mcy-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    # ── Sol üst bilgi paneli ───────────────────────────────────
    pipe_status = center is not None
    panel_lines = [
        (f"LATERAL : {lat}",            COL.get(lat, (200,200,200))),
        (f"YAW     : {yaw}",            COL.get(yaw, (200,200,200))),
        (f"HATA    : {err:+d} px",      (200, 200, 200)),
        (f"ACI     : {angle:.1f} deg",  (200, 200, 200)),
        (f"BORU    : {'TESPIT' if pipe_status else 'YOK'}",
                                         (80,220,80) if pipe_status else (0,0,220)),
        (f"MARKER  : {len(ordered_ids)} adet", (180, 255, 180)),
    ]
    ph = len(panel_lines) * 24 + 16
    cv2.rectangle(vis, (0, 0), (240, ph), (0, 0, 0), -1)
    cv2.rectangle(vis, (0, 0), (240, ph), (80, 80, 80), 1)
    for i, (txt, col) in enumerate(panel_lines):
        cv2.putText(vis, txt, (8, 18 + i*24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # ── Alt bar — sonuç listesi ────────────────────────────────
    result_str = "IDs: " + (",".join(str(x) for x in ordered_ids) if ordered_ids else "---")
    cv2.rectangle(vis, (0, fh-32), (fw, fh), (0, 0, 0), -1)
    cv2.putText(vis, result_str, (8, fh-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 150), 1, cv2.LINE_AA)

    # ── Alt orta — büyük yön oku ───────────────────────────────
    _draw_big_arrow(vis, lat, fw, fh)

    # ── Sağ alt — maske thumbnail ──────────────────────────────
    th, tw = fh//4, fw//4
    thumb = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (tw, th))
    cv2.putText(thumb, "MASKE", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    vis[fh-th:fh, fw-tw:fw] = thumb

    return vis


def _draw_big_arrow(vis, cmd, fw, fh):
    cx = fw // 2
    cy = fh - 50
    s, t = 28, 4
    c = COL.get(cmd, (200,200,200))
    if cmd in ("SOL", "DUZELT_SOL"):
        cv2.arrowedLine(vis, (cx+s, cy), (cx-s, cy), c, t, tipLength=0.45)
    elif cmd in ("SAG", "DUZELT_SAG"):
        cv2.arrowedLine(vis, (cx-s, cy), (cx+s, cy), c, t, tipLength=0.45)
    elif cmd == "DUZELT":
        cv2.arrowedLine(vis, (cx, cy+s), (cx, cy-s), c, t, tipLength=0.45)
    elif cmd == "BORU_YOK":
        # Çarpı işareti
        cv2.line(vis, (cx-s, cy-s), (cx+s, cy+s), c, t)
        cv2.line(vis, (cx+s, cy-s), (cx-s, cy+s), c, t)


# ─── HSV TUNER ────────────────────────────────────────────────
def run_tuner(cap, lo, hi):
    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    def _n(_): pass
    cv2.createTrackbar("H min","HSV Tuner", int(lo[0]), 179, _n)
    cv2.createTrackbar("S min","HSV Tuner", int(lo[1]), 255, _n)
    cv2.createTrackbar("V min","HSV Tuner", int(lo[2]), 255, _n)
    cv2.createTrackbar("H max","HSV Tuner", int(hi[0]), 179, _n)
    cv2.createTrackbar("S max","HSV Tuner", int(hi[1]), 255, _n)
    cv2.createTrackbar("V max","HSV Tuner", int(hi[2]), 255, _n)
    print("[Tuner] Boru beyaz, arkaplan siyah olsun | q=kaydet  r=basa_sar")
    while True:
        ret, f = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        f = cv2.resize(f, (FRAME_W, FRAME_H))
        l = np.array([cv2.getTrackbarPos("H min","HSV Tuner"),
                      cv2.getTrackbarPos("S min","HSV Tuner"),
                      cv2.getTrackbarPos("V min","HSV Tuner")])
        u = np.array([cv2.getTrackbarPos("H max","HSV Tuner"),
                      cv2.getTrackbarPos("S max","HSV Tuner"),
                      cv2.getTrackbarPos("V max","HSV Tuner")])
        hsv  = cv2.cvtColor(cv2.GaussianBlur(f,(5,5),0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, l, u)
        out  = np.hstack([f,
                          cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                          cv2.bitwise_and(f, f, mask=mask)])
        cv2.putText(out, f"H[{l[0]},{u[0]}] S[{l[1]},{u[1]}] V[{l[2]},{u[2]}]",
                    (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 1)
        cv2.imshow("HSV Tuner", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            lo[:] = l; hi[:] = u
            print(f"\n  --hmin {l[0]} --hmax {u[0]} --smin {l[1]} --vmin {l[2]}")
            break
        elif k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.destroyAllWindows()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# ─── ANA DÖNGÜ ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--output",     default=None)
    ap.add_argument("--tune",       action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--hmin", type=int, default=15)
    ap.add_argument("--smin", type=int, default=80)
    ap.add_argument("--vmin", type=int, default=80)
    ap.add_argument("--hmax", type=int, default=40)
    ap.add_argument("--smax", type=int, default=255)
    ap.add_argument("--vmax", type=int, default=255)
    args = ap.parse_args()

    lo = np.array([args.hmin, args.smin, args.vmin])
    hi = np.array([args.hmax, args.smax, args.vmax])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[HATA] Acilamadi: {args.video}"); return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Bilgi] {int(cap.get(3))}x{int(cap.get(4))} | {total} frame")
    print(f"[Bilgi] HSV lo={lo.tolist()} hi={hi.tolist()}")

    if args.tune:
        run_tuner(cap, lo, hi)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (FRAME_W, FRAME_H))

    # Video hız hesabı
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    real_fps   = 30 if native_fps > 200 else native_fps
    # Kaç frame atlanacak: 1000fps videoda gerçek içerik 30fps
    SKIP       = max(1, int(native_fps / real_fps))
    print(f"[Bilgi] Native FPS: {native_fps:.0f} | Gercek FPS: {real_fps:.0f} | Her {SKIP} framede 1 isleniyor")

    smoother    = Smoother(SMOOTH_N)
    ordered_ids = []
    seen_ids    = set()
    frame_idx   = 0
    paused      = False

    lat = yaw = "BORU_YOK"
    err = 0
    angle = 0.0

    print("\nTuslar: q=cik  p=duraklat  r=basa_sar  s=kaydet\n")

    while True:
        if not paused:
            # SKIP kadar frame atla — sadece okunur, islenmez
            for _ in range(SKIP - 1):
                cap.read()
                frame_idx += 1
            ret, raw = cap.read()
            if not ret: break
            frame_idx += 1

        frame = cv2.resize(raw, (FRAME_W, FRAME_H))

        # ── Boru tespiti ──────────────────────────────────────
        mask, contour, center, rect, ang = detect_pipe(frame, lo, hi)

        if center is not None:
            angle   = ang
            raw_err = center[0] - FRAME_W // 2
            s_err   = smoother.update(raw_err)
            lat, _, yaw = compute_cmd(FRAME_W//2 + s_err, FRAME_W, angle)
            err     = s_err
        else:
            smoother.update(0)
            lat = yaw = "BORU_YOK"
            err = 0
            angle = 0.0

        # ── ArUco okuma (tüm frame) ───────────────────────────
        corners, ids = read_aruco(frame, mask)
        if ids is not None:
            for mid in ids.flatten():
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    ordered_ids.append(int(mid))
                    print(f"  [MARKER +] ID {mid:3d} | "
                          f"Simdiye kadar: {','.join(str(x) for x in ordered_ids)}")

        # ── Görsel ───────────────────────────────────────────
        vis = draw_frame(frame, mask, contour, rect, center,
                         lat, err, yaw, angle,
                         ordered_ids, corners, ids)

        if writer: writer.write(vis)

        if not args.no_display:
            cv2.imshow("Pipeline Inspector", vis)

        k = cv2.waitKey(1) & 0xFF
        if   k == ord("q"): break
        elif k == ord("p"):
            paused = not paused
            print(f"[{'DURAKLADI' if paused else 'DEVAM'}]")
        elif k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            print("[Basa sarıldı]")
        elif k == ord("s"):
            cv2.imwrite(f"frame_{frame_idx:05d}.jpg", raw)
            print(f"[Kaydedildi] frame_{frame_idx:05d}.jpg")

    # ── SONUÇ ─────────────────────────────────────────────────
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    print("\n" + "="*52)
    print("  GOREV SONUCU  (teslim formati)")
    print("="*52)
    result = ",".join(str(x) for x in ordered_ids)
    print(f"  {result if result else 'Hic marker okunamadi'}")
    print(f"  Toplam marker: {len(ordered_ids)}")
    print(f"  Tahmini puan : {len(ordered_ids)*10} + 25 (sira) + 25 (yon) = "
          f"{len(ordered_ids)*10 + 50} pt")
    print("="*52)


if __name__ == "__main__":
    main()