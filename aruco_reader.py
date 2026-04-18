"""
aruco_reader.py — TAC Challenge 2026 ArUco Okuyucu
Kullanim:
  python aruco_reader.py --video videos/aruco.mp4
  python aruco_reader.py --video videos/aruco.mp4 --dict ORIGINAL --confirm 2
"""

import cv2
import numpy as np
import argparse

DICT_MAP = {
    "4X4_100":  cv2.aruco.DICT_4X4_100,
    "4X4_50":   cv2.aruco.DICT_4X4_50,
    "4X4_250":  cv2.aruco.DICT_4X4_250,
    "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "5X5_100":  cv2.aruco.DICT_5X5_100,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   required=True)
    ap.add_argument("--dict",    default="ORIGINAL",
                    help=f"Sozluk secenekleri: {list(DICT_MAP.keys())}")
    ap.add_argument("--confirm", type=int, default=3,
                    help="Kac frame ust uste gorulmeli (default: 3)")
    args = ap.parse_args()

    adict  = cv2.aruco.getPredefinedDictionary(
                 DICT_MAP.get(args.dict, cv2.aruco.DICT_ARUCO_ORIGINAL))
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin    = 5
    params.adaptiveThreshWinSizeMax    = 23
    params.adaptiveThreshWinSizeStep   = 4
    params.minMarkerPerimeterRate      = 0.03
    params.errorCorrectionRate         = 0.6
    params.polygonalApproxAccuracyRate = 0.05
    detector = cv2.aruco.ArucoDetector(adict, params)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[HATA] Acilamadi: {args.video}"); return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    real_fps   = 30 if native_fps > 200 else native_fps
    SKIP       = max(1, int(native_fps / real_fps))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Bilgi] {int(cap.get(3))}x{int(cap.get(4))} | {total} frame | SKIP={SKIP}")
    print(f"[Bilgi] Sozluk={args.dict} | Onay={args.confirm} frame")
    print("Tuslar: q=cik  p=duraklat  r=basa_sar  s=kaydet\n")

    ordered_ids   = []
    seen_ids      = set()
    confirm_count = {}
    frame_idx     = 0
    paused        = False

    while True:
        if not paused:
            for _ in range(SKIP - 1):
                cap.read(); frame_idx += 1
            ret, raw = cap.read()
            if not ret: break
            frame_idx += 1

        frame = cv2.resize(raw, (640, 360))
        vis   = frame.copy()
        gray  = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        corners, ids, _ = detector.detectMarkers(gray)

        seen_this_frame = set()

        if ids is not None and len(corners) > 0:
            for i, corner in enumerate(corners):
                pts = corner[0].astype(np.int32)
                mid = int(ids[i][0])
                if mid < 1 or mid > 99: continue  # TAC: ID 1-99 arasi gecerli
                seen_this_frame.add(mid)
                confirm_count[mid] = confirm_count.get(mid, 0) + 1

                confirmed = mid in seen_ids
                color     = (0, 255, 150) if confirmed else (0, 200, 255)

                cv2.polylines(vis, [pts], True, color, 2)
                mcx = int(pts[:,0].mean())
                mcy = int(pts[:,1].mean())

                n   = confirm_count[mid]
                lbl = f"ID:{mid} ({min(n,args.confirm)}/{args.confirm})"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(vis, (mcx-tw//2-4, mcy-th-8),
                              (mcx+tw//2+4, mcy+2), (0,0,0), -1)
                cv2.putText(vis, lbl, (mcx-tw//2, mcy-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                if n >= args.confirm and mid not in seen_ids:
                    seen_ids.add(mid)
                    ordered_ids.append(mid)
                    print(f"  [ONAY] ID {mid:3d} | "
                          f"Liste: {','.join(str(x) for x in ordered_ids)}")

        # Görülmeyenleri sıfırla
        for mid in list(confirm_count):
            if mid not in seen_this_frame and mid not in seen_ids:
                confirm_count[mid] = 0

        # Bilgi paneli
        cv2.rectangle(vis, (0,0), (380, 52), (0,0,0), -1)
        cv2.putText(vis, f"Sozluk:{args.dict}  Onay:{args.confirm}f  Frame:{frame_idx}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
        cv2.putText(vis, f"Onaylanan: {len(seen_ids)}  |  Anlik: {len(seen_this_frame)}",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,255,150), 1)

        result = ",".join(str(x) for x in ordered_ids) if ordered_ids else "---"
        cv2.rectangle(vis, (0, 328), (640, 360), (0,0,0), -1)
        cv2.putText(vis, f"IDs: {result}", (8, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80,255,150), 1)

        cv2.imshow("ArUco Reader", vis)

        k = cv2.waitKey(1) & 0xFF
        if   k == ord("q"): break
        elif k == ord("p"): paused = not paused
        elif k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); frame_idx = 0
        elif k == ord("s"):
            fname = f"aruco_{frame_idx:05d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[Kaydedildi] {fname}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  SONUC (teslim formati)")
    print("="*50)
    result = ",".join(str(x) for x in ordered_ids)
    print(f"  {result if result else 'Hic marker okunamadi'}")
    print(f"  Toplam: {len(ordered_ids)} marker")
    print("="*50)

if __name__ == "__main__":
    main()