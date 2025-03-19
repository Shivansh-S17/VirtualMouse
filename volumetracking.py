import cv2
import time
import numpy as np
import HTModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

ptime = 0
detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

prev_vol = -65.0
alpha = 0.7

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    if len(lmlist) != 0:
        x1, y1, z1 = lmlist[4][1], lmlist[4][2], lmlist[4][3]
        x2, y2, z2 = lmlist[8][1], lmlist[8][2], lmlist[8][3]

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + ((z2 - z1)*500) ** 2)
        min_dist, max_dist = 30, 250  
        norm_length = np.clip((length - min_dist) / (max_dist - min_dist), 0, 1)
        target_dB = min_vol + (max_vol - min_vol) * norm_length
        target_dB = alpha * target_dB + (1 - alpha) * prev_vol
        prev_vol = target_dB
        vol_scalar = np.interp(target_dB, [min_vol, max_vol], [0, 1])
        volume.SetMasterVolumeLevelScalar(vol_scalar, None)
        current_dB = volume.GetMasterVolumeLevel()

        print(f"Hand Distance: {length:.2f} | Target Volume: {vol_scalar*100:.2f}% | Actual System Volume (dB): {current_dB:.2f}")

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        if length < min_dist:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
