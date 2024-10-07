import dlib
import sys
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import queue

# from light_variability import adjust_gamma

FACE_DOWNSAMPLE_RATIO = 1.5
Yukseklik = 460

thresh = 0.27
modelPath = "shape_predictor_68_face_landmarks.dat"
sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)
#Göz indexleri
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]
#Işık ayaralarının yapılması. Yüzün daha net görünmesi için
Uykusayaci = 0
uyku = 0
durum = 0
blinkTime = 0.15  # 150ms
uykuTime = 1.5  # 1200ms
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

invGamma = 1.0 / GAMMA
table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(0, 256)]).astype("uint8")

#Konfigürasyon ayarları
def gamma_correction(image):
    return cv2.LUT(image, table)

# Göörüntü histogramının piksel yoğunluk dağılımını güncelleyerek
# görüntünün genel kontrastını ayarlayan temel bir görüntü işleme tekniğidir.
# Bunu yapmak, düşük kontrastlı alanların çıktı görüntüsünde daha yüksek kontrast elde etmesini sağlar.
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

#Uyarı aldığımız zaman çalması için oluşturduğumuz parça.
def soundAlert(path, threadStatusQ):
    while True:
        if not threadStatusQ.empty():
            FINISHED = threadStatusQ.get()
            if FINISHED:
                break
        playsound.playsound(path)

#Aşağıdaki kısımda gözün alt ve üst kısımları arasındaki mesafe ölçülmüştür.
#Gözün uzaklık durumunu almak için yazılmıştır. 1 boyutlu array'in mesafini ölçer. Örnek olarak;
#>>> from scipy.spatial import distance
#>>> distance.euclidean([1, 0, 0], [0, 1, 0]) == 1.4142135623730951
#>>> distance.euclidean([1, 1, 0], [0, 1, 0]) == 1.0
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def checkEyeStatus(landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
#Sol gözün indexleri belirlenir.
    hullLeftEye = []
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)
#Sağ gözün indexleri belirlenir.
    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))

    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    #Her iki göz için (x, y)-koordinatları verildiğinde aşağıdaki satırlarda göz en-boy oranlarını hesaplıyoruz.
    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)
#Gözün ortasını almak için toplayıp 2'ye bölüyoruz.
    ear = (leftEAR + rightEAR) / 2.0
    #############################################################################

    eyeStatus = 1  #1 için açık, 0 için kapalı demektir.
    if (ear < thresh):
        eyeStatus = 0

    return eyeStatus


def checkBlinkStatus(eyeStatus):
    global durum, Uykusayaci, uyku
    if (durum >= 0 and durum <= falseBlinkLimit):
        if (eyeStatus):
            durum = 0

        else:
            durum += 1

    elif (durum >= falseBlinkLimit and durum < uykuLimit):
        if (eyeStatus):
            Uykusayaci += 1
            durum = 0

        else:
            durum += 1


    else:
        if (eyeStatus):
            durum = 0
            uyku = 1
            Uykusayaci += 1

        else:
           uyku = 1


def getLandmarks(im):
    imSmall = cv2.resize(im, None,
                         fx=1.0 / FACE_DOWNSAMPLE_RATIO,
                         fy=1.0 / FACE_DOWNSAMPLE_RATIO,
                         interpolation=cv2.INTER_LINEAR)

    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points


capture = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = capture.read()

Toplamzaman = 0.0
validFrames = 0
dummyFrames = 100

print("Kalibrasyon dogrudur.")
while (validFrames < dummyFrames):
    validFrames += 1
    t = time.time()
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height) / Yukseklik
    frame = cv2.resize(frame, None,
                       fx=1 / IMAGE_RESIZE,
                       fy=1 / IMAGE_RESIZE,
                       interpolation=cv2.INTER_LINEAR)

    # adjusted = gamma_correction(frame)
    adjusted = histogram_equalization(frame)

    landmarks = getLandmarks(adjusted)
    timeLandmarks = time.time() - t

    if landmarks == 0:
        validFrames -= 1
        cv2.putText(frame, "Yuz algilanmadi", (10, 30), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "yada uzaklikla alakali problem var", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("Goz Durum", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit()

    else:
        Toplamzaman += timeLandmarks

print("Kalibrasyon yapıldı.")

kbs = Toplamzaman / dummyFrames
print("Kare basina saniye 'ms'".format(kbs * 1000))

uykuLimit = uykuTime / kbs
falseBlinkLimit = blinkTime / kbs
print("uyku limit: {}, false blink limit: {}".format(uykuLimit, falseBlinkLimit))

if __name__ == "__main__":
    vid_writer = cv2.VideoWriter('output-low-light-2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                 (frame.shape[1], frame.shape[0]))
    while (1):
        try:
            t = time.time()
            ret, frame = capture.read()
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / Yukseklik
            frame = cv2.resize(frame, None,fx=1 / IMAGE_RESIZE, fy=1 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)


            adjusted = histogram_equalization(frame)
            landmarks = getLandmarks(adjusted)
            #Kodun bu kısmı tamamen yüzün algılanmadığı durumların saptanması için tasarlanmışıtr.
            #Yüzün saptanmadığı yada gözlerin saptanamadığı durumlarda yeni bir frame açılır.
            if landmarks == 0: #Yüzün var ama hatlar olmadığı durum içindir.

                validFrames -= 1
                cv2.putText(frame, "Yuz algilanmadi.", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Yada uzaklik ile ilgili problem var.", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Surucu Kamerasi 1", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            eyeStatus = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            for i in range(0, len(leftEyeIndex)):
                cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1, (0, 0, 255), -1,
                           lineType=cv2.LINE_AA)

            for i in range(0, len(rightEyeIndex)):
                cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1, (0, 0, 255), -1,
                           lineType=cv2.LINE_AA)

            if uyku:
                cv2.putText(frame, "! ! ! DIKKATLI GITMELISIN ! ! !", (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                            2, cv2.LINE_AA)
                if not ALARM_ON:
                    ALARM_ON = True
                    threadStatusQ.put(not ALARM_ON)
                    thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ,))
                    thread.setDaemon(True)
                    thread.start()

            else:
                cv2.putText(frame, "UykuSayac : {}".format(Uykusayaci), (260, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 0, 255), 2, cv2.LINE_AA)
                # (0, 400)
                ALARM_ON = False

            cv2.imshow("Surucu Kamerasi", frame)
            vid_writer.write(frame)

            k = cv2.waitKey(1)
            if k == ord('r'):
                durum = 0
                uyku = 0
                ALARM_ON = False
                threadStatusQ.put(not ALARM_ON)

            elif k == 27:
                break

            # print("Time taken", time.time() - t)

        except Exception as e:
            print(e)

    capture.release()
    vid_writer.release()
    cv2.destroyAllWindows()