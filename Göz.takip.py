import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#ilk kısım dosyanın nerede kayıtlı olduğu ile ilgili kısım (cv2 kütüphanesi içersinide)
#ikinci kısım ise belirli olan ve hazır olarak öğretilmiş 'yüzün' algılanması için gerekli olan esas kısımdır.
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#ilk kısım dosyanın nerede kayıtlı olduğu ile ilgili kısım (cv2 kütüphanesi içersinide.)
#ikinci kısım ise belirli olan ve hazır olarak öğretilmiş olan 'gözün' algılanması için gerekli olan esas kısımdır.
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Çerveyi gri tonlamaya dönüştürüyoruz.RGB to BGR
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #buradaki 1.3 değeri pixel sayısını azaltmak için konulmuştur ve sayı ne kadar 1'e yakın olursa hata payı alacaktır.
    #10.000x10.000 pixel bir fotoğrafı ele almak zor olacağı için boyutunu ayarlamak gerekmektedir.

    for (x, y, w, h) in faces:#Kamerada olan tüm yüzleri tarayamak için for dögüsüne girdik
        #x, y, w ,h vermemizin sebebi aslında bir diktörtgeni vermesidir. X ve Y genişliktir.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 1, 0), 4)#(255,0,0)Mavi , dikdörtgenin kalınlığı
        #(x,y) sol alt köşe iken  (x + w, y + h) sağ alt köşeyi temsil eder. İkinci kısım renk belirlemek içindir.
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #Yukarıdaki 2 satırı yazma sebebimiz dikdörtgenin içerisinde gözü aradığımız için sadece dikdörtgen içerisinde
        #arama yapmasını sağlamaktır. Böylelikle spesifik bir alanı arayabiliriz.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            #Bu for döngüsünde göz bulunması halinde dikdörtgen çizimi yapılmaktadır.

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()