from deepface import DeepFace
import cv2 
import matplotlib.pyplot as plt 
from pathlib import Path


imgs = []
p = Path('imgs')
for image in p.iterdir():
    imgs.append(image)
l= []
for index in range(len(imgs)):

    img1 = cv2.imread(str(imgs[index]))
    #plt.imshow(img1[:,:,::-1])
    #plt.show()     #to show which image we are analyzing



    result = DeepFace.analyze(img1, actions=['emotion'])
    l.append(result['dominant_emotion'])
print(l)