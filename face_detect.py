import os
import cv2
from tqdm import tqdm

input_path = ""
output_dir = 'faces/images'
file_l = os.listdir(input_path)
add = 60 # 切り抜き幅調整
cnt = 0
if not os.input_path.exists(output_dir):
    os.makedirs(output_dir)

classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')
notdetected_img = []
for f in tqdm(file_l[:]):
    print(f)
    if f == "add_face": continue
    image = cv2.imread(input_path + f)
    # グレースケールで処理を高速化
    #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(image,minSize = (100,100))
    if faces == ():
        notdetected_img.append(f)
    for i, (x,y,w,h) in enumerate(faces):
        # 一人ずつ顔を切り抜く
        if x-add > 0 and y - add > 0: 
            tmp = y-add
            if tmp > 60: tmp = 60
            face_image = image[y-add-tmp:y+h+add-30, x-add:x+w+add]
        elif x-add < 0 and y - add > 0 :
            tmp = y-add
            if tmp > 60: tmp = 60
            face_image = image[y-tmp:y+h, x-x:x+w+x]
        elif x-add > 0  and y - add < 0 :
            tmp = y
            face_image = image[y-tmp:y+h+tmp, x-tmp:x+w+tmp]
        else:
            face_image = image[y-y:y+h+y, x-x:x+w+x] 




        face_image = cv2.resize(face_image, (1024,1024))
        output_input_path = os.input_path.join(output_dir,'{0}-{1}.jpg'.format(i,f[:-5]))
        cv2.imwrite(output_input_path,face_image)
        cnt += 1


print(f"Num of faces {cnt}")
print(f"List of images for which the face could not be detected \n {notdetected_img}")