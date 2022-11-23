import sqlite3
from PIL import Image, ImageFont
import tkinter as tk
import os
import hand as htm  # import modul hand
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from gtts import gTTS
from playsound import playsound
import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
#################################  GUI DU LIEU LEN FIREBASE################################
cred = credentials.Certificate('my-data-ceba1-firebase-adminsdk-genhz-df6b7229c4.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://my-data-ceba1-default-rtdb.firebaseio.com/'
})
ref = db.reference('CONTROL')
################################################################################################
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils
songontay = 0
FolderPath="Fingers"
lst=os.listdir(FolderPath)
lst_2=[]  # khai báo list chứa các mảng giá trị của các hình ảnh/
for i in lst:
    #print(i)
    image=cv2.imread(f"{FolderPath}/{i}")  # Fingers/1.jpg , Fingers/2.jpg ...
    #print(f"{FolderPath}/{i}")
    lst_2.append(image)
# print(len(lst_2))
pTime=0
detector =htm.handDetector(detectionCon=0.65)
#0.75 độ chính xác 75%
fingerid= [4,8,12,16,20]
def detectPose(image, pose, display=True):
    # Create a copy of the input image.
    output_image = image.copy()
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    # Initialize a list to store the detected landmarks.
    landmarks = []
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # Otherwise
    else:
        # Return the output image and the found landmarks.
        return output_image, landmarks
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
# Iterate until the video is accessed successfully.
def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle
# Setup Pose function for video.
#pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def button_getdata():
    cap = cv2.VideoCapture(0)  # 0 is default webcam
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    sampleNum = 0
    # Hàm cập nhật tên và ID vào CSDL
    def insertOrUpdate(id, name):
        conn = sqlite3.connect(r"C:\Users\Nhu\Downloads\Face Recognition\dataset/data.db")
        cursor = conn.execute('SELECT * FROM people WHERE ID=' + str(id))
        isRecordExist = 0
        for row in cursor:
            isRecordExist = 1
        if isRecordExist == 1:
            cmd = "UPDATE people SET Name=' " + str(name) + " ' WHERE ID=" + str(id)
        else:
            cmd = "INSERT INTO people(ID,Name) Values(" + str(id) + ",'" + str(name) + " ' )"
        conn.execute(cmd)
        conn.commit()
        conn.close()
    id = input('Nhập mã người nhà:')
    name = input('Nhập tên :')
    insertOrUpdate(id, name)
    while (True):
        ret, img = cap.read()
        # Lật ảnh cho đỡ bị ngược
        img = cv2.flip(img, 1)
        # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
        centerH = img.shape[0] // 2;
        centerW = img.shape[1] // 2;
        sizeboxW = 300;
        sizeboxH = 400;
        cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                      (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
        # Đưa ảnh về ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Nhận diện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật quanh mặt nhận được
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if not os.path.exists(r"C:\Users\Nhu\Downloads\Face Recognition\dataset"):
                os.makedirs(r"C:\Users\Nhu\Downloads\Face Recognition\dataset")
            sampleNum = sampleNum + 1
            # Ghi dữ liệu khuôn mặt vào thư mục dataSet
            cv2.imwrite(r"C:\Users\Nhu\Downloads\Face Recognition\dataset/User." + id + '.' + str(sampleNum) + ".jpg",
                gray[y:y + h, x:x + w])
        cv2.imshow('CHỤP ẢNH ', img)
        cv2.waitKey(1)
        # Check xem có bấm q hoặc trên 100 ảnh sample thì thoát
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        elif sampleNum > 20:
            break
    cap.release()
    cv2.destroyAllWindows()
def button_trainmodel():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        # Lấy tất cả các file trong thư mục
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # create empth face list
        faceSamples = []
        # create empty ID list
        Ids = []
        # now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            if (imagePath[-3:] == "jpg"):
                print(imagePath[-3:])
                # loading the image and converting it to gray scale
                pilImage = Image.open(imagePath).convert('L')
                # Now we are converting the PIL image into numpy array
                imageNp = np.array(pilImage, 'uint8')
                # getting the Id from the image
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                # extract the face from the training image sample
                faces = detector.detectMultiScale(imageNp)
                # If a face is there then append that in the list as well as Id of it
                for (x, y, w, h) in faces:
                    faceSamples.append(imageNp[y:y + h, x:x + w])
                    Ids.append(Id)
        return faceSamples, Ids
    # Lấy các khuôn mặt và ID từ thư mục dataSet
    faceSamples, Ids = getImagesAndLabels(r"C:\Users\Nhu\Downloads\Face Recognition\dataset")
    # Train model để trích xuất đặc trưng các khuôn mặt và gán với từng nhan viên
    recognizer.train(faceSamples, np.array(Ids))
    # Lưu model
    recognizer.save(r"C:\Users\Nhu\Downloads\Face Recognition/Recognizer/trainner.yml")
    print("Trained!")
def facerecognizer():
    # Khởi tạo bộ phát hiện khuôn mặt
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml');
    # Khởi tạo bộ nhận diện khuôn mặt
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r"C:\Users\Nhu\Downloads\Face Recognition/Recognizer/trainner.yml")
    id = 0
    # set text style
    fontface = cv2.FONT_HERSHEY_COMPLEX
    fontscale = 1
    fontcolor = (0, 255, 0)
    fontcolor1 = (0, 0, 255)

    def classifyPose(landmarks, output_image, display=False):

        # Initialize the label of the pose. It is not known at this stage.
        label = 'Unknown Pose'

        # Specify the color (Red) with which the label will be written on the image.
        color = (0, 0, 255)

        # Calculate the required angles.
        # ----------------------------------------------------------------------------------------------------------------

        # Get the angle between the left shoulder, elbow and wrist points.
        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        # Get the angle between the right shoulder, elbow and wrist points.
        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        # Get the angle between the left elbow, shoulder and hip points.
        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

        # Get the angle between the right hip, shoulder and elbow points.
        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        # Get the angle between the left hip, knee and ankle points.
        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # Get the angle between the right hip, knee and ankle points
        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        # ----------------------------------------------------------------------------------------------------------------
        # Check if the both arms are straight.
        if (left_elbow_angle > 70 and left_elbow_angle < 195) or (right_elbow_angle > 70 and right_elbow_angle < 195):
            # Check if shoulders are at the required angle.
            if (left_shoulder_angle > 80 and left_shoulder_angle < 110) or (
                    right_shoulder_angle > 80 and right_shoulder_angle < 110):
                # ----------------------------------------------------------------------------------------------------------------
                # Check if it is the Hello pose.
                # ----------------------------------------------------------------------------------------------------------------
                # Check if both legs are straight
                if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                    # Specify the label of the pose that is hello pose.
                    label = 'Hello'
        # Check if the pose is classified successfully
        if label != 'Unknown Pose':
            # Update the color (to green) with which the label will be written on the image.
            color = (0, 255, 0)
            # Write the label on the output image.
        cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        # Check if the resultant image is specified to be displayed.
        if display:
            # Display the resultant image.
            plt.figure(figsize=[10, 10])
            plt.imshow(output_image[:, :, ::-1]);
            plt.title("Output Image");
            plt.axis('off');
        else:
            # Return the output image and the classified label.
            return output_image, label
    def audio(a):
        if(a==10):
            Text = " Chào ông chủ" + str(profile[1])
            print("please wait...processing")
            TTS = gTTS(text=Text, lang='vi')
            # Save to mp3 in current dir.
            TTS.save("voice.mp3")
            playsound("voice.mp3", False)
            os.remove("voice.mp3")
    def unknow(b):
        if (b==10):
            Text = " Có người lạ"
            print("please wait...processing")
            TTS = gTTS(text=Text, lang='vi')
            # Save to mp3 in current dir.
            TTS.save("voice.mp3")
            playsound("voice.mp3", False)
            os.remove("voice.mp3")
    # Hàm lấy thông tin người dùng qua ID
    def getProfile(id):
        conn = sqlite3.connect(r"C:\Users\Nhu\Downloads\Face Recognition\dataset/data.db")
        cursor = conn.execute("SELECT * FROM people WHERE ID=" + str(id))
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile
    # Khởi tạo camera
    cam = cv2.VideoCapture(0);
    cam.set(3, 1280)
    cam.set(4, 960)
    a = 0
    b = 0
    # Initialize a resizable window.
    #cv2.namedWindow('Face', cv2.WINDOW_NORMAL)
    while cam.isOpened():
        #khuon mat
        # Đọc ảnh từ camera
        ret, img = cam.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False) # phát hiện vị trí
        #d = threading.Thread(target=senddata, args=(songontay,))
        # Lật ảnh cho đỡ bị ngược
        img = cv2.flip(img, 1)
        # Vẽ khung chữ nhật để định vị vùng người dùng đưa mặt vào
        centerH = img.shape[0] // 2;
        centerW = img.shape[1] // 2;
        sizeboxW = 300;
        sizeboxH = 400;
        cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                      (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
        # Chuyển ảnh về xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Phát hiện các khuôn mặt trong ảnh camera
        faces = faceDetect.detectMultiScale(gray, 1.3, 5);
        # Lặp qua các khuôn mặt nhận được để hiện thông tin
        for (x, y, w, h) in faces:

            # Vẽ hình chữ nhật quanh mặt
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Nhận diện khuôn mặt, trả ra 2 tham số id: mã nhân viên và dist (dộ sai khác)
            id, dist = recognizer.predict(gray[y:y + h, x:x + w])
            profile = None
            # Nếu độ sai khác < 60% thì lấy profile
            def senddata(songontay):
                if songontay == 0:
                    ref.update({
                        'Data': 0
                    })
                if songontay == 1:
                    ref.update({
                        'Data': 1
                    })
                if songontay == 2:
                    ref.update({
                        'Data': 2
                    })
                if songontay == 3:
                    ref.update({
                        'Data': 3
                    })
                if songontay == 4:
                    ref.update({
                        'Data': 4
                    })
                if songontay == 5:
                    ref.update({
                        'Data': 5
                    })

            if (dist <= 60):
                profile = getProfile(id)
            #Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
            if (profile != None):
                cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), fontface, fontscale, fontcolor, 2)
                a += 1  
                b = 0
                #print(a)
                audio(a)
                # Check if frame is not read properly.
                if not ret:
                    # Continue to the next iteration to read the next frame and ignore the empty camera frame.
                    continue
                # Flip the frame horizontally for natural (selfie-view) visualization.
                # img = cv2.flip(img, 1)
                # Get the width and height of the frame
                frame_height, frame_width, _ = img.shape
                # Resize the frame while keeping the aspect ratio.
                img = cv2.resize(img, (int(frame_width * (640 / frame_height)), 640))
                # Perform Pose landmark detection.
                img, landmarks = detectPose(img, pose_video, display=False)
                # Check if the landmarks are detected.
                if landmarks:
                    # Perform the Pose Classification.
                    img, _ = classifyPose(landmarks, img, display=False)
                    def classifyPose(landmarks, output_image, display=False):
                        # Initialize the label of the pose. It is not known at this stage.
                        label = 'Unknown Pose'
                        # Specify the color (Red) with which the label will be written on the image.
                        color = (0, 0, 255)
                        # Calculate the required angles.
                        # ----------------------------------------------------------------------------------------------------------------
                        # Get the angle between the left shoulder, elbow and wrist points.
                        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
                        # Get the angle between the right shoulder, elbow and wrist points.
                        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                        # Get the angle between the left elbow, shoulder and hip points.
                        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                        # Get the angle between the right hip, shoulder and elbow points.
                        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                        # Get the angle between the left hip, knee and ankle points.
                        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                        # Get the angle between the right hip, knee and ankle points
                        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                        # ---------------------------------------------------------------------------------------------------------------
                        # Check if the both arms are straight.
                        if (left_elbow_angle > 70 and left_elbow_angle < 195) or (
                                right_elbow_angle > 70 and right_elbow_angle < 195):
                            # Check if shoulders are at the required angle.
                            if (left_shoulder_angle > 80 and left_shoulder_angle < 110) or (
                                    right_shoulder_angle > 80 and right_shoulder_angle < 110):
                                # ----------------------------------------------------------------------------------------------------------------
                                # Check if it is the Hello pose.
                                # ----------------------------------------------------------------------------------------------------------------
                                # Check if both legs are straight
                                if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                                    # Specify the label of the pose that is hello pose.
                                    label = 'Hello'
                                    # print(lmList)
                                    if len(lmList) != 0:
                                        fingers = []
                                        if lmList[fingerid[0]][1] < lmList[fingerid[4]][1]:
                                            # viết cho ngón cái (ý tường là điểm 4 ở bên trái hay bên phải điểm 2 )
                                            if lmList[fingerid[0]][1] < lmList[fingerid[0] - 1][1]:
                                                fingers.append(1)
                                            else:
                                                fingers.append(0)
                                            for id in range(1, 5):
                                                if lmList[fingerid[id]][2] < lmList[fingerid[id] - 2][2]:
                                                    fingers.append(1)
                                                else:
                                                    fingers.append(0)
                                        else:
                                            if lmList[fingerid[0]][1] > lmList[fingerid[0] - 1][1]:
                                                fingers.append(1)
                                            else:
                                                fingers.append(0)
                                            for id in range(1, 5):
                                                if lmList[fingerid[id]][2] < lmList[fingerid[id] - 2][2]:
                                                    fingers.append(1)
                                                else:
                                                    fingers.append(0)

                                        # print(fingers)
                                        songontay = fingers.count(1)
                                        print(songontay)
                                        senddata(songontay)
                                        # chú ý mỗi bức ảnh sẽ đẩy về giá trị của 1 mảng có chiều rông, cao khác nhau
                                        # ví dụ ảnh 0.png : print(lst_2[0].shape) kết quả (126, 110, 3)
                                        # frame[0:126,0:110] = lst_2[0]
                                        # do các bức ảnh 0-5.png khác nhau các giá trị wisth, height nên phải get theo shape
                                        h, w, c = lst_2[songontay - 1].shape
                                        img[0:h, 0:w] = lst_2[
                                            songontay - 1]  # nếu số ngón tay =0 thì lst_2[-1] đẩy về phần tử cuối cùng của list là ảnh 6

                                        # vẽ thêm hình chữ nhật hiện số ngón tay
                                        cv2.rectangle(img, (0, 200), (150, 400), (0, 255, 0), -1)
                                        cv2.putText(img, str(songontay), (30, 390), cv2.FONT_HERSHEY_PLAIN, 10,
                                                    (255, 0, 0), 5)
                        # Check if the pose is classified successfully
                        if label != 'Unknown Pose':
                            # Update the color (to green) with which the label will be written on the image.
                            color = (0, 255, 0)
                            # Write the label on the output image.
                        cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                        # Check if the resultant image is specified to be displayed.
                        if display:
                            # Display the resultant image.
                            plt.figure(figsize=[10, 10])
                            plt.imshow(output_image[:, :, ::-1]);
                            plt.title("Output Image");
                            plt.axis('off');
                        else:
                            # Return the output image and the classified label.
                            return output_image, label
            else:
                cv2.putText(img, "Name: Unknown", (x, y + h + 30), fontface, fontscale, fontcolor1, 2)
                a = 0
                b += 1
                #print(b)
                unknow(b)
        #if (a,b ==200):
            #a=b=0
        #ok, frame = camera_video.read(),ok thanh ret,cảmera thành cam
        cv2.imshow('Pose Classification', img)
        if cv2.waitKey(1) == ord('q'):
            break;
    cam.release()
    cv2.destroyAllWindows()
# 1 số nút nhấn và giao diện
root = tk.Tk()
root.title("HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")
canvas = tk.Canvas(root, height=700, width=700)
canvas.pack()
frame = tk.Frame(root, bg="#643c17")
frame.place(relwidth=1, relheight=1)

facerecognizer = tk.Button(frame, text="NHẬN DIỆN", height=3, width=30, bg="#ff56aa", command=facerecognizer)
facerecognizer.place(relx=0.3, rely=0.5)

button_trainmodel = tk.Button(frame, text="TRAINING", height=3, width=30, bg="#ff56aa", command=button_trainmodel)
button_trainmodel.place(relx=0.3, rely=0.3)

button_getdata = tk.Button(frame, text="LẤY THÔNG TIN NGƯỜI NHÀ", height=3, width=30, bg="#ff56aa", command=button_getdata)
button_getdata.place(relx=0.3, rely=0.1)

root.mainloop()


