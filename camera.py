import cv2
import numpy as np
import functions as f

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

victor = f.load_face_image("images/faceRecognition/victor1.jpg")
# victor = f.load_face_image("images/faceRecognition/obama.png")
victor_encoding = f.get_face_descriptor(victor)[0]
procces_frame = True

face_locations = []
known_names = ["Victor"]
known_encodings = [victor_encoding]
frame_face_encodings = []
frame_names = []
while True:
    ret, frame = video.read()
    # frame = f.gaussian_filter(frame, 2, 1.6)
    frame = cv2.flip(frame, 1)
    # frame = f.lut_chart(frame,  1)
    if procces_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = f.get_face_locations(rgb_frame)
        if len(face_locations) > 0:
            frame_face_encodings = f.get_face_descriptor(rgb_frame, face_locations)

            for i in range(len(frame_face_encodings)):
                matches = f.compare_faces(known_encodings, frame_face_encodings[i])
                # print(matches)
                name = "unknown"

                if True in matches:
                    face_distances = f.get_face_difference(known_encodings, frame_face_encodings[i])
                    best_match = np.argmin(face_distances)

                    if matches[best_match]:
                        name = known_names[best_match]

                if i >= len(frame_names):
                    frame_names.append(name)
                else:
                    frame_names[i] = name

    print(frame_names)
    procces_frame = not procces_frame

    for (top, right, bottom, left), name in zip(face_locations, frame_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        print(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
