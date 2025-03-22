import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import pymongo

# 🔹 Kết nối MongoDB
client = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["face_vectors"]

# 🔹 Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# 🔹 Mở camera
cap = cv2.VideoCapture(1)

# 🟢 HÀM: Tìm user trong DB dựa vào vector khuôn mặt
def find_existing_user(face_vector):
    users = collection.find()
    for user in users:
        stored_vector = np.array(user["vector"])
        match = face_recognition.compare_faces([stored_vector], face_vector, tolerance=0.5)
        if match[0]:  
            return user["user_id"]  # Trả về user_id nếu khớp
    return None  # Không tìm thấy

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Chuyển ảnh sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.height * w), int(bboxC.width * w)

            # Vẽ khung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Cắt ảnh khuôn mặt
            face_img = frame[y:y + h, x:x + w]

            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                # 🔹 Chuyển ảnh về dạng vector
                small_rgb_face = cv2.resize(face_img, (150, 150))
                face_encoding = face_recognition.face_encodings(small_rgb_face)

                if len(face_encoding) > 0:
                    face_vector = face_encoding[0]

                    # 🔎 Kiểm tra user
                    found_id = find_existing_user(face_vector)

                    if found_id:
                        cv2.putText(frame, f"User: {found_id}", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(f"[INFO] Nhận diện thành công! User ID: {found_id}")
                    else:
                        cv2.putText(frame, "Unknown", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print("[WARNING] Không tìm thấy user trong database!")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
