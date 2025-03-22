import cv2
import mediapipe as mp
import os
import time
import uuid
import face_recognition
import numpy as np
import pymongo

# 🔹 Kết nối MongoDB
client = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["face_vectors"]

# 🔹 Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# 🔹 Thư mục lưu khuôn mặt
output_dir = "detected_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 🔹 Mở camera (0: webcam mặc định, 1: iVCam)
cap = cv2.VideoCapture(0)

# Lưu trạng thái nhận diện
last_saved_time = 0

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

            # Độ tin cậy
            confidence = round(detection.score[0] * 100, 2)
            label = f"Face: {confidence}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Cắt ảnh khuôn mặt
            face_img = frame[y:y + h, x:x + w]

            # Kiểm tra ảnh có hợp lệ không
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                # 🔹 Chuyển ảnh về dạng vector
                small_rgb_face = cv2.resize(face_img, (150, 150))
                face_encoding = face_recognition.face_encodings(small_rgb_face)

                if len(face_encoding) > 0:
                    face_vector = face_encoding[0]

                    # 🔎 Kiểm tra xem có user nào đã đăng ký với khuôn mặt này chưa
                    found_id = find_existing_user(face_vector)

                    # 🔹 Nếu chưa có, tạo user_id mới và lưu vào MongoDB
                    if found_id is None:
                        found_id = str(uuid.uuid4())[:8]  # Tạo ID mới
                        collection.insert_one({
                            "user_id": found_id,
                            "vector": face_vector.tolist(),
                            "created_at": time.time()
                        })
                        print(f"[NEW USER] Đã thêm user {found_id} vào database!")

                    # Chỉ lưu ảnh nếu thời gian cách lần trước >= 3 giây
                    current_time = time.time()
                    if current_time - last_saved_time > 3:
                        filename = os.path.join(output_dir, f"{found_id}_{confidence}.jpg")
                        cv2.imwrite(filename, face_img)
                        last_saved_time = current_time
                        print(f"[INFO] Lưu ảnh {filename} cho user {found_id}")

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
