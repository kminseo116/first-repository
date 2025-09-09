import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import zmq
import base64
import time

# 토픽 설정
DEFAULT_ROS_DETECTED = {
    'button': '/object_detection/button',
    'fire_on': '/object_detection/fire/on',
    'fire_off': '/object_detection/fire/off',
    'door': '/object_detection/door',
}

# ZeroMQ 컨텍스트
context = zmq.Context()

# 영상 전송용 PUB 소켓
socket_video = context.socket(zmq.PUB)
socket_video.bind("tcp://*:5555")

# 탐지 결과 발행용 단일 PUB 소켓
socket_detection = context.socket(zmq.PUB)
socket_detection.bind("tcp://*:5556")

# 드라이브 토픽 수신용 단일 SUB 소켓
socket_drive = context.socket(zmq.SUB)
socket_drive.connect("tcp://localhost:5557")
for topic in ['button_start', 'fire_start', 'door_start', 'reset']:
    socket_drive.setsockopt_string(zmq.SUBSCRIBE, topic)

# 모델 로드
button_model = YOLO('/home/kminseo/Documents/mission/Button/runs/detect/train/weights/last.pt')
fire_model = YOLO('/home/kminseo/Documents/mission/Fire/train2/weights/last.pt')
door_model = YOLO('/home/kminseo/Documents/mission/Door/runs/weights/last.pt')

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Color 프레임에 맞춰 정렬
align_to = rs.stream.color
align = rs.align(align_to)

# 탐지 활성화 플래그
detection_enabled = {
    'button': False,
    'fire': False,
    'door': False
}

# 토픽별 발행 횟수 카운터 (버튼과 문에만 사용)
detection_count = {
    'button': 0,
    'door': 0,
}
required_counts = 50  # 버튼과 문에만 적용

# 최종 토픽 단일 발행 추적 (버튼과 문에만 사용)
published_once = {
    'button': False,
    'door': False,
}

# 화재 캡처 관련 변수
fire_capture_count = 0
fire_last_capture_time = 0
fire_capture_interval = 3  # 초

# ZMQ 폴러 설정
poller = zmq.Poller()
poller.register(socket_drive, zmq.POLLIN)

# 탐지 처리 함수 (버튼과 문에만 사용)
def handle_detection(detected, max_conf, key, temp_topic, final_topic, published_once, detection_count, socket_detection):
    if detected and max_conf >= 0.85:
        detection_count[key] += 1
        socket_detection.send_multipart([temp_topic.encode(), b''])
        print(f"{key.capitalize()} temp detection sent ({detection_count[key]}/{required_counts})")
        
        if detection_count[key] >= required_counts and not published_once[final_topic]:
            socket_detection.send_multipart([DEFAULT_ROS_DETECTED[final_topic].encode(), b''])
            published_once[final_topic] = True
            print(f"Sent final {DEFAULT_ROS_DETECTED[final_topic]} (conf: {max_conf:.2f}, counts: {detection_count[key]})")
    else:
        detection_count[key] = 0
        print(f"{key.capitalize()} detection reset (conf: {max_conf:.2f})")

try:
    while True:
        # 드라이브 토픽 확인 (논블록킹)
        socks = dict(poller.poll(timeout=10))
        if socket_drive in socks:
            try:
                topic = socket_drive.recv_string()
                if topic == "reset":
                    detection_enabled = {key: False for key in detection_enabled}
                    detection_count = {key: 0 for key in detection_count}
                    published_once = {key: False for key in published_once}
                    fire_capture_count = 0
                    fire_last_capture_time = 0
                    print("Reset all flags and counters")
                else:
                    detection_enabled = {key: False for key in detection_enabled}
                    detection_count = {key: 0 for key in detection_count}
                    published_once = {key: False for key in published_once}
                    fire_capture_count = 0
                    fire_last_capture_time = 0

                    if topic == "button_start":
                        detection_enabled['button'] = True
                        print("Button detection enabled")

                    elif topic == "fire_start":
                        detection_enabled['fire'] = True
                        fire_last_capture_time = time.time() - fire_capture_interval  # 첫 캡처 즉시 실행
                        print("Fire detection enabled")

                    elif topic == "door_start":
                        detection_enabled['door'] = True
                        print("Door detection enabled")

            except zmq.ZMQError as e:
                print(f"ZMQ 수신 오류: {str(e)}")

        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # 탐지 플래그
        any_detection = False
        
        button_detected = False
        door_detected = False
        button_max_conf = 0
        door_max_conf = 0

        # 버튼 모델 처리
        if detection_enabled['button']:
            results_button = button_model(source=color_image, verbose=False)
            
            for result in results_button:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence < 0.85 or class_id not in [0, 1, 2]:
                        continue
                    
                    any_detection = True
                    button_detected = True
                    
                    button_max_conf = max(button_max_conf, confidence)
                    
                    color_map = {0: ("blue", (255, 0, 0)), 1: ("green", (0, 255, 0)), 2: ("yellow", (0, 255, 255))}
                    color, box_color = color_map.get(class_id, ("unknown", (0, 0, 255)))
                    
                    label = f"id: {class_id} conf: {confidence:.2f} color: {color}"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                    cv2.putText(color_image, label, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

        # 문 모델 처리
        if detection_enabled['door']:
            results_door = door_model(source=color_image, verbose=False)

            for result in results_door:
                boxes = result.boxes
                masks = result.masks

                if masks is not None:
                    masks = masks.data.cpu().numpy()
                    for i, mask in enumerate(masks):
                        if i < len(boxes):
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                            confidence = boxes.conf[i].cpu().numpy()
                            class_id = int(boxes.cls[i].cpu().numpy())

                            if confidence < 0.85:
                                continue

                            any_detection = True
                            door_detected = True

                            door_max_conf = max(door_max_conf, confidence)

                            # 노이즈 제거 및 연속성 강화
                            mask_binary = (mask > 0.8).astype(np.uint8)
                            kernel = np.ones((3, 3), np.uint8)
                            mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
                            mask_binary = cv2.erode(mask_binary, kernel, iterations=1)
                            
                            # 윤곽선 추출
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
                                largest_contour = max(contours, key=cv2.contourArea)

                            label = f"id: {class_id} conf: {confidence:.2f}"
                            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

        # 버튼 및 문 탐지 결과 처리
        handle_detection(
            detected=button_detected,
            max_conf=button_max_conf,
            key='button',
            temp_topic='/object_detection/button_temp',
            final_topic='button',
            published_once=published_once,
            detection_count=detection_count,
            socket_detection=socket_detection
        )

        handle_detection(
            detected=door_detected,
            max_conf=door_max_conf,
            key='door',
            temp_topic='/object_detection/door_temp',
            final_topic='door',
            published_once=published_once,
            detection_count=detection_count,
            socket_detection=socket_detection
        )

        # 화재 캡처 및 처리 (3초 간격으로 3번)
        current_time = time.time()
        if detection_enabled['fire'] and current_time - fire_last_capture_time >= fire_capture_interval:
            fire_capture_count += 1
            print(f"Capturing image {fire_capture_count}")
            
            # 화재 탐지 수행
            results_fire = fire_model(source=color_image, verbose=False)
            fire_status = None
            max_conf = 0

            for result in results_fire:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    if confidence < 0.8:
                        continue

                    any_detection = True

                    fire_map = {0: ("off", (255, 255, 255)), 1: ("on", (0, 0, 255))}
                    fire_status, box_color = fire_map.get(class_id, ("unknown", (0, 0, 255)))

                    if fire_status == "unknown":
                        print(f"Invalid fire class_id {class_id} in image {fire_capture_count}, skipping")
                        continue

                    max_conf = max(max_conf, confidence)

                    label = f"id: {class_id} conf: {confidence:.2f} fire: {fire_status}"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                    cv2.putText(color_image, label, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

            # 각 캡처의 탐지 결과에 따라 즉시 토픽 발행
            if fire_status in ["on", "off"]:
                topic_to_send = DEFAULT_ROS_DETECTED[f'fire_{fire_status}']
                socket_detection.send_multipart([topic_to_send.encode(), b''])
                print(f"Sent {topic_to_send} for image {fire_capture_count} (conf: {max_conf:.2f})")

            fire_last_capture_time = current_time

            if fire_capture_count >= 3:
                detection_enabled['fire'] = False
                fire_capture_count = 0
                print("Fire detection completed and disabled")

        if not any_detection:
            cv2.putText(color_image, "Looking For", (35, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # base64로 인코딩하여 영상 전송
        _, buffer = cv2.imencode('.jpg', color_image)
        jpg_as_text = base64.b64encode(buffer)
        socket_video.send(jpg_as_text)

finally:
    socket_detection.close()
    socket_drive.close()
    socket_video.close()
    context.term()
    pipeline.stop()
    cv2.destroyAllWindows()