import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import zmq
import base64

# 토픽 설정
DEFAULT_ROS_DETECTED = {
    'safebox': '/object_detection/safebox/safebox',
    'human_red': '/object_detection/human/red',
    'human_blue': '/object_detection/human/blue',
    'finish': '/object_detection/finish',
}

# ZeroMQ 컨텍스트
context = zmq.Context()
socket_video = context.socket(zmq.PUB)
socket_video.bind("tcp://*:5555")  # 영상 전송 to zmq_SUB.py

# 탐지 결과 발행용 단일 PUB 소켓
socket_detection = context.socket(zmq.PUB)
socket_detection.bind("tcp://*:5556")

# 드라이브 토픽 수신용 단일 SUB 소켓
socket_drive = context.socket(zmq.SUB)
socket_drive.connect("tcp://localhost:5557")
for topic in ['safebox_detection_again', 'human_start_1', 'human_start_2', 'human_start_3', 'finish_start']:
    socket_drive.setsockopt_string(zmq.SUBSCRIBE, topic)

# 모델 로드
safebox_model = YOLO('/home/kminseo/Documents/mission/safebox/runs/detect/train/weights/last.pt')
human_model = YOLO('/home/kminseo/Documents/mission/Fire/train2/weights/last.pt')    # 문으로 테스트
color_model = YOLO('/home/kminseo/Documents/mission/Button/runs/detect/train/weights/last.pt')  # 일단 버튼으로 테스트

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
    'safebox': True,
    'human': False,
    'finish': False
}

# 토픽별 발행 횟수 카운터
detection_count = {
    'human_red': 0,
    'human_blue': 0,
    'finish': 0
}
required_counts = 50  # 30번 발행 필요

# 최종 토픽 단일 발행 추적
published_once = {
    'human_red': False,
    'human_blue': False,
    'finish': False
}

# ZMQ 폴러 설정
poller = zmq.Poller()
poller.register(socket_drive, zmq.POLLIN)

# 탐지 처리 함수
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
                    detection_enabled = {'safebox': True, 'human': False, 'finish': False}
                    detection_count = {'safebox': 0, 'human_red': 0, 'human_blue': 0, 'finish': 0}
                    published_once = {'human_red': False, 'human_blue': False, 'finish': False}
                    print("Reset all flags and counters")              

                else:
                    if topic == "safebox_detection_again":
                        detection_enabled['safebox'] = False
                        print("Safebox detection disabled")

                    elif topic == "human_start_1":
                        detection_enabled['human'] = True
                        print("human_1 detection enabled")

                    elif topic == "human_start_2":
                        detection_enabled['human'] = True
                        print("human_2 detection enabled")

                    elif topic == "human_start_3":
                        detection_enabled['human'] = True
                        print("human_3 detection enabled")

                    elif topic == "finish_start":
                        detection_enabled['finish'] = True
                        print("Finish detection enabled")

            except zmq.ZMQError as e:
                print(f"ZMQ 수신 오류: {str(e)}")

        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 색상 프레임을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 탐지 플래그
        any_detection = False

        safebox_detected = False
        human_red_detected = False
        human_blue_detected = False
        finish_detected = False

        safebox_max_conf = 0
        human_red_max_conf = 0
        human_blue_max_conf = 0
        finish_max_conf = 0

        if detection_enabled['safebox']:
            results_safebox = safebox_model(source=color_image, verbose=False)
            
            for result in results_safebox:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence < 0.85:
                        continue
                    
                    any_detection = True
                    safebox_detected = True
                    
                    safebox_max_conf = max(safebox_max_conf, confidence)

                    # 바운딩 박스 중심 좌표 계산
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2

                    # 객체 중심의 깊이 값 가져오기 (mm 단위)
                    obj_depth_value = depth_frame.get_distance(obj_center_x, obj_center_y) * 1000  # mm 변환

                    # safebox 데이터 string으로 구성 및 즉시 발행
                    safebox_data = f"x:{obj_center_x} depth:{obj_depth_value:.2f}".encode()
                    socket_detection.send_multipart([DEFAULT_ROS_DETECTED['safebox'].encode(), safebox_data])
                    print(f"Safebox detection sent (x: {obj_center_x}, depth: {obj_depth_value:.2f}, conf: {confidence:.2f})")

                    label = f"id: {class_id} conf: {confidence:.2f} Depth: {obj_depth_value:.2f}mm X: {obj_center_x}"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0) ,2, cv2.LINE_AA)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

        # 마네킹 모델 처리
        if detection_enabled['human']:
            results_human = human_model(source=color_image, verbose=False)

            for result in results_human:
                boxes = result.boxes
                masks = result.masks if hasattr(result, 'masks') else None

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

                            human_map = {0: ("red", (0, 0, 255)), 1: ("blue", (255, 0, 0))}
                            human_status, box_color = human_map.get(class_id, ("unknown", (0, 0, 255)))

                            if human_status == "unknown":

                                print(f"Invalid fire class_id {class_id}, skipping")
                                continue

                            label = f"id: {class_id} conf: {confidence:.2f} fire: {human_status}"
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                            cv2.putText(color_image, label, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

                            if human_status == "red":
                                human_red_detected = True
                                human_red_max_conf = max(human_red_max_conf, confidence)
                    
                            elif human_status == "blue":
                                human_blue_detected = True
                                human_blue_max_conf = max(human_blue_max_conf, confidence)

        # color 모델 결과 처리
        if detection_enabled['finish']:
            results_color = color_model(source=color_image, verbose=False)
            
            for result in results_color:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence < 0.85 :
                        continue
                    
                    any_detection = True
                    finish_detected = True
                    
                    finish_max_conf = max(finish_max_conf, confidence)

                    label = f"FINISH conf: {confidence:.2f}"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), 2, cv2.LINE_AA)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0, 255, 0), 2)

        # 프레임당 탐지 토픽 발행 및 카운트
        handle_detection(
            detected=human_red_detected,
            max_conf=human_red_max_conf,
            key='human_red',
            temp_topic='/object_detection/human/red_temp',
            final_topic='human_red',
            published_once=published_once,
            detection_count=detection_count,
            socket_detection=socket_detection
        )

        handle_detection(
            detected=human_blue_detected,
            max_conf=human_blue_max_conf,
            key='human_blue',
            temp_topic='/object_detection/human/blue_temp',
            final_topic='human_blue',
            published_once=published_once,
            detection_count=detection_count,
            socket_detection=socket_detection
        )

        handle_detection(
            detected=finish_detected,
            max_conf=finish_max_conf,
            key='finish',
            temp_topic='/object_detection/finish_temp',
            final_topic='finish',
            published_once=published_once,
            detection_count=detection_count,
            socket_detection=socket_detection
        )

        if not any_detection:
           cv2.putText(color_image, "Looking For", (35, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        
        # base64로 인코딩하여 영상 전송
        _, buffer = cv2.imencode('.jpg', color_image)
        jpg_as_text = base64.b64encode(buffer)
        socket_video.send(jpg_as_text)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

