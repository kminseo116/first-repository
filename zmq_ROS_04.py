import zmq
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

# 토픽 설정
DEFAULT_ROS_DETECTED = {
    'safebox': '/object_detection/safebox/safebox',
    'human_red': '/object_detection/human/red',
    'human_blue': '/object_detection/human/blue',
    'finish': '/object_detection/finish',
}

DEFAULT_ROS_DRIVE = {
    'safebox': '/drive/safebox_detection_again',
    'human_1': '/drive/human_start_1',
    'human_2': '/drive/human_start_2',
    'human_3': '/drive/human_start_3',
    'finish': '/drive/finish_start',
}

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_publisher_node')

        # ROS 퍼블리셔 생성 (Empty 메시지)
        self.safebox_1_publisher = self.create_publisher(
            String, 
            DEFAULT_ROS_DETECTED['safebox'], 
            10)

        self.human_red_publisher = self.create_publisher(
            Empty, 
            DEFAULT_ROS_DETECTED['human_red'], 
            10)
        
        self.human_blue_publisher = self.create_publisher(
            Empty, 
            DEFAULT_ROS_DETECTED['human_blue'], 
            10)
        
        self.finish_publisher = self.create_publisher(
            Empty, 
            DEFAULT_ROS_DETECTED['finish'], 
            10)

        # ROS 서브스크라이버 생성
        self._subscriber = self.create_subscription(
            Empty,
            DEFAULT_ROS_DRIVE['safebox'],
            self.safebox_start_again_callback,
            10)
        
        self.fire_subscriber = self.create_subscription(
            Empty,
            DEFAULT_ROS_DRIVE['human_1'],
            self.human_start_1_callback,
            10)
        
        self.door_subscriber = self.create_subscription(
            Empty,
            DEFAULT_ROS_DRIVE['human_2'],
            self.human_start_2_callback,
            10)
        
        self.door_subscriber = self.create_subscription(
            Empty,
            DEFAULT_ROS_DRIVE['human_3'],
            self.human_start_3_callback,
            10)
        
        self.door_subscriber = self.create_subscription(
            Empty,
            DEFAULT_ROS_DRIVE['finish'],
            self.finish_start_callback,
            10)

        # ZeroMQ 컨텍스트 생성
        try:
            self.zmq_context = zmq.Context()

            # 탐지 소켓 (SUB)
            self.socket_detection = self.zmq_context.socket(zmq.SUB)
            self.socket_detection.connect("tcp://localhost:5556")
            self.socket_detection.setsockopt_string(zmq.SUBSCRIBE, "/object_detection/")

            # Drive 소켓 (PUB)
            self.socket_drive = self.zmq_context.socket(zmq.PUB)
            self.socket_drive.bind("tcp://*:5557")

            self.get_logger().info('ZeroMQ 컨텍스트 및 소켓 초기화 완료')

        except Exception as e:
            self.get_logger().error(f'ZeroMQ 초기화 실패: {str(e)}')
            raise

    def safebox_start_again_callback(self, msg):
        try:
            self.socket_drive.send_string("safebox_detection_again")
            self.get_logger().info('ZMQ 발행: safebox_detection_again')
        except Exception as e:
            self.get_logger().error(f'ZMQ safebox_detection_again 발행 실패: {str(e)}')

    def human_start_1_callback(self, msg):
        try:
            self.socket_drive.send_string("human_start_1")
            self.get_logger().info('ZMQ 발행: human_start_1')
        except Exception as e:
            self.get_logger().error(f'ZMQ human_start_1 발행 실패: {str(e)}')

    def human_start_2_callback(self, msg):
        try:
            self.socket_drive.send_string("human_start_2")
            self.get_logger().info('ZMQ 발행: human_start_2')
        except Exception as e:
            self.get_logger().error(f'ZMQ human_start_2 발행 실패: {str(e)}')

    def human_start_3_callback(self, msg):
        try:
            self.socket_drive.send_string("human_start_3")
            self.get_logger().info('ZMQ 발행: human_start_3')
        except Exception as e:
            self.get_logger().error(f'ZMQ human_start_3 발행 실패: {str(e)}')

    def finish_start_callback(self, msg):
        try:
            self.socket_drive.send_string("finish_start")
            self.get_logger().info('ZMQ 발행: finish_start')
        except Exception as e:
            self.get_logger().error(f'ZMQ finish_start 발행 실패: {str(e)}')

    def listen_and_publish(self):
        poller = zmq.Poller()
        poller.register(self.socket_detection, zmq.POLLIN)

        while rclpy.ok():
            try:
                socks = dict(poller.poll(timeout=10))  # 10ms 타임아웃
                while self.socket_detection in socks and socks[self.socket_detection] == zmq.POLLIN:
                    topic, data = self.socket_detection.recv_multipart(flags=zmq.NOBLOCK)
                    topic = topic.decode()

                    if topic in DEFAULT_ROS_DETECTED.values():  # final 토픽만 처리 (temp 무시)
                        if topic == "/object_detection/safebox/safebox":
                            msg = String()
                            msg.data = data.decode()
                            self.safebox_1_publisher.publish(msg)
                            self.get_logger().info(msg.data)

                        elif topic == "/object_detection/human/red":
                            msg = Empty()
                            self.human_red_publisher.publish(msg)
                            self.get_logger().info('ROS 발행: red_detected')

                        elif topic == "/object_detection/human/blue":
                            msg = Empty()
                            self.human_blue_publisher.publish(msg)
                            self.get_logger().info('ROS 발행: blue_detected')

                        elif topic == "/object_detection/finish":
                            msg = Empty()
                            self.finish_publisher.publish(msg)
                            self.get_logger().info('ROS 발행: finish_detected')

                    # 임시 토픽(/button_temp, /fire/on_temp 등)은 무시
                    socks = dict(poller.poll(timeout=0))
                    
            except zmq.Again:
                pass
            
            except Exception as e:
                self.get_logger().error(f'메시지 수신 또는 발행 실패: {str(e)}')
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        node.listen_and_publish()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()