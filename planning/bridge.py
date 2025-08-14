# bridge between isaac gym and ROS2 (improved)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import socket
import threading
import json
import time

HOST = '127.0.0.1'
PORT = 9999
RECV_BUF = 4096

class SocketBridgeNode(Node):
    def __init__(self):
        super().__init__('socket_bridge')
        # 发布到 MoveIt 的 sim_state topic（把 isaac 发来的 pose 发布为 ROS）
        self.publisher_ = self.create_publisher(PoseStamped, 'sim_state', 10)

        # 从 ROS 接收要发给 Isaac 的命令（例如抓取命令等）
        self.subscription_ = self.create_subscription(String, 'cmd_target', self.cmd_callback, 10)

        # 从 MoveIt 接收 joint states 并转发给 Isaac
        self.joint_sub_ = self.create_subscription(JointState, 'moveit_joint_states', self.moveit_joint_callback, 10)

        # socket client connection & sync
        self.client_conn = None
        self.conn_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # 启动 socket server 线程 (daemon)
        t = threading.Thread(target=self.start_socket_server, daemon=True)
        t.start()
        self.get_logger().info('SocketBridgeNode started, socket thread launched.')

    def start_socket_server(self):
        """Accept loop: accept a client and then read lines (JSON terminated by newline)."""
        # 创建 server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(1.0)  # 用于及时检查 shutdown_event

        self.get_logger().info(f"Socket server listening on {HOST}:{PORT}")
        while not self.shutdown_event.is_set():
            try:
                self.get_logger().info("Waiting for isaac gym connection (accept)...")
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                with self.conn_lock:
                    self.client_conn = conn
                    self.client_conn.settimeout(1.0)
                self.get_logger().info(f"Connected by {addr}")

                buffer = ""
                # 接收循环
                while not self.shutdown_event.is_set():
                    try:
                        recv = conn.recv(RECV_BUF)
                        if not recv:
                            self.get_logger().warn("Connection closed by client.")
                            break
                        text = recv.decode('utf-8')
                        buffer += text
                        # 每行是一个 JSON 消息（以换行结束）
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                            except Exception as e:
                                self.get_logger().error(f"JSON decode error: {e} -- line: {line}")
                                continue

                            # 处理不同类型的消息（来自 Isaac）
                            mtype = msg.get("type", "")
                            if mtype == "state":
                                self.handle_state_message(msg.get("data", {}))
                            elif mtype == "joint_state":
                                # 如果 Isaac 也发 joint_state（双向可用）
                                self.handle_joint_state_message(msg.get("data", {}))
                            else:
                                self.get_logger().info(f"Received unknown type from client: {mtype}")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        self.get_logger().error(f"Socket recv error: {e}")
                        break

                # 清理断开的连接
                with self.conn_lock:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    if self.client_conn is conn:
                        self.client_conn = None
                self.get_logger().info("Client connection closed, waiting for new client...")
            except Exception as e:
                self.get_logger().error(f"Accept loop error: {e}")
                time.sleep(1.0)

        # 退出时关闭 server
        try:
            server.close()
        except Exception:
            pass
        self.get_logger().info("Socket accept thread exiting.")

    def handle_state_message(self, data):
        """把从 Isaac 发来的 'state' 转为 PoseStamped 发布到 sim_state."""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()

            pos = data.get("position", [0.0, 0.0, 0.0])
            rot = data.get("rotation", [0.0, 0.0, 0.0, 1.0])

            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])

            pose_msg.pose.orientation.x = float(rot[0])
            pose_msg.pose.orientation.y = float(rot[1])
            pose_msg.pose.orientation.z = float(rot[2])
            pose_msg.pose.orientation.w = float(rot[3])

            self.publisher_.publish(pose_msg)
            self.get_logger().debug(f"Published PoseStamped from client: pos={pos}, rot={rot}")
        except Exception as e:
            self.get_logger().error(f"handle_state_message error: {e}")

    def handle_joint_state_message(self, data):
        """可选：如果 Isaac 主动发 joint_state 给 bridge，允许桥把它转为 ROS 并发布（这里不实现 ROS 发布，按需扩展）。"""
        self.get_logger().debug("Received joint_state from client (ignored by default).")

    def cmd_callback(self, msg: String):
        """当 ROS 侧发来要给 Isaac 的命令时（String，内容为 JSON），原样封装并发给 TCP 客户端。"""
        payload = None
        try:
            # msg.data 可能就是 JSON 字符串，也可能是字典的 str 表示
            payload = json.loads(msg.data)
        except Exception:
            # 退回到原始字符串
            payload = msg.data

        out = {"type": "cmd", "data": payload}
        self._send_json(out)

    def moveit_joint_callback(self, msg: JointState):
        """把 MoveIt 发来的 JointState 转发给 Isaac Gym（JSON 格式）。"""
        try:
            # 调试输出，确认收到 MoveIt 数据
            self.get_logger().info(f"[MoveIt→Bridge] JointState received: "
                                f"names={list(msg.name)}, pos={list(msg.position)}")

            data = {
                "name": list(msg.name),
                "position": [float(x) for x in msg.position],
                "velocity": [float(x) for x in msg.velocity] if len(msg.velocity) > 0 else [],
                "effort": [float(x) for x in msg.effort] if len(msg.effort) > 0 else []
            }
            out = {"type": "joint_state", "data": data}

            # 调试输出，确认发送数据
            self.get_logger().info(f"[Bridge→Isaac] Sending joint_state JSON: {out}")

            self._send_json(out)
            self.get_logger().debug("Forwarded moveit_joint_states to client.")
        except Exception as e:
            self.get_logger().error(f"moveit_joint_callback error: {e}")


    def _send_json(self, obj):
        if not self.client_conn:
            self.get_logger().warn("No client connected, cannot send message.")
            return False
        try:
            text = json.dumps(obj) + '\n'
            self.client_conn.sendall(text.encode('utf-8'))
            return True
        except Exception as e:
            self.get_logger().error(f"Error sending to client: {e}")
            try:
                self.client_conn.close()
            except Exception:
                pass
            self.client_conn = None
            return False

    def destroy_node(self):
        """优雅关闭：先停止 socket 线程，再调用基类 destroy_node."""
        self.get_logger().info("Shutting down SocketBridgeNode...")
        self.shutdown_event.set()
        # 关闭连接
        with self.conn_lock:
            try:
                if self.client_conn:
                    self.client_conn.close()
            except Exception:
                pass
            self.client_conn = None
        # 给线程一点时间退出
        time.sleep(0.2)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SocketBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
