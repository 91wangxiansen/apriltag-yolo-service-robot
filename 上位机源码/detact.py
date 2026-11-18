import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import json
import time
from pupil_apriltags import Detector
from typing import Dict, List, Tuple, Optional


class AprilTagPoseSolver:
    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None, ground_width: float = 1.0, ground_height: float = 1.0):
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [1025.7742, 0.0, 684.9937],
                [0.0, 1028.7293, 297.3318],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix.astype(np.float64)

        if dist_coeffs is None:
            self.dist_coeffs = np.array([0.116388, -0.248768, -0.021682, 0.011116, 0.385795], dtype=np.float64)
        else:
            self.dist_coeffs = dist_coeffs.astype(np.float64)

        self.detector = Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        self.rvec = None
        self.tvec = None
        self.rotation_matrix = None
        self.ground_width = ground_width
        self.ground_height = ground_height
        self.is_pose_solved = False

    def detect_tags(self, gray_img: np.ndarray) -> List:
        return self.detector.detect(gray_img)

    def solve_pose_from_tags(self, detections: List, required_tag_count: int = 4) -> bool:
        tag_map = {}
        for det in detections:
            tag_id = det.tag_id
            if 0 <= tag_id <= 3:
                tag_map[tag_id] = det.center

        if len(tag_map) < required_tag_count:
            return False

        image_points = []
        for i in range(4):
            if i in tag_map:
                image_points.append(tag_map[i])

        if len(image_points) != 4:
            return False

        image_points = np.array(image_points, dtype=np.float64)

        w = self.ground_width * 1000.0
        h = self.ground_height * 1000.0

        object_points = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0]
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

        if success:
            self.rvec = rvec
            self.tvec = tvec
            self.rotation_matrix, _ = cv2.Rodrigues(rvec)
            self.is_pose_solved = True
            return True
        return False

    def coordinate_image_to_world(self, pixel_point: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.is_pose_solved or self.rvec is None:
            return None

        u, v = float(pixel_point[0]), float(pixel_point[1])

        uv_pt = np.array([u, v, 1], dtype=np.float64).reshape(3, 1)

        rot_inv = np.linalg.inv(self.rotation_matrix)
        cam_inv = np.linalg.inv(self.camera_matrix)

        M_1 = rot_inv @ cam_inv @ uv_pt
        M_2 = rot_inv @ self.tvec

        z_const = 0.0
        S = (z_const + M_2[2, 0]) / M_1[2, 0]

        world_pt = rot_inv @ (S * cam_inv @ uv_pt - self.tvec)

        world_pt_meters = world_pt.flatten() / 1000.0

        return world_pt_meters

    def draw_coordinate_system(self, img: np.ndarray, origin_pixel: Tuple[int, int] = None):
        if not self.is_pose_solved:
            return img

        axis_length_mm = 200.0
        reference_obj = np.array([
            [0.0, 0.0, 0.0],
            [axis_length_mm, 0.0, 0.0],
            [0.0, axis_length_mm, 0.0],
            [0.0, 0.0, axis_length_mm]
        ], dtype=np.float64)

        ref_img_pts, _ = cv2.projectPoints(
            reference_obj,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        ref_img_pts = ref_img_pts.astype(int).reshape(-1, 2)

        origin = tuple(map(int, ref_img_pts[0]))

        cv2.line(img, origin, tuple(map(int, ref_img_pts[1])), (0, 0, 255), 3)
        cv2.putText(img, "X", tuple(map(int, ref_img_pts[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.line(img, origin, tuple(map(int, ref_img_pts[2])), (0, 255, 0), 3)
        cv2.putText(img, "Y", tuple(map(int, ref_img_pts[2])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.line(img, origin, tuple(map(int, ref_img_pts[3])), (255, 0, 0), 3)
        cv2.putText(img, "Z", tuple(map(int, ref_img_pts[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

    def draw_detected_tags(self, img: np.ndarray, detections: List):
        for det in detections:
            corners = det.corners.astype(int)
            center = det.center.astype(int)

            cv2.polylines(img, [corners], True, (0, 255, 255), 2)
            cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)

            info_text = f"ID:{det.tag_id}"
            cv2.putText(img, info_text, (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if self.is_pose_solved:
                world_pos = self.coordinate_image_to_world(tuple(center))
                if world_pos is not None:
                    pos_text = f"({world_pos[0]:.2f}, {world_pos[1]:.2f})m"
                    cv2.putText(img, pos_text, (center[0] + 10, center[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return img

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detect_tags(gray)

        result_img = frame.copy()
        result_img = self.draw_detected_tags(result_img, detections)

        pose_success = self.solve_pose_from_tags(detections)

        if pose_success:
            result_img = self.draw_coordinate_system(result_img)

        return result_img, pose_success


# ===================== 【实时输出给MCP脚本】 =====================
def send_status_to_mcp(is_fall, hand_raised, person_foot_center=None, robot_position=None):
    status = {
        "fall_detected": bool(is_fall),
        "hand_raised": bool(hand_raised),
        "person_foot_center": person_foot_center,  # {"x": xx, "y": yy} or None
        "robot_position": robot_position,  # {"x": xx, "y": yy} or None
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "fall_detection"
    }
    with open("robot_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


# ===================== 模型加载 =====================
model_pose = YOLO(r"yolo11n-pose.pt")
model_detect = YOLO(r"D:\deeplearning\ultralytics-8.3.163\model\train\weights\best.pt")

CONF_THRESH = 0.8
FALL_FRAMES = 5
HAND_FRAMES = 3

# 【重要】这里改成你训练的机器人类别ID（通常是0，如果不是请修改）
ROBOT_CLASS_ID = 0

fall_history = deque(maxlen=FALL_FRAMES)
hand_history = deque(maxlen=HAND_FRAMES)

# 初始化 AprilTag 位姿解算器
solver = AprilTagPoseSolver(ground_width=2.722, ground_height=1.61)

results = model_pose(source=0, stream=True)

for result in results:
    frame = result.orig_img.copy()
    h, w = frame.shape[:2]
    pose_img = frame.copy()
    is_fall = False
    hand_raised = False
    trunk_angle = 0.0
    bbox_ratio = 0.0
    person_foot_center = None
    person_foot_pixel = None
    robot_position = None
    robot_pixel = None

    # ===================== AprilTag 检测和解算位姿 =====================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = solver.detect_tags(gray)
    pose_success = solver.solve_pose_from_tags(detections)

    # ===================== 机器人检测 =====================
    detect_res = model_detect(frame, verbose=False)
    detect_img = detect_res[0].plot()

    # 绘制 AprilTag 和坐标系
    detect_img = solver.draw_detected_tags(detect_img, detections)
    if pose_success:
        detect_img = solver.draw_coordinate_system(detect_img)

    # ========== 【修复BUG】只检测 Robot 类别，过滤掉人/其他目标 ==========
    if len(detect_res[0].boxes) > 0:
        # 获取所有检测框的 类别ID、坐标、置信度
        cls_ids = detect_res[0].boxes.cls.cpu().numpy()
        boxes_xyxy = detect_res[0].boxes.xyxy.cpu().numpy()
        confs = detect_res[0].boxes.conf.cpu().numpy()

        # 遍历所有检测框，只找类别为 ROBOT_CLASS_ID 的目标
        for i in range(len(cls_ids)):
            if int(cls_ids[i]) == ROBOT_CLASS_ID and confs[i] > CONF_THRESH:
                # 找到机器人，提取坐标
                robot_box = boxes_xyxy[i]
                x1, y1, x2, y2 = robot_box
                box_height = y2 - y1
                box_center_x = (x1 + x2) / 2
                # 底端向上 1/3 高度处
                robot_y = y2 - box_height / 3
                robot_pixel = {
                    "x": float(box_center_x),
                    "y": float(robot_y)
                }
                robot_position = {
                    "x": float(box_center_x),
                    "y": float(robot_y)
                }
                # 转换机器人位置到真实世界坐标
                if solver.is_pose_solved:
                    world_pos = solver.coordinate_image_to_world((robot_position["x"], robot_position["y"]))
                    if world_pos is not None:
                        robot_position = {
                            "x": float(world_pos[0]),
                            "y": float(world_pos[1])
                        }
                break  # 只取第一个检测到的机器人

    if result.keypoints is None or len(result.keypoints.xy) == 0:
        pose_img = result.plot(img=pose_img)

        # ===================== 界面输出状态 =====================
        cv2.putText(pose_img, "Status: No Person", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Top-Down Pose + Fall/Hand Detection", pose_img)
        cv2.imshow("Detection", detect_img)
        send_status_to_mcp(is_fall, hand_raised, person_foot_center, robot_position)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    kps_xy = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()
    if len(kps_xy) < 17:  # 确保有足够的关键点(包括脚踝)
        send_status_to_mcp(is_fall, hand_raised, person_foot_center, robot_position)
        continue


    def valid(idx):
        return kps_conf[idx] > CONF_THRESH


    l_sho = kps_xy[5] if valid(5) else None
    r_sho = kps_xy[6] if valid(6) else None
    l_wri = kps_xy[9] if valid(9) else None
    r_wri = kps_xy[10] if valid(10) else None
    l_hip = kps_xy[11] if valid(11) else None
    r_hip = kps_xy[12] if valid(12) else None
    l_ankle = kps_xy[15] if valid(15) else None  # 左脚踝
    r_ankle = kps_xy[16] if valid(16) else None  # 右脚踝

    # ===================== 计算双脚中心点 =====================
    person_foot_pixel = None
    if l_ankle is not None and r_ankle is not None:
        foot_center_x = (l_ankle[0] + r_ankle[0]) / 2
        foot_center_y = (l_ankle[1] + r_ankle[1]) / 2
        person_foot_pixel = {
            "x": float(foot_center_x),
            "y": float(foot_center_y)
        }
        person_foot_center = {
            "x": float(foot_center_x),
            "y": float(foot_center_y)
        }
    elif l_ankle is not None:
        person_foot_pixel = {
            "x": float(l_ankle[0]),
            "y": float(l_ankle[1])
        }
        person_foot_center = {
            "x": float(l_ankle[0]),
            "y": float(l_ankle[1])
        }
    elif r_ankle is not None:
        person_foot_pixel = {
            "x": float(r_ankle[0]),
            "y": float(r_ankle[1])
        }
        person_foot_center = {
            "x": float(r_ankle[0]),
            "y": float(r_ankle[1])
        }
    
    # 转换双脚中心点到真实世界坐标
    if person_foot_center and solver.is_pose_solved:
        world_pos = solver.coordinate_image_to_world((person_foot_center["x"], person_foot_center["y"]))
        if world_pos is not None:
            person_foot_center = {
                "x": float(world_pos[0]),
                "y": float(world_pos[1])
            }

    if l_sho is not None and r_sho is not None and l_hip is not None and r_hip is not None:
        sho_cx = (l_sho[0] + r_sho[0]) / 2
        sho_cy = (l_sho[1] + r_sho[1]) / 2
        hip_cx = (l_hip[0] + r_hip[0]) / 2
        hip_cy = (l_hip[1] + r_hip[1]) / 2
        box = result.boxes.xywh[0].cpu().numpy()
        bbox_ratio = box[2] / (box[3] + 1e-5)
        trunk_dx = abs(sho_cx - hip_cx)
        trunk_dy = abs(sho_cy - hip_cy)
        trunk_angle = np.degrees(np.arctan2(trunk_dy, trunk_dx + 1e-5))

        valid_kps = kps_xy[kps_conf > CONF_THRESH]
        kp_spread_ratio = 1.0
        hip_height_ratio = 0.0

        if len(valid_kps) >= 3:
            kp_x_range = valid_kps[:, 0].max() - valid_kps[:, 0].min()
            kp_y_range = valid_kps[:, 1].max() - valid_kps[:, 1].min()
            kp_spread_ratio = kp_x_range / (kp_y_range + 1e-5)
            bbox_top = box[1] - box[3] / 2
            bbox_h = box[3]
            hip_height_ratio = (hip_cy - bbox_top) / (bbox_h + 1e-5)

        fall_A = (bbox_ratio > 1.3) and (trunk_angle < 45)
        is_fall_now = fall_A

        hand_raised_now = False
        if not is_fall_now and trunk_angle > 60:
            if l_wri is not None and l_wri[1] < l_sho[1] - 20:
                hand_raised_now = True
            if r_wri is not None and r_wri[1] < r_sho[1] - 20:
                hand_raised_now = True

        fall_history.append(is_fall_now)
        hand_history.append(hand_raised_now)

        is_fall = sum(fall_history) >= FALL_FRAMES
        hand_raised = sum(hand_history) >= HAND_FRAMES

    send_status_to_mcp(is_fall, hand_raised, person_foot_center, robot_position)
    pose_img = result.plot(img=pose_img)

    # ===================== 【核心：把检测结果显示在界面上】 =====================
    # 显示跌倒状态
    fall_text = f"FALL: {'DETECTED' if is_fall else 'Normal'}"
    fall_color = (0, 0, 255) if is_fall else (0, 255, 0)

    # 显示抬手状态
    hand_text = f"HAND RAISED: {'YES' if hand_raised else 'NO'}"
    hand_color = (0, 255, 0) if hand_raised else (255, 255, 0)

    # 显示角度与比例
    info_text = f"Angle: {trunk_angle:.1f}  Ratio: {bbox_ratio:.2f}"

    # 时间
    time_text = time.strftime("%Y-%m-%d %H:%M:%S")

    # 绘制到画面
    cv2.putText(pose_img, fall_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, fall_color, 2)
    cv2.putText(pose_img, hand_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, hand_color, 2)
    cv2.putText(pose_img, info_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(pose_img, time_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ===================== 【新增：在画面上标记双脚中心点和机器人位置】 =====================
    # 标记人体双脚中心点
    if person_foot_pixel is not None:
        # 绘制像素坐标点
        foot_x, foot_y = int(person_foot_pixel["x"]), int(person_foot_pixel["y"])
        cv2.circle(pose_img, (foot_x, foot_y), 8, (0, 255, 0), -1)  # 绿色圆点
        # 显示像素坐标
        pixel_text = f"Foot: ({int(person_foot_pixel['x'])}, {int(person_foot_pixel['y'])})"
        cv2.putText(pose_img, pixel_text, (foot_x + 10, foot_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 标记机器人位置
    if robot_pixel is not None:
        # 绘制像素坐标点
        robot_x, robot_y = int(robot_pixel["x"]), int(robot_pixel["y"])
        cv2.circle(detect_img, (robot_x, robot_y), 10, (255, 0, 0), -1)  # 蓝色圆点
        # 显示像素坐标
        pixel_text = f"Robot: ({int(robot_pixel['x'])}, {int(robot_pixel['y'])})"
        cv2.putText(detect_img, pixel_text, (robot_x + 12, robot_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 跌倒红色大框
    if is_fall:
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
        cv2.rectangle(pose_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(pose_img, "WARNING: FALL ALERT!", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 显示窗口
    cv2.imshow("Top-Down Pose + Fall/Hand Detection", pose_img)
    cv2.imshow("Detection", detect_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()