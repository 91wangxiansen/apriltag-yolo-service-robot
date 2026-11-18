import socket
import time
import json
import math


# 小车IP地址和端口
CAR_IP = '192.168.12.1'
CAR_PORT = 8888

# 发送频率 (秒)
SEND_INTERVAL = 0.1

# JSON 文件路径
JSON_FILE = 'robot_status.json'

# 死区半径 (米)
DEAD_ZONE_RADIUS = 0.5

# 创建UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def calculate_distance(point1, point2):
    """计算两点之间的距离"""
    if point1 is None or point2 is None:
        return None
    dx = point1['x'] - point2['x']
    dy = point1['y'] - point2['y']
    return math.sqrt(dx*dx + dy*dy)


def calculate_direction_vector(robot_pos, person_pos):
    """计算机器人指向人体的方向向量"""
    if robot_pos is None or person_pos is None:
        return None
    dx = person_pos['x'] - robot_pos['x']
    dy = person_pos['y'] - robot_pos['y']
    distance = calculate_distance(robot_pos, person_pos)
    if distance == 0:
        return None
    return {'x': dx/distance, 'y': dy/distance}


def calculate_navigation_target(robot_pos, person_pos):
    """计算死区外最近合法导航目标点"""
    if robot_pos is None or person_pos is None:
        return None
    
    distance = calculate_distance(robot_pos, person_pos)
    if distance is None:
        return None
    
    # 如果机器人在死区内，后退到死区边界
    if distance < DEAD_ZONE_RADIUS:
        direction = calculate_direction_vector(person_pos, robot_pos)  # 反向向量
        if direction is None:
            return None
        target_x = person_pos['x'] + direction['x'] * DEAD_ZONE_RADIUS
        target_y = person_pos['y'] + direction['y'] * DEAD_ZONE_RADIUS
        return {'x': target_x, 'y': target_y}
    else:
        # 否则，目标点就是当前机器人位置（或保持在死区外）
        return robot_pos


def calculate_target_orientation(robot_pos, person_pos):
    """计算目标点朝向，使机器人正对人体中心"""
    if robot_pos is None or person_pos is None:
        return None
    
    dx = person_pos['x'] - robot_pos['x']
    dy = person_pos['y'] - robot_pos['y']
    
    # 计算朝向角度（弧度）
    angle = math.atan2(dy, dx)
    return angle


def process_data(data):
    """处理数据并计算导航信息"""
    robot_pos = data.get('robot_position')
    person_pos = data.get('person_foot_center')
    fall_detected = data.get('fall_detected', False)
    hand_raised = data.get('hand_raised', False)
    
    # 计算距离
    distance = calculate_distance(robot_pos, person_pos)
    
    # 计算方向向量
    direction = calculate_direction_vector(robot_pos, person_pos)
    
    # 计算导航目标点
    navigation_target = calculate_navigation_target(robot_pos, person_pos)
    
    # 计算目标朝向
    target_orientation = calculate_target_orientation(robot_pos, person_pos)
    
    # 构建发送数据
    goal_x = navigation_target['x'] if navigation_target else 0.0
    goal_y = navigation_target['y'] if navigation_target else 0.0
    target_yaw = target_orientation if target_orientation else 0.0
    robot_x = robot_pos['x'] if robot_pos else 0.0
    robot_y = robot_pos['y'] if robot_pos else 0.0
    fall_flag = 1 if fall_detected else 0
    hand_up_flag = 1 if hand_raised else 0
    
    # 构建报文数据
    send_data = {
        'goal_x': goal_x,
        'goal_y': goal_y,
        'target_yaw': target_yaw,
        'robot_x': robot_x,
        'robot_y': robot_y,
        'fall_flag': fall_flag,
        'hand_up_flag': hand_up_flag,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'source': 'navigation_system'
    }
    
    return send_data


def send_navigation_data():
    """发送导航数据到小车"""
    try:
        # 读取JSON文件
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理数据
        send_data = process_data(data)
        
        # 将数据转换为字符串
        data_str = json.dumps(send_data, ensure_ascii=False)
        
        # 发送数据
        sock.sendto(data_str.encode('utf-8'), (CAR_IP, CAR_PORT))
        print(f"发送导航数据: {data_str[:150]}...")
        return True
    except Exception as e:
        print(f"发送失败: {e}")
        return False


def main():
    print("=== 机器人导航数据发送程序 ===")
    print(f"目标IP: {CAR_IP}:{CAR_PORT}")
    print(f"JSON文件: {JSON_FILE}")
    print(f"发送间隔: {SEND_INTERVAL}秒")
    print(f"死区半径: {DEAD_ZONE_RADIUS}米")
    print("按 Ctrl+C 退出程序")
    
    try:
        while True:
            send_navigation_data()
            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 关闭套接字
        sock.close()
        print("程序结束")


if __name__ == "__main__":
    main()