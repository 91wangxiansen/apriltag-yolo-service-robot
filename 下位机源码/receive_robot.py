#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import json
import rospy
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

# UDP 服务器配置
SERVER_IP = ''  # 监听所有接口
SERVER_PORT = 8888

# ROS 话题配置
GOAL_TOPIC = '/move_base/goal'

# 状态变量
publish_enabled = False  # 发布开关
last_hand_up_flag = 0  # 上一次抬手状态

# 创建 UDP 套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, SERVER_PORT))

def publish_goal(goal_x, goal_y, target_yaw):
    """发布导航目标到 move_base"""
    rospy.init_node('robot_navigation_goal_publisher', anonymous=True)
    pub = rospy.Publisher(GOAL_TOPIC, MoveBaseActionGoal, queue_size=10)
    
    # 创建目标消息
    goal_msg = MoveBaseActionGoal()
    goal_msg.header.stamp = rospy.Time.now()
    goal_msg.header.frame_id = 'map'  # 使用地图坐标系
    
    # 设置位置
    goal_msg.goal.target_pose.pose.position.x = goal_x
    goal_msg.goal.target_pose.pose.position.y = goal_y
    goal_msg.goal.target_pose.pose.position.z = 0.0
    
    # 设置朝向（从 yaw 角度转换为四元数）
    quat = quaternion_from_euler(0, 0, target_yaw)
    goal_msg.goal.target_pose.pose.orientation.x = quat[0]
    goal_msg.goal.target_pose.pose.orientation.y = quat[1]
    goal_msg.goal.target_pose.pose.orientation.z = quat[2]
    goal_msg.goal.target_pose.pose.orientation.w = quat[3]
    
    # 发布消息
    pub.publish(goal_msg)
    rospy.loginfo("发布导航目标: x={}, y={}, yaw={}".format(goal_x, goal_y, target_yaw))

def main():
    global publish_enabled, last_hand_up_flag
    
    print("=== 机器人导航目标接收程序 ===")
    print("监听 UDP 端口: {}".format(SERVER_PORT))
    print("ROS 话题: {}".format(GOAL_TOPIC))
    print("按 Ctrl+C 退出程序")
    
    try:
        while True:
            # 接收 UDP 数据
            data, addr = sock.recvfrom(1024)  # 缓冲区大小 1024 字节
            
            try:
                # 解析 JSON 数据
                json_data = json.loads(data)
                
                # 提取数据
                goal_x = json_data.get('goal_x', 0.0)
                goal_y = json_data.get('goal_y', 0.0)
                target_yaw = json_data.get('target_yaw', 0.0)
                fall_flag = json_data.get('fall_flag', 0)
                hand_up_flag = json_data.get('hand_up_flag', 0)
                
                # 处理抬手开关
                if hand_up_flag == 1 and last_hand_up_flag == 0:
                    publish_enabled = not publish_enabled
                    print("发布开关: {}".format("开启" if publish_enabled else "关闭"))
                last_hand_up_flag = hand_up_flag
                
                # 处理跌倒发布
                if publish_enabled and fall_flag == 1:
                    print("检测到跌倒，发布导航目标")
                    publish_goal(goal_x, goal_y, target_yaw)
                elif publish_enabled and fall_flag == 0:
                    print("未检测到跌倒，不发布导航目标")
                
            except Exception as e:
                print("解析数据失败: {}".format(e))
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 关闭套接字
        sock.close()
        print("程序结束")

if __name__ == "__main__":
    main()