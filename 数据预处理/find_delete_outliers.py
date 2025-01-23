import math
import os
from tkinter import messagebox

import numpy as np


def read_txt_skip_first_line(file_path):
    # 初始化一个列表来存储跳过第一行后的内容
    lines = []
    # 打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取第一行并跳过
        file.readline()
        # 读取剩下的行并添加到列表中
        for line in file:
            # print("line:",line.replace("[","").replace("]",""))
            lines.append(line.replace("[","").replace("]","").strip())  # 使用strip()去除每行末尾的换行符
    return lines

def haversine(lat1, lon1, lat2, lon2):
    # 将度数转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算差值
    delta_lat = np.abs(lat2_rad - lat1_rad)
    delta_lon = np.abs(lon2_rad - lon1_rad)

    # 计算a
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2

    # 计算c
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球的平均半径（公里）
    R = 6371

    # 计算距离
    distance = R * c
    return distance
def isExistRegion(Inflect_Point,find_outliers,R,Region_Center):
    # print("current_vehicel=", current_vehicel)30.545054,114.336986
    Region_Center_lng = Region_Center[1]
    Region_Center_lat = Region_Center[0]
    Filter_Inflection_Point = []
    for i in range(len(Inflect_Point)):
        parts = Inflect_Point[i].split(",")
        current_point_lng = float(parts[0])
        current_point_lat = float(parts[1])
        # print(Inflect_Point[i])
        # print("current_point_lng=",current_point_lng, "current_point_lat=",current_point_lat)
        # 经（纬）度相差0.00001大约是实际距离的1.1119492664455877 米
        is_in_region = (current_point_lng <= Region_Center_lng + R
                        and current_point_lng >= Region_Center_lng - R
                        and current_point_lat >= Region_Center_lat - R
                        and current_point_lat <= Region_Center_lat + R)
        if is_in_region:
            value = str(current_point_lng)+","+str(current_point_lat)
            Filter_Inflection_Point.append(value)
    # print(Filter_Inflection_Point[0:2],len(Filter_Inflection_Point))

    # for point in Filter_Inflection_Point:
    #     print(point)

    write_file(Filter_Inflection_Point,find_outliers)

    return Filter_Inflection_Point

def write_file(my_list,find_outliers):
    with open(find_outliers, "w") as file:
        file.write("lng,lat" + "\n")
    with open(find_outliers, "a") as file:
        for data in my_list:
            file.write(data + "\n")
def isPathExist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def show_prompt():
    result = messagebox.askquestion('提示', '是否要执行程序？')
    if result == 'yes':
        messagebox.showinfo('提示', '操作已确认。')
        return True
    else:
        messagebox.showinfo('提示', '操作已取消。')
        return False
if __name__ == "__main__":
    load_file = "Filter_Inflection_All_Centroid_Points/Filter_Inflection_All_Centroid_Points.txt"
    find_outliers = "outliers_points.txt"
    Region_Center = [30.5615670,114.3405804]
    # 经（纬）度相差0.00001大约是实际距离的1.1119492664455877 米
    R = 0.0002
    # isPathExist(result_file_path)
    Inflect_Point = read_txt_skip_first_line(load_file)
    # 30.536156,114.330432
    delete_point = isExistRegion(Inflect_Point,find_outliers,R,Region_Center)
    # print(delete_point)

    #删除异常点
    new_load_file = []
    for point in Inflect_Point:
        # print(point)
        if point not in delete_point:
            new_load_file.append(point)
    print(len(new_load_file),len(delete_point),len(Inflect_Point))
    load_file1 = "Filter_Inflection_All_Centroid_Points/Filter_Inflection_All_Centroid_Points.txt"
    flag = show_prompt()
    if flag:
        print("操作已执行")
        write_file(new_load_file,load_file1)
    else:
        print("操作已取消")



