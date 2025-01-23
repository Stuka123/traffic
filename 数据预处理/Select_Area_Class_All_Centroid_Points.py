import math
import os

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
def isExistRegion(Inflect_Point,result_file_path,result_name):
    # print("current_vehicel=", current_vehicel)30.545054,114.336986
    Region_Center_lng = 114.336986
    Region_Center_lat = 30.545054
    Filter_Inflection_Point = []
    for i in range(len(Inflect_Point)):
        parts = Inflect_Point[i].split(",")
        current_point_lng = float(parts[0])
        current_point_lat = float(parts[1])
        print(Inflect_Point[i])
        print("current_point_lng=",current_point_lng, "current_point_lat=",current_point_lat)
        # 经（纬）度相差0.00001大约是实际距离的1.1119492664455877 米
        is_in_region = (current_point_lng <= Region_Center_lng + 0.02
                        and current_point_lng >= Region_Center_lng - 0.02
                        and current_point_lat >= Region_Center_lat - 0.02
                        and current_point_lat <= Region_Center_lat + 0.02)
        if is_in_region:
            value = str(current_point_lng)+","+str(current_point_lat)
            Filter_Inflection_Point.append(value)
    print(Filter_Inflection_Point[0:2],len(Filter_Inflection_Point))
    write_file(Filter_Inflection_Point,result_file_path,result_name)

def write_file(my_list, result_file_path,result_name):
    with open(result_file_path+result_name, "w") as file:
        file.write("lng,lat" + "\n")
    with open(result_file_path+result_name, "a") as file:
        for data in my_list:
            file.write(data + "\n")
def isPathExist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
if __name__ == "__main__":
    load_file = "../../my_result/traffic_wuhan_data/result/all_day/sort_time/point_result/merge_point/Clusters/Class_All_Centroid_Points/Class_All_Centroid_Points.txt"

    # csv_file_path = "Class_All_Centroid_Points/2019-10-17-19-Class_All_Centroid_Points.txt"
    result_file_path = "Filter_Inflection_All_Centroid_Points/"
    result_name = "Filter_Inflection_All_Centroid_Points.txt"
        # "Filter_Inflection_Point/2019-10-17-19-Class_All_Centroid_Points.txt"
    isPathExist(result_file_path)
    Inflect_Point = read_txt_skip_first_line(load_file)
    isExistRegion(Inflect_Point,result_file_path,result_name)

    #用来计算距离的
    # Region_Center_lng = 118.768311
    # Region_Center_lat = 32.030166

    # Region_Margin_lng = 118.73339
    # Region_Margin_lat = 32.03246
    # distance = haversine(Region_Center_lng,Region_Center_lat,Region_Margin_lng,Region_Margin_lat)
    # print("distance=",distance*1000,"米")