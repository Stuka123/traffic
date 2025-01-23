import math

import numpy as np


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
if __name__ == "__main__":
    print("test")
    distance = 1000 * haversine(118.7664329083476,32.05996481083991, 118.76796317355158,32.05974419351058)
    time = 10
    print(distance/time)

    # 425 -> {163: 15.15581271331948}
    # 163 -> {31: 1.3157970775355796, 34: 0.7447995695736719, 125: 1.3637065551104717, 425: 7.761664410120936}