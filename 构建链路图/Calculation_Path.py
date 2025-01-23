import math

import numpy as np
import pandas as pd

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
def adjacency_list_to_matrix(Centroid_Points_Correspond_Index,num_vertices):
    # sigma = np.sqrt(10)
    # epsilon = 0.5
    #初始化邻接矩阵
    adj_matrix = [[float(0)] * num_vertices for _ in range(num_vertices)]
    for i in adj_matrix:
        print(i)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i != j:
                distance = haversine(Centroid_Points_Correspond_Index[i][0], Centroid_Points_Correspond_Index[i][1],
                                     Centroid_Points_Correspond_Index[j][0], Centroid_Points_Correspond_Index[j][1])
                adj_matrix[i][j] = float(distance*1000)
                # distance = haversine(Centroid_Points_Correspond_Index[i][0],Centroid_Points_Correspond_Index[i][1],Centroid_Points_Correspond_Index[j][0],Centroid_Points_Correspond_Index[j][1])
                # print("distance = ", distance,"----?",np.exp(-distance/sigma*sigma))
                # if np.exp(-distance/sigma) >= epsilon:
                # adj_matrix[i][j] = float(np.exp(-distance/sigma*sigma))
    print(len(adj_matrix[0]),len(adj_matrix[1]))
    for i in adj_matrix:
        print(i)
    return adj_matrix
if __name__ == "__main__":
    print("calculation path")
    Path_Centroid = f"../DT_Clustering-main/Class_All_Centroid_Points/2019-10-17-19-Filter_Inflection_Point.txt"
    Centroid_Points = pd.read_csv(Path_Centroid, delimiter=',')
    Centroid_Points_Correspond_Index = []
    # 初始化图，并将图中每一个点对应的坐标保存到列表中
    for k in range(len(Centroid_Points)):
        current_point = Centroid_Points.iloc[k:k + 1].reset_index(drop=True).to_numpy()
        temp = current_point[0][0], current_point[0][1]
        Centroid_Points_Correspond_Index.append(temp)
    print(Centroid_Points_Correspond_Index[0], len(Centroid_Points_Correspond_Index))
    print(len(Centroid_Points_Correspond_Index))

    result_adj_matrix = adjacency_list_to_matrix(Centroid_Points_Correspond_Index,len(Centroid_Points_Correspond_Index))
    # 将矩阵转换为DataFrame
    df = pd.DataFrame(result_adj_matrix, index=[str(i) for i in range(len(result_adj_matrix))],
                      columns=[str(i) for i in range(len(result_adj_matrix))])

    # 定义CSV文件路径
    csv_file_path = 'result_distance_matrix.csv'
    df.to_csv(csv_file_path)
    # print("lng=",Centroid_Points_Correspond_Index[0][0],"lat=",Centroid_Points_Correspond_Index[0][1])
    # distance = haversine(Centroid_Points_Correspond_Index[0][0], Centroid_Points_Correspond_Index[0][1],Centroid_Points_Correspond_Index[1][0], Centroid_Points_Correspond_Index[1][1])
    # print("distance=",distance)
    # for i in range(len(Centroid_Points_Correspond_Index)):
    #     for j in range(len(Centroid_Points_Correspond_Index)):
    #         print("i=",Centroid_Points_Correspond_Index[i],"------j=",Centroid_Points_Correspond_Index[j])
