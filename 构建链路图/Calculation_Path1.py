import json
import math

import numpy as np
import pandas as pd
class Graph:
    def __init__(self):
        # 使用字典来存储扩展邻接表
        self.adj_list = {}
    def add_vertex(self, vertex):
        # 如果顶点不在图中，则添加它
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []
    def add_edge(self, vertex1, vertex2, weight):
        # 添加一条带权重的无向边
        if vertex1 not in self.adj_list:
            self.add_vertex(vertex1)
        if vertex2 not in self.adj_list:
            self.add_vertex(vertex2)
        if (vertex2,weight) not in self.adj_list[vertex1]:
            self.adj_list[vertex1].append([vertex2, weight])
    def add_weight(self, vertex1, vertex2,temp_index):
        if vertex1 not in self.adj_list:
            return False
        if vertex2 not in self.adj_list:
            return False

        self.adj_list[vertex1][temp_index][1] += 1
        return True
    def __str__(self):
        # 打印图的扩展邻接表表示
        result = ""
        for vertex in self.adj_list:
            if len(self.adj_list[vertex])!=0:
                edges = self.adj_list[vertex]
                edges_str = ", ".join(f"({v}, {w})" for v, w in edges)
                result += f"{vertex} -> [{edges_str}]\n"
        return result

    def to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.adj_list, file)

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as file:
            adj_list = json.load(file)
        graph = Graph()
        graph.adj_list = adj_list
        return graph
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
    print(len(adj_matrix[0]),len(adj_matrix[1]))
    for i in adj_matrix:
        print(i)
    return adj_matrix

def save_title(csv_file_path):
    with open(csv_file_path, 'w') as csvfile:
        csvfile.write("from,to,cost")
        csvfile.write("\n")
def save_data(csv_file_path,orgin,target,cost):
    with open(csv_file_path, 'a') as csvfile:
        csvfile.write(str(orgin)+","+str(target)+","+str(cost)+"\n")
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

    # 定义CSV文件路径
    csv_file_path = 'graph_file/beifen2/result_distance_matrix.csv'
    # df.to_csv(csv_file_path)
    save_title(csv_file_path)
    # result_adj_matrix = adjacency_list_to_matrix(Centroid_Points_Correspond_Index,len(Centroid_Points_Correspond_Index))

    # 通过整个图来构建文件
    All_graph_name = "graph_file/all/2019-10-17-19_day_17-Construct_Filter_result"
    loaded_graph = Graph.from_json(All_graph_name + '.json')
    all_result = loaded_graph.adj_list
    # 遍历原始图
    for vertex in all_result:
        print(vertex, "->", all_result[vertex])
        for edge in all_result[vertex]:
            point1 = Centroid_Points_Correspond_Index[int(vertex)]
            if all_result[vertex] != []:
                point2 = Centroid_Points_Correspond_Index[int(edge[0])]
                distance = haversine(point1[0], point1[1], point2[0], point2[1])*1000
                save_data(csv_file_path,vertex,edge[0],distance)



    # # 将矩阵转换为DataFrame
    # df = pd.DataFrame(result_adj_matrix, index=[str(i) for i in range(len(result_adj_matrix))],
    #                   columns=[str(i) for i in range(len(result_adj_matrix))])


    # print("lng=",Centroid_Points_Correspond_Index[0][0],"lat=",Centroid_Points_Correspond_Index[0][1])
    # distance = haversine(Centroid_Points_Correspond_Index[0][0], Centroid_Points_Correspond_Index[0][1],Centroid_Points_Correspond_Index[1][0], Centroid_Points_Correspond_Index[1][1])
    # print("distance=",distance)
    # for i in range(len(Centroid_Points_Correspond_Index)):
    #     for j in range(len(Centroid_Points_Correspond_Index)):
    #         print("i=",Centroid_Points_Correspond_Index[i],"------j=",Centroid_Points_Correspond_Index[j])
