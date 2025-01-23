import copy
from datetime import datetime
import math
import os
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

import WGS_TO_GCJ02
import sys
sys.setrecursionlimit(15000)

#定义带权邻接表
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
        # if ((vertex2,weight) not in self.adj_list[vertex1]) and ((vertex1,weight) not in self.adj_list[vertex2]):
        #     # 在vertex1的邻接表中添加(vertex2, weight)元组
        #     self.adj_list[vertex1].append((vertex2, weight))
        if (vertex2,weight) not in self.adj_list[vertex1]:
            # 在vertex1的邻接表中添加(vertex2, weight)元组
            self.adj_list[vertex1].append((vertex2, weight))
        if (vertex1,weight) not in self.adj_list[vertex2]:
            # 因为是无向图，所以在vertex2的邻接表中也要添加(vertex1, weight)元组
            self.adj_list[vertex2].append((vertex1, weight))
    def traverse_half_adj_list(self):
        # total_weight = 0
        # count = 0
        weight_set = []
        # 遍历每个顶点及其邻接表
        for vertex, neighbors in self.adj_list.items():
            # 只遍历邻接表的一半（通过比较顶点编号）
            for neighbor, weight in neighbors:
                if neighbor > vertex:  # 确保只遍历一半，避免重复计算
                    weight_set.append(weight)
                    # total_weight += weight
                    # count += 1
        return weight_set
    def traverse_all_adj_list(self):
        # total_weight = 0
        # count = 0
        point_to_edge_mean = {}
        # 遍历每个顶点及其邻接表
        for vertex, neighbors in self.adj_list.items():
            # print(vertex,neighbors)
            weight_all = []
            for neighbor, weight in neighbors:
                weight_all.append(weight)
            mean_p_i = np.mean(weight_all)
            point_to_edge_mean[vertex] = mean_p_i
        return point_to_edge_mean
    def all_adj_list_F_p_i(self,mean_D,Sta_Dev_D,point_to_edge_mean):
        F_p = {}
        # 遍历每个顶点及其邻接表
        for vertex, neighbors in self.adj_list.items():
            F_p_i = mean_D + mean_D /point_to_edge_mean[vertex]  * Sta_Dev_D
            F_p[vertex] = F_p_i
        return F_p

    def delete_edge(self,F_p):
        # 遍历邻接表的每一项
        for vertex1, edges in list(self.adj_list.items()):  # 使用list()来避免在遍历过程中修改字典导致的错误
            # 创建一个新列表来存储要保留的边，以避免在遍历过程中修改原列表
            new_edges = []
            for edge in edges:
                vertex2, weight = edge
                if weight < F_p[vertex1]:
                    new_edges.append(edge)  # 保留权重不小于阈值的边
                else:
                    if vertex2 in self.adj_list:
                        # 使用列表推导式来过滤掉到vertex1的边
                        self.adj_list[vertex2] = [(neigh, w) for (neigh, w) in self.adj_list[vertex2] if
                                                  neigh != vertex1]
            # 更新vertex1的邻接表
            self.adj_list[vertex1] = new_edges
        # for vertex1, neighbors in list(self.adj_list.items()):  # 使用list()来避免在遍历过程中修改字典导致的错误
        #     print("vertex1: {}, neighbors: {}".format(vertex1, neighbors))
        #     for vertex2, weight in list(neighbors):  # 同样使用list()来避免在内部循环中修改字典
        #         if weight >= F_p[vertex1]:
        #             # 删除vertex1到vertex2的边
        #             del self.adj_list[vertex1][vertex2]
        #             # 删除vertex2到vertex1的边（对于无向图这是必需的）
        #             del self.adj_list[vertex2][vertex1]
        return True
    def build_bidirectional_weighted_adj_list(self):
        bidirectional_adj_list = {}
        for u, neighbors in self.adj_list.items():
            new_edges = []
            # print("u=",u,"neighbors=",neighbors)
            bidirectional_adj_list[u] = {}
            for v, weight in neighbors:
                # print("v=",v,"weight=",weight,"neighbors=",neighbors)
                # print("self.adj_list[v]=",self.adj_list[v])
                point_1 = [tup[0] for tup in self.adj_list[v]]
                weight_2 = [tup[1] for tup in self.adj_list[v]]
                flag = -1
                for i in range(len(point_1)):
                    if u == point_1[i]:
                        flag = i
                        break
                # print("self.adj_list[v][u]=", point_1)
                # print("self.adj_list[v][u]1=", weight_2)
                # 检查反向边是否存在且权重是否一致
                if flag != -1 and weight_2[flag] == weight:
                    new_edges.append((v,weight))
                    bidirectional_adj_list[u][v] = weight
            self.adj_list[u] = new_edges

    def is_exsit_triangle(self,tri):
        # 遍历每个顶点及其邻接表
        for point in tri:
            edges = []
            for val in self.adj_list[point]:
                edges.append(int(val[0]))
            sub_point = [int(x) for x in tri if x != point]
            for sub in sub_point:
                if sub not in edges:
                    return False
        return True
    def find_connected_components(self):
        visited = set()
        components = []
        def dfs(v, component):
            visited.add(v)
            component.add(v)
            # print("v=",v,"visited=", visited,"component=",component)
            for neighbor,weight in self.adj_list[v]:
                # print("neighbor=",neighbor,"weight=",weight)
                if neighbor not in visited:
                    dfs(neighbor, component)
        for vertex in self.adj_list:
            # print("vertex=",vertex)
            if vertex not in visited:
                component = set()
                dfs(vertex, component)
                components.append(component)
            # print("---------------")
        return components
    def generate_sub_graph(self,sub_graph,set):
        for vertex in self.adj_list:
            if len(self.adj_list[vertex])!=0 and vertex in set:
                for neighbor,weight in self.adj_list[vertex]:
                    sub_graph.add_edge(vertex, neighbor, weight)
        return sub_graph
    def __str__(self):
        # 打印图的扩展邻接表表示
        result = ""
        for vertex in self.adj_list:
            edges = self.adj_list[vertex]
            edges_str = ", ".join(f"({v}, {w})" for v, w in edges)
            result += f"{vertex} -> [{edges_str}]\n"
        return result
def isPathExist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    #     print(f"目录 '{directory_path}' 已创建。")
    # else:
    #     print(f"目录 '{directory_path}' 已经存在。")
#找出所有车辆ID
def select_single_vehicel(attribute,file_name):
    df = pd.read_csv(Origin_Path + file_name + '.txt', delimiter=',')
    l = len(df)
    print("数据集长度：", l)
    # print(df)
    with open(Result_Path + file_name + '车辆ID.txt', 'w', encoding='utf-8') as file:
        file.write('')
    list_total = []
    for i in range(len(df)):
        element = df.loc[i][attribute]
        if element not in list_total:
            list_total.append(element)
            with open(Result_Path + file_name+'车辆ID.txt', 'a', encoding='utf-8') as file:
                file.write(str(element))
                file.write(',')
        # print(i)
        # if i == 30:
        #     break
    print("长度：",len(list_total))

def time_sorted(vehicel_file_name,result_vehicel_file_name,Origin_Path1,Result_Path1):
    Origin_Path = Origin_Path1
    Result_Path = Result_Path1+'/sort_time/'

    print("Origin_Path=",Origin_Path + vehicel_file_name+ ".txt")
    print("Result_Path=",Result_Path + vehicel_file_name+ "_result.txt")
    # 检查文件夹是否存在
    if not os.path.exists(Result_Path):
        # 如果不存在，则创建文件夹
        os.makedirs(Result_Path)

    if os.path.exists(Origin_Path + vehicel_file_name+ ".txt") == False:
        return False
    else:
        input_file = open(Origin_Path + vehicel_file_name + ".txt", "r", encoding="utf-8")
        output_file = open(Result_Path + result_vehicel_file_name + ".txt", "w", encoding="utf-8")
        print(Result_Path + result_vehicel_file_name + ".txt")
        table = []
        header = input_file.readline()  # 读取并弹出第一行
        # print("file_name=", vehicel_file_name)
        for line in input_file:
            col = line.split(',')  # 每行分隔为列表，好处理列格式
            col[6] = col[6][0:-2]
            col[7] = col[7][0:-2]
            result = WGS_TO_GCJ02.transform(float(col[2]), float(col[1]))
            if result != None:
                col[1] = result[0]
                col[2] = result[1]
                table.append(col)

            # result = WGS_TO_GCJ02.transform(float(col[4]), float(col[3]))
            # if result != None:
            #     col[3] = result[0]
            #     col[4] = result[1]
            #     table.append(col)  # 嵌套列表table[[8,8][*,*],...]

        # print(table)
        table_sorted = sorted(table, key=itemgetter(6,7))  # 先后按列索引3,4排序
        # table_sorted = sorted(table, key=itemgetter(2, 3))  # 先后按列索引3,4排序
        output_file.write(header)
        for row in table_sorted:  # 遍历读取排序后的嵌套列表
            row = [str(x) for x in row]  # 转换为字符串格式，好写入文本
            # print(row)
            output_file.write(",".join(row))

        input_file.close()
        output_file.close()
        return True

def Judgment_intersection(Origin_Path,Result_Path,file_name,header):
    # isPathExist(Result_Path + "center_point_" + file_name + '.txt')
    # isPathExist(Result_Path + "adjoin_point_" + file_name + '.txt')

    output_file = open(Result_Path + "center_point_" + file_name + ".txt", "w", encoding="utf-8")
    output_file1 = open(Result_Path + "adjoin_point_" + file_name + ".txt", "w", encoding="utf-8")
    output_file.write(header)
    output_file.write("\n")
    output_file1.write(header)
    output_file1.write("\n")
    output_file.close()
    output_file1.close()
    k = 1
    for j in range(1, 20000):
        data,flag = loading_data(Origin_Path, file_name + str(j) + "_result")
        if flag != False:
            print("第%s个文件" % j)
            for i in range(1, len(data) - 1):
                # 重置索引
                current = data.iloc[i - 1:i + 2].reset_index(drop=True)
                res1 = judg_speed(current)
                res2 = judg_angle(current)
                res3 = judg_distance(current)
                if res1 and res2 and res3:
                    k += 1
                    add_center_point = current.iloc[1:2]
                    add_adjoin_point1 = current.iloc[0:1]
                    add_adjoin_point2 = current.iloc[2:3]
                    add_center_point.to_csv(Result_Path + "center_point_" + file_name + '.txt', mode='a', header=False,
                                            index=False)
                    add_adjoin_point1.to_csv(Result_Path + "adjoin_point_" + file_name + '.txt', mode='a', header=False,
                                             index=False)
                    add_adjoin_point2.to_csv(Result_Path + "adjoin_point_" + file_name + '.txt', mode='a', header=False,
                                             index=False)
    print("点总数：",k)

#加载具体车辆轨迹数据
def loading_data(Origin_Path,file_name):
    if os.path.exists(Origin_Path + file_name + ".txt") == False:
        return None,False
    else:
        df = pd.read_csv(Origin_Path + file_name + '.txt', delimiter=',')
        return df,True

#距离m
distance_Threshold = 25
#速度m/s
speed_Threshold = 15
#角度°
angle_Threshold = 20
angle_Threshold_Max = 150
def judg_angle(df):
    direction1 = df['方向'][0]
    direction2 = df['方向'][1]
    direction3 = df['方向'][2]
    if direction1>180:
        direction1 = direction1 - 360
    if direction2>180:
        direction2 = direction2 - 360
    if direction3>180:
        direction3 = direction3 - 360
    error_value1 = np.abs(direction1 - direction2)
    error_value2 = np.abs(direction2 - direction3)
    error_value3 = np.abs(direction1 - direction3)
    if (error_value1 >=angle_Threshold and error_value1<=angle_Threshold_Max) and (error_value2 >=angle_Threshold and error_value2<=angle_Threshold_Max) and (error_value3 >=angle_Threshold and error_value3<=angle_Threshold_Max):
        return True
    return False
def judg_speed(df):
    #第一、二，二、三个点之间的时间
    time1 = df['数据发送时间'][0]
    time2 = df['数据发送时间'][1]
    time3 = df['数据发送时间'][2]
    time_format = "%Y-%m-%d %H:%M:%S"
    time1 = datetime.strptime(time1,time_format)
    time2 = datetime.strptime(time2,time_format)
    time3 = datetime.strptime(time3,time_format)
    diff_time1 = time2 - time1
    diff_time2 = time3 - time2

    #换算为秒为单位:s
    diff_time_s1 = diff_time1.total_seconds()
    diff_time_s2 = diff_time2.total_seconds()

    if diff_time_s1 == 0 or diff_time_s2 == 0:
        return False
    #第一、二，二、三个点之间的距离,单位：m
    distance1 = haversine(df['纬度'][0],df['经度'][0],df['纬度'][1],df['经度'][1])*1000
    distance2 = haversine(df['纬度'][1], df['经度'][1], df['纬度'][2], df['经度'][2])*1000

    #计算平均速度m/s
    average_speed1 = distance1/diff_time_s1
    average_speed2 = distance2/diff_time_s2
    if average_speed1 <= speed_Threshold and average_speed2 <= speed_Threshold:
        return True
    return False
def judg_distance(df):
    lng1 = df['经度'][0]
    lat1 = df['纬度'][0]
    lng2 = df['经度'][1]
    lat2 = df['纬度'][1]
    lng3 = df['经度'][2]
    lat3 = df['纬度'][2]

    distance1 = haversine(lng1, lat1, lng2, lat2) * 1000
    distance2 = haversine(lng2, lat2, lng3, lat3) * 1000

    if distance1 != 0 and distance2 != 0:
        if distance1 <= distance_Threshold and distance2 <= distance_Threshold:
            return True
    return False

#将两点经纬度差换算为距离
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

def MaxSide(vertex1, vertex2, vertex3):
    return (max(distance(vertex1, vertex2), distance(vertex1, vertex3), distance(vertex2, vertex3)))

def sortingKey(simplex):
    return (-1*MaxSide(points[simplex[0]], points[simplex[1]], points[simplex[2]]))
def compute_all_point(Origin_Path,Result_Path,file_name):
    data = pd.read_csv(Origin_Path + file_name + '.txt', delimiter=',')
    lenth = len(data)
    print(lenth)
    if lenth % 2 != 0:
        lenth = lenth - 1
    header = "经度,纬度"
    output_file = open(Result_Path + "final_merge_point.txt", "w", encoding="utf-8")
    output_file.write(header)
    output_file.write("\n")
    output_file.close()
    i = 0
    k = 0
    while i < lenth:
        print("进度：", i / len(data) * 100, "%")
        current = data.iloc[i:i + 2].reset_index(drop=True)
        A1, B1, C1, A2, B2, C2 = find_equition(current.loc[0]['方向'], current.loc[1]['方向'],
                                               current.loc[0]['纬度'], current.loc[0]['经度'],
                                               current.loc[1]['纬度'], current.loc[1]['经度'])
        intersection = find_intersection(A1, B1, C1, A2, B2, C2)
        if intersection and intersection[0]>20 and intersection[0]<40 and intersection[1]>110 and intersection[1]<130:
            k+=1
            res = str(+ intersection[1]) + "," + str(intersection[0])
            # print(res)
            with open(Result_Path + 'final_merge_point.txt', 'a', encoding='utf-8') as file:
                file.write(res)
                file.write('\n')
            output_file.close()
        i += 2
    print("数据长度:",k)
#计算交叉坐标点
def find_intersection(A1,B1,C1,A2,B2,C2):
    # 构建系数矩阵和常数向量
    coefficients = np.array([[A1,B1], [A2,B2]])
    constants = np.array([-C1, -C2])
    # 使用 numpy.linalg.solve 解方程组
    try:
        solution = np.linalg.solve(coefficients, constants)
        x, y = solution
        return x, y
    except np.linalg.LinAlgError:
        # 如果系数矩阵是奇异的（即行列式为0），表示两条直线平行或重合
        return None

#计算两条直线的参数
def find_equition(theta1,theta2,lng1,lat1,lng2,lat2):
    #计算两个方程的斜率k
    A1 = np.sin(theta1/180*np.pi)
    B1 = -np.cos(theta1/180*np.pi)
    C1 = -lat1 * B1 - lng1 * A1
    A2 = np.sin(theta2/180*np.pi)
    B2 = -np.cos(theta2/180*np.pi)
    C2 = -lat2 * B2 - lng2 * A2

    return A1, B1, C1, A2, B2, C2

def distance(pointA, pointB):
    return math.sqrt(math.pow(pointA[0]-pointB[0], 2)+math.pow(pointA[1]-pointB[1], 2))
#计算德劳内三角形的所有边及其对应的长度
def calculate_edge_all_and_distance(points,tri):
    graph = Graph()
    for triangle in tri:
        # 计算每一条边的长度
        distance1 = distance(points[triangle[0]], points[triangle[1]])
        distance2 = distance(points[triangle[1]], points[triangle[2]])
        distance3 = distance(points[triangle[0]], points[triangle[2]])
        # 记录所有边及其对应的长度
        graph.add_edge(triangle[0], triangle[1], distance1)
        graph.add_edge(triangle[1], triangle[2], distance2)
        graph.add_edge(triangle[0], triangle[2], distance3)
    return graph
#计算整个子图的均值和标准差
def calculate_mean_and_standvar(graph):
    weight_set = graph.traverse_half_adj_list()
    # print("weight_set=",weight_set)
    mean_D = np.mean(weight_set)
    Sta_Dev_D = np.sqrt(np.var(weight_set))
    return mean_D, Sta_Dev_D
#计算子图中每一个点所连接的所有边的均值
def calculate_all_point_counter_edge_mean(graph):
    point_to_edge_mean = graph.traverse_all_adj_list()
    return point_to_edge_mean

#计算子图中每一个点所对应的判别函数F_p的值
def Function_P(mean_D,Sta_Dev_D,point_to_edge_mean,graph):
    F_p = graph.all_adj_list_F_p_i(mean_D,Sta_Dev_D,point_to_edge_mean)
    return F_p

#计算外接圆
def calculate_bounding_circle(points):
    # 初始化最大距离和对应的两个点
    max_distance = 0
    point1, point2 = None, None
    print("计算外接圆")
    # 遍历所有点对，找到距离最大的那一对
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            if distance > max_distance:
                max_distance = distance
                point1, point2 = points[i], points[j]
    print("计算完成")
    # 计算圆心（两个最远点的中点）
    center_x = (point1[0] + point2[0]) / 2
    center_y = (point1[1] + point2[1]) / 2

    # 计算半径
    radius = max_distance / 2

    return center_x, center_y, radius

def are_graphs_equal(graph1, graph2):
    for key1 in graph1.adj_list:
        if key1 not in graph2.adj_list:
            return False
        if graph1.adj_list[key1] != graph2.adj_list[key1]:
            return False
    return True
def write_file(my_list,out_name,file_name):
    with open(out_name+file_name, "a") as file:
        # 你可以使用 for 循环和 write() 方法逐行写入
        # print("mylisy=",my_list)
        file.write(str(my_list) + "\n")  # 在每个元素后添加换行符

def DT_Clustering(points,tri,R,min_pts,out_name,file_name):
    #计算德劳内三角形的边及其对应的长度
    graph = calculate_edge_all_and_distance(points,tri)
    temp_graph = copy.deepcopy(graph)
    print(are_graphs_equal(graph,temp_graph))
    # print("===========原始的邻接表============")
    # 计算整个子图的均值和标准差
    mean_D, Sta_Dev_D = calculate_mean_and_standvar(graph)
    #计算子图中每一个点所连接边的长度及其所有边的均值
    point_to_edge_mean = calculate_all_point_counter_edge_mean(graph)
    #计算判别函数F_p对应点的值
    F_p = Function_P(mean_D,Sta_Dev_D,point_to_edge_mean,graph)
    #删除边
    graph.delete_edge(F_p)
    # print("===========删除边过后的邻接表============")
    graph.build_bidirectional_weighted_adj_list()
    # print("===========删除边过后的邻接表2============")
    print(are_graphs_equal(graph,temp_graph))
    #通过邻接表获取子图
    is_list = isinstance(points, list)
    if is_list == False:
        points1 = points.tolist()
    else:
        points1 = points
    result = []
    components = graph.find_connected_components()
    for i, component in enumerate(components):
        print(f"子图： {i + 1}: {component}")
        set = list(component)
        set1 = []
        for j in set:
            set1.append(points1[j])
        # print("set1=",set1)
        sub_graph = Graph()
        graph.generate_sub_graph(sub_graph,set)
        if len(set1) >= min_pts:
            if len(set1) > 2:
                # 计算外接圆
                center_x, center_y, radius = calculate_bounding_circle(set1)
                # print(f"外接圆的圆心为: ({center_x}, {center_y})")
                # print(f"外接圆的半径为: {radius}")
                if radius > R:
                    # print("-------------------------------------------------------------------------------")
                    # print("邻接表是否改变=",are_graphs_equal(graph,temp_graph))
                    if are_graphs_equal(graph, temp_graph):
                        result.append(set1)
                        write_file(set1,out_name,file_name)
                    else:
                        result.append(DT_Clustering1(points1, set1, sub_graph, R,min_pts,out_name,file_name))
                else:
                    result.append(set1)
                    write_file(set1,out_name,file_name)
            else:
                result.append(set1)
                write_file(set1,out_name,file_name)
        # print("当前的result=",result)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("result=",result)
    with open(out_name + file_name, "a") as file:
        file.write("\n")  # 在每个元素后添加换行符
    return result
def DT_Clustering1(points,sub_points,graph,R,min_pts,out_name,file_name):
    # print("points=",len(points),"sub_points=",len(sub_points))
    if len(sub_points) == 0:
        return sub_points
    temp_graph = copy.deepcopy(graph)
    # print("===========原始的邻接表============")
    # 计算整个子图的均值和标准差
    mean_D, Sta_Dev_D = calculate_mean_and_standvar(graph)
    # print("mean_D=", mean_D,"Sta_Dev_D=", Sta_Dev_D)
    #计算子图中每一个点所连接边的长度及其所有边的均值
    point_to_edge_mean = calculate_all_point_counter_edge_mean(graph)
    # print("point_to_edge_mean=",point_to_edge_mean)

    #计算判别函数F_p对应点的值
    F_p = Function_P(mean_D,Sta_Dev_D,point_to_edge_mean,graph)
    # print("F_p=",F_p)
    #删除边
    graph.delete_edge(F_p)
    # print("===========删除边过后的邻接表============")
    graph.build_bidirectional_weighted_adj_list()
    # print("===========删除边过后的邻接表2============")
    print(are_graphs_equal(graph, temp_graph))
    #通过邻接表获取子图
    is_list = isinstance(sub_points, list)
    # print("is_list=",is_list)
    if is_list == False:
        sub_points1 = sub_points.tolist()
    else:
        sub_points1 = sub_points
    components = graph.find_connected_components()
    result = []
    for i, component in enumerate(components):
        #当前子图所有点的索引
        set = list(component)
        #所有点的坐标
        set1 = []
        for j in set:
            set1.append(points[j])
        #去掉邻接表中孤立的点
        sub_graph = Graph()
        graph.generate_sub_graph(sub_graph,set)
        if len(set1) >= min_pts:
            if len(set1) >= 2:
                # 计算外接圆
                center_x, center_y, radius = calculate_bounding_circle(set1)
                print(f"外接圆的圆心为: ({center_x}, {center_y})")
                print(f"外接圆的半径为: {radius}")
                if radius > R:
                    print("-------------------------------------------------------------------------------")
                    print("邻接表是否改变=", "没变" if are_graphs_equal(graph, temp_graph) else "改变")
                    if are_graphs_equal(graph, temp_graph) == False:
                        result.append(DT_Clustering1(points, set1, sub_graph, R, min_pts,out_name,file_name))
                    else:
                        # return set1
                        result.append(set1)
                        write_file(set1,out_name,file_name)
                else:
                    # return set1
                    result.append(set1)
                    write_file(set1,out_name,file_name)
            else:
                # return set1
                result.append(set1)
                write_file(set1,out_name,file_name)
        # print("递归过程中：",result)
    return result
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

def write_file_class_points(Result_Path,my_list,name):
    with open(Result_Path+"/"+name, "w") as file:
        file.write("lng,lat" + "\n")
    with open(Result_Path+"/"+name, "a") as file:
        for i in range(0,len(my_list),2):
            print(my_list[i]+ "," +my_list[i+1])
            file.write(my_list[i].replace(" ","")+ "," + my_list[i+1].replace(" ","") + "\n")
def write_file_all_points(Result_Path,my_list,result_file_path):
    # with open("Class_All_Points/"+name, "w") as file:
    #     file.write("lng,lat" + "\n")
    with open(Result_Path+result_file_path, "a") as file:
        for i in range(0,len(my_list),2):
            print(my_list[i]+ "," +my_list[i+1])
            file.write(my_list[i].replace(" ","")+ "," + my_list[i+1].replace(" ","") + "\n")

def compute_file_centroid_all_points(Path):
    # 打开文件
    with open(Path, 'r', encoding='utf-8') as file:
        # 读取第一行并跳过
        file.readline()
        sum = [0.0, 0.0]
        begin = 0
        # 读取剩下的行并添加到列表中
        for line in file:
            sum[0] += float(line.split(",")[0])
            sum[1] += float(line.split(",")[1])
            begin += 1
    sum[0] = sum[0] / begin
    sum[1] = sum[1] / begin
    return sum[0], sum[1]

def write_file_centroid_all_points(lng, lat,result_file_path):
    with open(result_file_path, "a") as file:
        file.write(str(lng) + "," + str(lat) + "\n")
def first_step(file_name,Origin_Path,Result_Path):
    # 找出所有车牌的ID   车辆ID
    attribute = '车辆ID'
    select_single_vehicel(attribute, file_name)
    # 加载车辆ID文件
    file_name_id = "车辆ID"
    vihicel_id_road_path = Result_Path + file_name + file_name_id
    print("路径：", vihicel_id_road_path)
    # 加载所有车辆的文件
    vehicel = pd.read_csv(Origin_Path + file_name + ".txt", delimiter=',')
    # 按照车牌号将每辆车单独分为一个文件
    with open(vihicel_id_road_path + '.txt', 'r', encoding="utf-8") as file:
        content = file.read()
        vehicel_id = content.split(',')
    i = 1
    for veh in vehicel_id:
        if veh != "":
            print("车辆ID：", veh)
            filtered_df_city = vehicel[vehicel[file_name_id] == int(veh)]
            filtered_df_city.to_csv(Result_Path + file_name + '/' + 'vehicel' + str(i) + '.txt', index=False)
            i += 1
def second_step(Origin_Path2,Result_Path2,origin_vehicel_file_name):
    for j in range(1,20000):
        result_vehicel_file_name = "vehicel"+str(j)+"_result"
        res = time_sorted(origin_vehicel_file_name+str(j),result_vehicel_file_name,Result_Path2,Result_Path2)
        if res == False:
            break
def third_step(Origin_Path3,Result_Path3,file_name):
    input_file = open(Origin_Path3 + file_name + "1_result.txt", "r", encoding="utf-8")
    header = input_file.readline()  # 读取并弹出第一行
    print(header)
    Judgment_intersection(Origin_Path3, Result_Path3, file_name, header)
    # header = "车辆ID,经度,纬度,速度,方向,状态,数据发送时间,数据接收时间,车型,行政区号,具体道路位置,道路等级,城市,区县"
def sixth_step(Clusters_Points,class_file_path,result_file_path2,save_file_name):
    datas = read_txt_skip_first_line(Clusters_Points)
    k = 1
    # 添加标题
    with open(result_file_path2 + save_file_name, "w") as file:
        file.write("lng,lat" + "\n")
    for data in datas:
        print("data=", data)
        if len(data) > 1:
            temp = data.split(",")
            # print("len=",len(temp))
            write_file_class_points(class_file_path, temp, "Class" + str(k) + ".txt")
            k = k + 1
            write_file_all_points(result_file_path2, temp, save_file_name)

    save_center_file_name = "Class_All_Centroid_Points.txt"
    j = 1
    with open(result_file_path2 + save_center_file_name, "w") as file:
        file.write("lng,lat" + "\n")
    index = 1
    while True:
        Path = class_file_path + "Class" + str(index) + ".txt"
        if os.path.exists(Path) == False:
            print("不存在该文件")
            print("计算质心完成！")
            break
        else:
            lng, lat = compute_file_centroid_all_points(Path)
            # print("lng,lat=", lng, lat)
            write_file_centroid_all_points(lng, lat, result_file_path2 + save_center_file_name)
            index += 1
if __name__ == '__main__':
    # file_name = '2019-10-19'
    file_name = 'all_day'
    Origin_Path = '../traffic_wuhan_data/'
    Result_Path = '../../my_result/traffic_wuhan_data/result/'
    print(Result_Path + file_name)
    isPathExist(Result_Path + file_name)
    print("----------------------------------1、找出所有车辆的ID，并将每辆车对应的轨迹点分别保存--------------------------------------")
    # first_step(file_name,Origin_Path,Result_Path)
    print("----------------------------------2、将每辆车对应的轨迹按时间进行排序-------------------------------------")
    Origin_Path2 = Origin_Path+file_name + "/"
    Result_Path2 = Result_Path+file_name + "/"
    origin_vehicel_file_name = "vehicel"
    # second_step(Origin_Path2,Result_Path2,origin_vehicel_file_name)
    print("----------------------------------3、根据轨迹数据筛选拐点-------------------------------------")
    file_name3 = 'vehicel'
    Origin_Path3 = Result_Path2 + 'sort_time/'
    Result_Path3 = Result_Path2 + 'sort_time/point_result/'
    isPathExist(Result_Path3)
    # third_step(Origin_Path3,Result_Path3,file_name3)
    print("----------------------------------4、合并上一步筛选出的拐点-------------------------------------")
    file_name4 = 'adjoin_point_vehicel'
    Origin_Path4 = Origin_Path3 + 'point_result/'
    Result_Path4 = Result_Path3 + 'merge_point/'
    isPathExist(Result_Path4)
    # compute_all_point(Origin_Path4,Result_Path4,file_name4)
    print("----------------------------------5、对上一步得到的合并点进行聚类-------------------------------------")
    file_name5 = "My_Clusters.txt"
    csv_file_path = Result_Path4 + "final_merge_point.txt"
    out_name = Result_Path4 + "Clusters/"
    isPathExist(out_name)
    points = []
    # Open the CSV file and read its contents
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_file.readline()
        for line in csv_file:
            values = line.strip().split(",")
            x, y = map(float, values)  # Convert values to float
            points.append([x, y])
    # Convert the list of points to a NumPy array
    points = np.array(points)
    # 构建德劳内三角形剖分
    triang = Delaunay(points)
    with open(out_name + file_name5, "w") as file:
        file.write(csv_file_path + "\n")  # 在每个元素后添加换行符
    # 构建德劳内三角形
    tri = Delaunay(points)
    maxSideLengths = []
    tri.simplices = sorted(tri.simplices, key=sortingKey)
    R = 0.001  # 0.05  经（纬）度相差0.00001大约是实际距离的1.1119492664455877 米
    min_pts = 5
    results = DT_Clustering(points[:], tri.simplices, R, min_pts, out_name, file_name5)
    print("----------------------------------6、整合清洗上述聚类出的点-------------------------------------")
    Clusters_Points = Result_Path4 + "Clusters/My_Clusters.txt"
    class_file_path = Result_Path4 + "Clusters/Class/"
    result_file_path2 = Result_Path4 + "Clusters/Class_All_Centroid_Points/"
    save_file_name = "Class_All_Points.txt"
    isPathExist(class_file_path)
    isPathExist(result_file_path2)
    sixth_step(Clusters_Points,class_file_path,result_file_path2,save_file_name)





