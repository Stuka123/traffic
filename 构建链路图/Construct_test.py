import datetime
import json

import pandas as pd

Sign = False
number = 1
class Graph:
    def __init__(self):
        # 使用字典来存储扩展邻接表
        self.adj_list = {}
    def add_vertex(self, vertex):
        # 如果顶点不在图中，则添加它
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []
    def add_edge(self, vertex1, vertex2, weight):
        global number
        # 添加一条带权重的无向边
        if vertex1 not in self.adj_list:
            self.add_vertex(vertex1)
        if vertex2 not in self.adj_list:
            self.add_vertex(vertex2)
        if (vertex2,weight) not in self.adj_list[vertex1]:
            # 在vertex1的邻接表中添加(vertex2, weight)元组
            # print("-==========",self.adj_list[vertex1])
            p_i = [vertex2,weight,number]
            self.adj_list[vertex1].append(p_i)
            number += 1
    def add_weight(self, vertex1, vertex2, temp_index):
        if vertex1 not in self.adj_list:
            return False
        if vertex2 not in self.adj_list:
            return False
        # print(self.adj_list[vertex1][temp_index])
        self.adj_list[vertex1][temp_index][1] += 1

        # print("进入权重函数1", vertex1, vertex2, self.adj_list[vertex1], self.adj_list[vertex1][0])
        # self.adj_list[vertex1][0][1] = 2
        # print("进入权重函数2", vertex1, vertex2, self.adj_list[vertex1], self.adj_list[vertex1][0])
        # index = 0
        # for vertex, weight in self.adj_list[vertex1]:
        #     if vertex == vertex2:
        #         self.adj_list[vertex1][index][1] += 1
        #     index += 1
        # print("进入权重函数3", vertex1, vertex2, self.adj_list[vertex1], self.adj_list[vertex1][0])
        return True
    # def print_graph(self):
    #     # 打印图的元素及其权重
    #     for vertex in self.adj_list:
    #         adj_vertices = self.adj_list[vertex]
    #         print(f"{vertex}: {adj_vertices}")  # 排序后打印，以便更清晰地查看
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
def isExistRegion(current_vehicel,Current_Centroid_Points):
    global Sign
    Judge = False
    current_vehicel = current_vehicel.reset_index(drop=True).to_numpy()
    # print("current_vehicel=", current_vehicel)
    current_vehicel_lng = current_vehicel[0][1]
    current_vehicel_lat = current_vehicel[0][2]

    # print("current_vehicel_lng=",current_vehicel_lng,"current_vehicel_lat=",current_vehicel_lat)
    Centroid_Points_lng = Current_Centroid_Points[0]
    Centroid_Points_lat = Current_Centroid_Points[1]

    #经（纬）度相差0.00001大约是实际距离的1.1119492664455877 米
    is_in_region = (current_vehicel_lng <= Centroid_Points_lng + 0.001
            and current_vehicel_lng >= Centroid_Points_lng - 0.001
            and current_vehicel_lat >= Centroid_Points_lat - 0.001
            and current_vehicel_lat <= Centroid_Points_lat + 0.001)

    if is_in_region and not Sign:
        # print("该轨迹点在区域内且未被统计过！")
        Sign = True
        Judge = True
        condition = 1
    elif is_in_region and Sign:
        # print("连续轨迹的点在区域内，已经被统计过，因此不统计")
        Judge = False
        condition = 2
    elif not is_in_region and Sign:
        # print("连续轨迹的点刚出区域，已经被统计过，因此不统计！")
        Sign = False
        Judge = False
        condition = 3
    else:
        # print("不在区域内，且标签值已经复原！")
        Judge = False
        condition = 4

    return Judge , current_vehicel_lng , current_vehicel_lat , Centroid_Points_lng , Centroid_Points_lat,condition

def Record_Vehcle(Last_Index,Current_Index,vehicel_index,Statistics_Intersection_Corresponding_To_Vehicel):
    Node = str(Last_Index)
    if Node not in Statistics_Intersection_Corresponding_To_Vehicel:
        Statistics_Intersection_Corresponding_To_Vehicel[Node] = []
    # print("vehicel_index",vehicel_index)
    Value = vehicel_index[0],vehicel_index[1],vehicel_index[2],Current_Index
    Statistics_Intersection_Corresponding_To_Vehicel[Node].append(Value)

def construct_adjacency_list(Last_Centroid_Points,Current_Centroid_Points,Last_Index,Current_Index,graph):
    # print("Last_Centroid_Points=",Last_Centroid_Points,"Last_Index=",Last_Index)
    # print("Current_Centroid_Points=",Current_Centroid_Points,"Current_Index=",Current_Index)
    flag1 = False
    temp_index = 0
    #判断当前节点是否在上一个节点的邻接表中
    for vertex,weight,number in graph.adj_list[Last_Index]:
        if vertex == Current_Index:
            flag1 = True
            break
        temp_index += 1
        # print("遍历",vertex,weight)
    # print("Flag=",flag)
    # print("之前一个节点相连接点的个数：",len(graph.adj_list[Last_Index]))
    if not flag1:
        graph.add_edge(Last_Index,Current_Index,1)
        # print("加边")
    else:
        graph.add_weight(Last_Index,Current_Index,temp_index)
        # print("增加权重")
def main():
    graph = Graph()
    Path_Centroid = f"../DT_Clustering-main/Class_All_Centroid_Points/2019-10-17-19-Class_All_Centroid_Points.txt"

    Graph_name = "graph_file/2019-10-17-19_day_17-Construct_result"
    print("Graph_name=",Graph_name)
    Centroid_Points = pd.read_csv(Path_Centroid, delimiter=',')
    Centroid_Points_Correspond_Index = []
    print("===================================初始化======================================")
    # 初始化图，并将图中每一个点对应的坐标保存到列表中
    for k in range(len(Centroid_Points)):
        current_point = Centroid_Points.iloc[k:k + 1].reset_index(drop=True).to_numpy()
        temp = current_point[0][0], current_point[0][1]
        Centroid_Points_Correspond_Index.append(temp)
        # print("temp=",temp)
        graph.add_vertex(k)
    # print(Centroid_Points_Correspond_Index)
    # print(len(Centroid_Points_Correspond_Index), Centroid_Points_Correspond_Index[0])

    print("===================================遍历车辆轨迹点=======================================")
    # day = [17,18,19]
    # list11 = [2,2,2]
    day = [17, 18, 19]
    list11 = [11154, 11180, 10558]
    list_i = 0
    result = []
    flag = 0
    Statistics_Intersection_Corresponding_To_Vehicel = {}
    for date in day:
        for k in range(1, list11[list_i]):  # 11153
            vehicel_flag = 0
            # print("date=", date)
            Path = f"../traffic_nanjing_data/result/2019-10-" + str(date) + "/sort_time/vehicel" + str(
                k) + "_result.txt"
            # print("Path=", Path)
            # 加载数据
            filtered_vehicel = pd.read_csv(Path, delimiter=',')
            print("第%s辆车" % k," 长度=", len(filtered_vehicel),"date=", date)
            # 遍历车辆的所有轨迹点
            for i in range(len(filtered_vehicel)):
                Current_vehicel_points = filtered_vehicel.iloc[i:i + 1]
                #手动找的时候记得索引加2
                original_indices = Current_vehicel_points.index.tolist()[0]
                # 遍历所有的质心
                for j in range(len(Centroid_Points_Correspond_Index)):
                    # 第k辆车中的第i个点,链路图中的Last_Centroid_Points, Current_Centroid_Points之间
                    Current_Centroid_Points = Centroid_Points_Correspond_Index[j]
                    Current_Index = j
                    # 判断当前轨迹点是否在质心区域
                    Judge, current_vehicel_lng, current_vehicel_lat, Centroid_Points_lng, Centroid_Points_lat, condition \
                        = isExistRegion(Current_vehicel_points, Current_Centroid_Points)
                    if Judge:
                        if flag == 0 or vehicel_flag == 0:
                            Last_Index = j
                            Last_Centroid_Points = Current_Centroid_Points
                            vehicel_index = k, original_indices,date
                            Last_Vehicle_Index = vehicel_index
                            flag += 1
                            vehicel_flag += 1
                        else:
                            if Last_Index != Current_Index:
                                Centroid_Points_temp = float(Centroid_Points_lng), float(Centroid_Points_lat)
                                result.append(Centroid_Points_temp)
                                # 将本轮质心与上一轮质心加入邻接表中
                                construct_adjacency_list(Last_Centroid_Points, Current_Centroid_Points, Last_Index,
                                                         Current_Index,
                                                         graph)
                                # 记录通过拐点的车辆轨迹及其方向
                                Record_Vehcle(Last_Index, Current_Index, Last_Vehicle_Index,
                                              Statistics_Intersection_Corresponding_To_Vehicel)
                                vehicel_index = k, original_indices,date
                                Last_Vehicle_Index = vehicel_index

                                # 记录上一次保存的节点
                                Last_Centroid_Points = Current_Centroid_Points
                                Last_Index = Current_Index
                                flag += 1
                                break
                            else:
                                Record_Vehcle(Last_Index, Current_Index, Last_Vehicle_Index,
                                              Statistics_Intersection_Corresponding_To_Vehicel)
                                vehicel_index = k, original_indices,date
                                Last_Vehicle_Index = vehicel_index
                # if i == 30:
                #     break
        list_i += 1
    # print(len(result),result)
    # print("======================")
    # print("Statistics_Intersection_Corresponding_To_Vehicel=",Statistics_Intersection_Corresponding_To_Vehicel)

    for i in graph.adj_list:
        if graph.adj_list[i] != []:
            print(i,graph.adj_list[i])
    graph.to_json(Graph_name+".json")

    # 将字典保存到JSON文件中
    with open('Statistics_Intersection_Corresponding_To_Vehicel_result.json', 'w', encoding='utf-8') as f:
        json.dump(Statistics_Intersection_Corresponding_To_Vehicel, f, ensure_ascii=False, indent=4)

    # loaded_graph = Graph.from_json(Graph_name+'.json')
    # print(loaded_graph.adj_list)  # 输出加载的图的数据
    #
    # # 读取并加载JSON文件中的数据
    # with open('Statistics_Intersection_Corresponding_To_Vehicel_result.json', 'r', encoding='utf-8') as f:
    #     loaded_data = json.load(f)
    #     # print(loaded_data)
    #     for data in loaded_data:
    #         # if len(loaded_data[data]) > 10:
    #         #     if data == str(5596):
    #         print("data=",data,"->",loaded_data[data],"--->",len(loaded_data[data]))

if __name__ == "__main__":
    print("开始时间：",datetime.datetime.now())
    main()
    print("结束时间：", datetime.datetime.now())



