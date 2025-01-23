import argparse
import datetime
import json
import os

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
def init_graph_and_Statistics_Intersection_Corresponding_To_Vehicel():
    graph = Graph()
    Path_Centroid = f"../DT_Clustering-main/Class_All_Centroid_Points/2019-10-17-19-Filter_Inflection_Point.txt"
    Centroid_Points = pd.read_csv(Path_Centroid, delimiter=',')
    Graph_name = "2019-10-17-19_day_17-Construct_Filter_result_"
    # print("Graph_name=", Graph_name)
    Centroid_Points_Correspond_Index = []
    # 初始化图，并将图中每一个点对应的坐标保存到列表中
    for k in range(len(Centroid_Points)):
        current_point = Centroid_Points.iloc[k:k + 1].reset_index(drop=True).to_numpy()
        temp = current_point[0][0], current_point[0][1]
        Centroid_Points_Correspond_Index.append(temp)
        graph.add_vertex(k)
    Statistics_Intersection_Corresponding_To_Vehicel = {}
    return graph ,Statistics_Intersection_Corresponding_To_Vehicel ,Graph_name,Centroid_Points_Correspond_Index
def main():
    print("===================================初始化======================================")
    # graph, Statistics_Intersection_Corresponding_To_Vehicel,Graph_name,Centroid_Points_Correspond_Index = init_graph_and_Statistics_Intersection_Corresponding_To_Vehicel()
    print("===================================遍历车辆轨迹点=======================================")
    # day = [17,18,19]
    # list11 = [11154,11180,10558]#,11180, 10558
    # list_i = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=int, default="17", choices=[17,18,19])
    parser.add_argument('--vehicel_len', type=int, default="17", choices=[2,11154, 11180, 10558])
    args = parser.parse_args()

    date1 = args.date
    vehicel_len = args.vehicel_len
    day = []
    list11 = []
    day.append(date1)
    list11.append(vehicel_len)

    list_i = 0
    flag = 0
    for date in day:
        for i in range(0,24):
            #初始化图、统计车辆速度的信息图
            result = []
            graph, Statistics_Intersection_Corresponding_To_Vehicel, Graph_name, Centroid_Points_Correspond_Index = init_graph_and_Statistics_Intersection_Corresponding_To_Vehicel()
            strtime = str(i)+"-"+str(i+1)
            if i < 9:
                start_time = '2019-10-'+str(date)+' 0'+str(i)+':00:00'
                end_time = '2019-10-'+str(date)+' 0'+str(i+1)+':00:00'
            elif i == 23:
                start_time = '2019-10-'+str(date)+' 23:00:00'
                end_time = '2019-10-'+str(date)+' 23:59:59'
            else:
                start_time = '2019-10-' + str(date) + ' ' + str(i) + ':00:00'
                end_time = '2019-10-' + str(date) + ' ' + str(i + 1) + ':00:00'
            print("start_time:",start_time,"-end_time:",end_time,strtime)
            start_datetime = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            end_datetime = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            # print(start_datetime,"-",end_datetime)
            #遍历所有车辆
            for k in range(1, list11[list_i]):  # 11153
                vehicel_flag = 0
                # print("date=", date)
                Path = f"../traffic_nanjing_data/result/2019-10-" + str(date) + "/sort_time/vehicel" + str(
                    k) + "_result.txt"
                # 加载数据
                vehicel = pd.read_csv(Path, delimiter=',')
                # print("start_time:","-end_time:",  start_datetime, end_datetime, strtime,"Path:",Path,"-------->",len(vehicel))
                filtered_vehicel = vehicel[(pd.to_datetime(vehicel['数据发送时间']) >= start_datetime) & (
                        pd.to_datetime(vehicel['数据发送时间']) <= end_datetime)]
                print("第%s辆车" % k, " 长度=", len(filtered_vehicel), "date=", date)
                # 遍历每一辆车所对应的所有轨迹点
                for i in range(len(filtered_vehicel)):
                    Current_vehicel_points = filtered_vehicel.iloc[i:i + 1]
                    # 手动找的时候记得索引加2
                    original_indices = Current_vehicel_points.index.tolist()[0]

                    if original_indices != 0:
                        filtered_vehicel_last = vehicel.iloc[original_indices-1:original_indices]["数据发送时间"].values.tolist()[0]
                        filtered_vehicel_current = vehicel.iloc[original_indices:original_indices + 1]["数据发送时间"].values.tolist()[0]
                        start_datetime1 = datetime.datetime.strptime(filtered_vehicel_last, '%Y-%m-%d %H:%M:%S')
                        end_datetime1 = datetime.datetime.strptime(filtered_vehicel_current, '%Y-%m-%d %H:%M:%S')
                        diff = (end_datetime1 - start_datetime1).total_seconds()
                    else:
                        diff = 301
                    if diff <= 300:
                        # 遍历所有的质心
                        for j in range(len(Centroid_Points_Correspond_Index)):
                            # 第k辆车中的第i个点。链路图中的Last_Centroid_Points, Current_Centroid_Points之间
                            Current_Centroid_Points = Centroid_Points_Correspond_Index[j]
                            Current_Index = j
                            # 判断当前轨迹点是否在质心区域
                            Judge, current_vehicel_lng, current_vehicel_lat, Centroid_Points_lng, Centroid_Points_lat, condition \
                                = isExistRegion(Current_vehicel_points, Current_Centroid_Points)
                            if Judge:
                                if flag == 0 or vehicel_flag == 0:
                                    Last_Index = j
                                    Last_Centroid_Points = Current_Centroid_Points
                                    vehicel_index = k, original_indices, date
                                    Last_Vehicle_Index = vehicel_index
                                    flag += 1
                                    vehicel_flag += 1
                                else:
                                    if Last_Index != Current_Index:
                                        Centroid_Points_temp = float(Centroid_Points_lng), float(Centroid_Points_lat)
                                        result.append(Centroid_Points_temp)
                                        # 将本轮质心与上一轮质心加入邻接表中
                                        construct_adjacency_list(Last_Centroid_Points, Current_Centroid_Points,
                                                                 Last_Index,
                                                                 Current_Index,
                                                                 graph)
                                        # 记录通过拐点的车辆轨迹及其方向
                                        Record_Vehcle(Last_Index, Current_Index, Last_Vehicle_Index,
                                                      Statistics_Intersection_Corresponding_To_Vehicel)
                                        vehicel_index = k, original_indices, date
                                        Last_Vehicle_Index = vehicel_index

                                        # 记录上一次保存的节点
                                        Last_Index = Current_Index
                                        Last_Centroid_Points = Current_Centroid_Points
                                        flag += 1
                                        break
                                    else:
                                        Record_Vehcle(Last_Index, Current_Index, Last_Vehicle_Index,
                                                      Statistics_Intersection_Corresponding_To_Vehicel)
                                        vehicel_index = k, original_indices, date
                                        Last_Vehicle_Index = vehicel_index
                        # if i == 30:
                        #     break

            #保存文件
            folder_path = 'graph_file/' + str(date) + '/' + strtime +'/'  # 将这里替换为实际想要检查和创建的文件夹路径
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"文件夹 {folder_path} 不存在，已成功创建。")
            else:
                print(f"文件夹 {folder_path} 已存在。")

            graph.to_json(folder_path+Graph_name + strtime + ".json")
            print(start_datetime,"-",end_datetime,"的图文件保存完成")

            # 将字典保存到JSON文件中
            with open(folder_path+'Statistics_Intersection_Corresponding_To_Vehicel_Filter_result' + strtime + '.json', 'w',
                      encoding='utf-8') as f:
                json.dump(Statistics_Intersection_Corresponding_To_Vehicel, f, ensure_ascii=False, indent=4)
            print(start_datetime,"-",end_datetime,"对应的点流量统计数据完成")
            print("-------------------------------------------")
        list_i += 1

    # print(len(result),result)
    # print("======================")
    # print("Statistics_Intersection_Corresponding_To_Vehicel=",Statistics_Intersection_Corresponding_To_Vehicel)

    # for i in graph.adj_list:
    #     if graph.adj_list[i] != []:
    #         print(i,"->",graph.adj_list[i])

    # loaded_graph = Graph.from_json(Graph_name+'.json')
    # print(loaded_graph.adj_list)  # 输出加载的图的数据
    #
    # # 读取并加载JSON文件中的数据
    # with open('Statistics_Intersection_Corresponding_To_Vehicel_Filter_result.json', 'r', encoding='utf-8') as f:
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



