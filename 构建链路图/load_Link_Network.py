import copy
import datetime
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
class Graph1:
    def __init__(self):
        # 使用字典来存储扩展邻接表
        self.adj_list = {}
    def add_vertex(self, vertex):
        # 如果顶点不在图中，则添加它
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []
    def add_edge(self, vertex1, vertex2):
        # 添加一条带权重的无向边
        if vertex1 not in self.adj_list:
            self.add_vertex(vertex1)
        if vertex2 not in self.adj_list:
            self.add_vertex(vertex2)
        if vertex2 not in self.adj_list[vertex1]:
            # print("vertex2=",vertex2,"vertex1=",vertex1)
            self.adj_list[vertex1].append(vertex2)

    def to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.adj_list, file)

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as file:
            adj_list = json.load(file)
        graph1 = Graph1()
        graph1.adj_list = adj_list
        return graph1

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

    def Inquire_lng_and_lat_to_velocity(vehicel_id, vehicel_index, vehicel_date):
        Path = f"../traffic_nanjing_data/result/2019-10-" + str(vehicel_date) + "/sort_time/vehicel" + str(
            vehicel_id) + "_result.txt"
        # 加载数据
        vehicel = pd.read_csv(Path, delimiter=',')
        current_vehicel = vehicel.reset_index(drop=True).to_numpy()
        # current_vehicel_lng = current_vehicel[vehicel_index][1]
        # current_vehicel_lat = current_vehicel[vehicel_index][2]
        velocity = current_vehicel[vehicel_index][3]
        # print("车辆坐标=", current_vehicel_lng, current_vehicel_lat)
        return velocity

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
def construc_point_to_speed(loaded_data):
    # loaded_data = {
    #     "100": [[1, 2, 17, 100], [1, 3, 17, 100], [1, 4, 17, 101]],
    #     "101": [[1, 4, 17, 102], [1, 5, 17, 103]],
    #     "102": [[1, 6, 17, 102], [1, 7, 17, 103], [1, 8, 17, 102], [1, 9, 17, 102], [1, 10, 17, 103], [1, 11, 17, 105]],
    #     "103": [[1, 12, 17, 105], [1, 13, 17, 103], [1, 14, 17, 102], [1, 15, 17, 109]],
    #     "104": [[1, 16, 17, 104], [1, 17, 17, 104], [2, 19, 17, 104], [2, 20, 17, 107]],
    #     "105": [[1, 18, 17, 106], [2, 21, 17, 105], [2, 22, 17, 105],[3,1,17,107]],
    # }
    loaded_data = loaded_data
    # for data in loaded_data:
    #     print("data=", data, "->", loaded_data[data], "--->", len(loaded_data[data]))
    orgin_loaded_data = copy.deepcopy(loaded_data)
    result = {}
    # print("*" * 100)
    #构建点对应各个方向速度的下标
    for data in loaded_data:
        next_start = 0
        record = 0
        same_lenth = 0
        # print("len(loaded_data[data])=",len(loaded_data[data]))
        for index in range(len(loaded_data[data])):
            if index+1 != len(loaded_data[data]):
                # print("index=", index, "---------------->", str(loaded_data[data][index][3]) == str(data))
                if str(loaded_data[data][index][3]) == str(data) and str(loaded_data[data][index][0]) == str(
                        loaded_data[data][index + 1][0]):
                    # if index != 0 and str(loaded_data[data][index][0])==str(loaded_data[data][index-1][0]):
                    # print("str(loaded_data[data][index][3])=", str(loaded_data[data][index]), "-------", str(data))
                    record += 1
                    same_lenth += 1
                else:
                    record += 1
                    same_lenth += 1
                    if record != 1:
                        # for i in range(next_start, index):
                        #     if str(loaded_data[data][i][3]) == str(data):
                        #         loaded_data[data][i][3] = loaded_data[data][index][3]
                        # print("next_start=", next_start, "record=", record, "same_lenth=", same_lenth, "index=", index)
                        if data not in result:
                            result[data] = []
                        if same_lenth == 1:
                            # print("使用瞬时速度",next_start,"方向:",data,"->",loaded_data[data][index][3])
                            value = next_start, loaded_data[data][index][3],
                            result[data].append(value)
                        elif same_lenth >= 2:
                            # print("求next_start 和 index 的平均速度",next_start, index,"方向:",data,"->",loaded_data[data][index][3])
                            value = next_start, index, loaded_data[data][index][3]
                            result[data].append(value)
                        else:
                            print("错误!")
                        # print("添加完成")
                        # print("---------------------------------------------")
                    else:
                        if data not in result:
                            result[data] = []
                        value = next_start, loaded_data[data][index][3],
                        result[data].append(value)
                    next_start = record
                    same_lenth = 0
            else:
                # print("index=", index, "---------------->", str(loaded_data[data][index][3]) == str(data))
                if str(loaded_data[data][index][3]) == str(data):
                    # if index != 0 and str(loaded_data[data][index][0])==str(loaded_data[data][index-1][0]):
                    # print("str(loaded_data[data][index][3])=", str(loaded_data[data][index]), "-------", str(data))
                    record += 1
                    same_lenth += 1
                else:
                    record += 1
                    same_lenth += 1
                    if record != 1:
                        # for i in range(next_start, index):
                        #     # print("str(loaded_data[data][i + 1][0])=", str(loaded_data[data][i]),
                        #     #       str(loaded_data[data][i + 1]))
                        #     # if i+1 != index:
                        #     if str(loaded_data[data][i][3]) == str(data):
                        #         loaded_data[data][i][3] = loaded_data[data][index][3]
                        # print("next_start=", next_start, "record=", record, "same_lenth=", same_lenth, "index=", index)
                        if data not in result:
                            result[data] = []
                        if same_lenth == 1:
                            # print("使用瞬时速度",next_start,"方向:",data,"->",loaded_data[data][index][3])
                            value = next_start, loaded_data[data][index][3],
                            result[data].append(value)
                        elif same_lenth >= 2:
                            # print("求next_start 和 index 的平均速度",next_start, index,"方向:",data,"->",loaded_data[data][index][3])
                            value = next_start, index, loaded_data[data][index][3]
                            result[data].append(value)
                        else:
                            print("错误!")
                        # print("添加完成")
                        # print("---------------------------------------------")
                    else:
                        if data not in result:
                            result[data] = []
                        value = next_start, loaded_data[data][index][3],
                        result[data].append(value)
                    next_start = record
                    same_lenth = 0

    # print("*" * 100)
    # for data in loaded_data:
    #     print("data=", data, "->", loaded_data[data], "--->", len(loaded_data[data]))
    # print("====================================")
    # print("清洗前的result=", result)
    result1 = copy.deepcopy(result)
    for key in result1:
        # print(key,"->", result1[key])
        for data in result1[key]:
            # print("data=",data,len(data))
            if len(data) == 2:
                if key == str(data[1]):
                    result[key].remove(data)
                    # print("data[1]=", data[1])
            if len(data) == 3:
                if key == str(data[2]):
                    result[key].remove(data)
                    # print("data[2]=", data[2])
    # print("清洗后的result=", result)
    # print("====================================")

    # 构建点对应各个方向速度的具体值
    save_point_corresponding_speed = {}
    for res in result:
        for index in range(len(result[res])):
            if len(result[res][index]) == 2:
                vehicel_id = orgin_loaded_data[res][result[res][index][0]][0]
                vehicel_index = orgin_loaded_data[res][result[res][index][0]][1]
                vehicel_date = orgin_loaded_data[res][result[res][index][0]][2]
                target = orgin_loaded_data[res][result[res][index][0]][3]
                speed = Inquire_lng_and_lat_to_velocity(vehicel_id, vehicel_index, vehicel_date)
                # print("当个坐标的瞬时速度speed=",speed,"m/s")
                if res not in save_point_corresponding_speed:
                    save_point_corresponding_speed[res] = []
                value = speed, target
                save_point_corresponding_speed[res].append(value)
            elif len(result[res][index]) == 3:
                # print("----"*20)
                # print("res=",res,"->",len(orgin_loaded_data[res]),"---->",result[res][index],"---")
                # print(orgin_loaded_data[res][46:58])
                # print("orgin_loaded_data=",orgin_loaded_data[res][result[res][index][0]],"------",orgin_loaded_data[res][result[res][index][1]])
                vehicel_id = orgin_loaded_data[res][result[res][index][0]][0]
                vehicel_index1 = orgin_loaded_data[res][result[res][index][0]][1]
                vehicel_index2 = orgin_loaded_data[res][result[res][index][1]][1]
                vehicel_date = orgin_loaded_data[res][result[res][index][0]][2]
                target = orgin_loaded_data[res][result[res][index][1]][3]
                # print("当前的索引点：",res," 求两个坐标的均值：",result[res][index][0],result[res][index][1])
                speed = Inquire_lng_and_lat_to_velocity1(vehicel_id, vehicel_index1, vehicel_index2, vehicel_date)
                # print("两个坐标的平均速度speed=",speed,"m/s")
                if speed!=np.inf:
                    if res not in save_point_corresponding_speed:
                        save_point_corresponding_speed[res] = []
                    value = speed, target
                    save_point_corresponding_speed[res].append(value)
            else:
                print("错误！")
    # print("save_point_corresponding_speed=", save_point_corresponding_speed)
    # for data in save_point_corresponding_speed:
    #     print(data,"->",save_point_corresponding_speed[data])

    #计算每一个方向所对应每一辆的速度
    merge_save_point_corresponding_speed = {}
    for key in save_point_corresponding_speed:
        # print(key,"->",merge_save_point_corresponding_speed[key])
        for data in save_point_corresponding_speed[key]:
            if key not in merge_save_point_corresponding_speed:
                merge_save_point_corresponding_speed[key] = {}
            if data[1] not in merge_save_point_corresponding_speed[key]:
                merge_save_point_corresponding_speed[key][data[1]] = []
                merge_save_point_corresponding_speed[key][data[1]].append(data[0])
                # merge_save_point_corresponding_speed[key][data[1]] = data[0]
            else:
                merge_save_point_corresponding_speed[key][data[1]].append(data[0])
                # merge_save_point_corresponding_speed[key][data[1]] = (merge_save_point_corresponding_speed[key][data[1]]+data[0])/2

    #计算每一个方向所有车辆的速度均值
    for key in merge_save_point_corresponding_speed:
        # print(key,"->",merge_save_point_corresponding_speed[key])
        for data in merge_save_point_corresponding_speed[key]:
            # print("data->",data)
            #在这个位置对每一个速度进行加噪
            merge_save_point_corresponding_speed[key][data] = float(np.sum(merge_save_point_corresponding_speed[key][data])/len(merge_save_point_corresponding_speed[key][data]))

    return merge_save_point_corresponding_speed
def Inquire_lng_and_lat_to_velocity(vehicel_id,vehicel_index,vehicel_date):
    Path = f"../traffic_nanjing_data/result/2019-10-"+str(vehicel_date)+"/sort_time/vehicel" + str(vehicel_id) + "_result.txt"
    # 加载数据
    vehicel = pd.read_csv(Path, delimiter=',')
    current_vehicel = vehicel.reset_index(drop=True).to_numpy()
    # current_vehicel_lng = current_vehicel[vehicel_index][1]
    # current_vehicel_lat = current_vehicel[vehicel_index][2]
    velocity = current_vehicel[vehicel_index][3]
    # print("车辆坐标=", current_vehicel_lng, current_vehicel_lat)
    return velocity
def Inquire_lng_and_lat_to_velocity1(vehicel_id,vehicel_index1,vehicel_index2,vehicel_date):
    Path = f"../traffic_nanjing_data/result/2019-10-"+str(vehicel_date)+"/sort_time/vehicel" + str(vehicel_id) + "_result.txt"
    # print("Path=",Path)
    # print("vehicel_id=",vehicel_id)
    # print("vehicel_index1=",vehicel_index1)
    # print("vehicel_index2=",vehicel_index2)
    # print("vehicel_date=",vehicel_date)
    # 加载数据
    vehicel = pd.read_csv(Path, delimiter=',')
    current_vehicel = vehicel.reset_index(drop=True).to_numpy()
    current_vehicel_lng1 = current_vehicel[vehicel_index1][1]
    current_vehicel_lat1 = current_vehicel[vehicel_index1][2]
    current_vehicel_lng2 = current_vehicel[vehicel_index2][1]
    current_vehicel_lat2 = current_vehicel[vehicel_index2][2]
    #单位：m
    distance = haversine(current_vehicel_lng1,current_vehicel_lat1,current_vehicel_lng2,current_vehicel_lat2)*1000
    time1 = current_vehicel[vehicel_index1][6]
    time2 = current_vehicel[vehicel_index2][6]
    time1_now = datetime.datetime.strptime(time1, '%Y-%m-%d %H:%M:%S')
    time2_now = datetime.datetime.strptime(time2, '%Y-%m-%d %H:%M:%S')
    #时间单位：转换为s
    diff_time = (time2_now-time1_now).total_seconds()
    if diff_time == 0:
        print("时间为0，错误")
        return np.inf
    # print("distance=",distance,"time1=",time1_now,"time2=",time2_now,"diff_time=",diff_time)
    velocity = distance/diff_time
    # print("速度=",velocity,"m/s")

    return velocity
def find_origin_to_target(graph,target):
    for vertex in graph:
        for edge in graph[vertex]:
            if edge[2] in target:
                # print("起始拐点到目标拐点",vertex, "->", edge[0],"---->边的编号：", edge[2])
                return edge[1]
    return None
def find_origin_speed_to_target(graph,result_speed_graph,Road_Network_Value):
    for k in range(1,8781):
        result_speed_graph[k] = 0
    for vertex in graph:
        for edge in graph[vertex]:
            # print(vertex,"---->",edge)
            for edge2 in Road_Network_Value[vertex]:
                if edge2[0] == edge:
                    result_speed_graph[edge2[2]] = graph[vertex][edge]
                    break

    # for edge in result_speed_graph:
    #     print(edge,"--->",result_speed_graph[edge])
    # find_origin_to_target(Road_Network_Value,[1554])
        # keys = graph[vertex].keys()

        # for edge in graph[vertex]:
        #     if edge[2] in target:
        #         print("起始拐点到目标拐点",vertex, "->", edge[0],"---->边的编号：", edge[2])
        #         return edge[1]
    # return None

def Load_Statistics_Intersection_Corresponding_To_Vehicel(day,date):
    with open('graph_file/' + str(
            day) + '/' + date + '/Statistics_Intersection_Corresponding_To_Vehicel_Filter_result' + date + '.json',
              'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    return loaded_data
if __name__ == "__main__":
    #加载链接网络
    Link_Network = Graph1.from_json('graph1.json')
    for vertex in Link_Network.adj_list:
        print(vertex, "->" ,Link_Network.adj_list[vertex][0:10],len(Link_Network.adj_list[vertex]))
    print(len(Link_Network.adj_list))
    print("-----"*30)

    arr = np.random.rand(72, 8780, 3)
    time_step = 0
    date1 = [17,18,19]
    for day in date1:
        for d in range(24):
            print("第%s天中的第%s小时"%(day,d))
            date = str(d) + "-" + str(d + 1)
            # 加载道路图
            Graph_name = "graph_file/"+str(day)+"/"+str(date)+"/new_2019-10-17-19_day_17-Construct_Filter_result_"+str(date)
            Road_Network = Graph.from_json(Graph_name + '.json')
            Road_Network_Value = Road_Network.adj_list
            # for vertex in Road_Network_Value:
            #     print(vertex, "->" ,Road_Network_Value[vertex])
            # 生成流量图
            result_flow_graph = {}
            for i in range(1, 8781):
                # print("i=",i)
                target = []
                target.append(i)
                flow_value = find_origin_to_target(Road_Network_Value, target)
                result_flow_graph[i] = flow_value
                # break
            # for vertex in result_flow_graph:
            #     print(vertex, "->" ,result_flow_graph[vertex])
            print("------------" * 10)
            # 生成速度图
            result_speed_graph = {}
            # 加载本轮的节点对应的经过车辆数量及信息结果
            Vehicel_Speed_Value = Load_Statistics_Intersection_Corresponding_To_Vehicel(day, date)
            # 计算每一个节点对应各个方向节点的平均速度
            merge_save_point_corresponding_speed = construc_point_to_speed(Vehicel_Speed_Value)
            # for vertex in merge_save_point_corresponding_speed:
            #     print(vertex, "->" ,merge_save_point_corresponding_speed[vertex])
            find_origin_speed_to_target(merge_save_point_corresponding_speed, result_speed_graph, Road_Network_Value)
            # p=0
            # for vertex in result_speed_graph:
            #     print(vertex, "->" ,result_speed_graph[vertex])
            #     p+=1
            # print("p=",p)
            # 构建权重矩阵
            for j in range(1,8781):
                # print("j=",j)
                value = result_flow_graph[j], 0 ,result_speed_graph[j]
                arr[time_step][j-1] = value

            time_step += 1
    print("time_step=",time_step)
    print(arr.shape,len(arr[0]))
    print(arr[0])
    # 使用 np.savez 函数将数组保存为 npz 文件
    np.savez('PEMS043.npz', data=arr)
            # print("value=", value)
            # print("arr=", arr[0][0])
            # arr[0][0]


    # for vertex in result_speed_graph:
    #     print(vertex, ":" ,result_speed_graph[vertex])

    # target = [290,291,554]
    # target = [7595, 329, 416, 583, 692, 1118, 1572, 1610, 2586, 2890, 3081, 3503, 3870, 4049, 4081, 4638, 4783, 5410, 5557, 5729, 5902, 7564, 7579, 7593, 7638, 8181, 8462]
    # find_origin_to_target(Road_Network_Value, target)