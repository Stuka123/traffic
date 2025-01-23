import json

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
def transform_point_to_edge(graph):
    graph1 = Graph1()
    for vertex in graph:
        # print(vertex,":",graph[vertex])
        for edge in graph[vertex]:
            # print("当前这条边=",edge,edge[2],edge[0])
            # print("hkjdlsam====",graph[str(edge[0])])
            for neighbor in graph[str(edge[0])]:
                # print("neighbor=",neighbor)
                # print("当前加入的点为：",edge[2],neighbor[2])
                graph1.add_edge(edge[2],neighbor[2])
    return graph1
def find_origin_to_target(graph,target):
    for vertex in graph:
        for edge in graph[vertex]:
            if edge[2] in target:
                print("起始拐点到目标拐点",vertex, "->", edge[0],"---->边的编号：", edge[2])

def calculate_len(graph):
    # 获取最大的编号
    max = 0
    for vertex in graph:
        for edge in graph[vertex]:
            if edge[2] > max:
                max = edge[2]
    print("len=", max)
    return max
def Calculate_Flow_Matrix(graph,all_len_id):
    result_list = []
    for i in range(all_len_id+1):
        result_list.append(0)

    total = 0
    for vertex in graph:
        total += len(graph[vertex])
    # print("total=",total)
    # print(len(result_list),all_len_id)
    # print("="*100)
    #获取对应编号的流量并保存到result_list集合中
    for i in range(1,all_len_id + 1):
        flag1 = 0
        flag2 = 0
        for vertex in graph:
            # print(vertex,"->",graph[vertex])
            for edge in graph[vertex]:
                if i == edge[2]:
                    # print("i=",i)
                    # print("i=",i,"----------","edge=",vertex,"---------->",edge)
                    result_list[i] = edge[1]
                    flag1 += 1
                if flag1 != 0:
                    flag2 += 1
                    break
            if flag2 != 0:
                break
    return result_list,all_len_id

def save_data(csv_file_path,date,result_list,time):
    with open(csv_file_path, 'a') as csvfile:
        csvfile.write(time+",")
        for i in range(1,len(result_list)):
            csvfile.write(str(result_list[i])+",")
        csvfile.write("\n")
        # csvfile.write("1--2,")
        # for data in data_list1:
        #     csvfile.write(str(data)+",")
if __name__ == '__main__':
    print("=======================加载统计的链接流量图==============================")
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

    #通过整个图来构建文件
    All_graph_name = "graph_file/all/2019-10-17-19_day_17-Construct_Filter_result"
    # All_graph_name = "graph_file/all/2019-10-17-19_day_17-Construct_Filter_result"
    loaded_graph = Graph.from_json(All_graph_name + '.json')
    all_result = loaded_graph.adj_list
    # 遍历原始图
    for vertex in all_result:
        if all_result[vertex] != []:
            print(vertex, "->", all_result[vertex])
    all_len_id = calculate_len(all_result)
    # 指定CSV文件路径
    csv_file_path = "graph_file/Flow.csv"
    print(f"数据已成功保存到 {csv_file_path}")
    with open(csv_file_path, 'w') as csvfile:
        csvfile.write("节点编号,")
        for i in range(1, all_len_id + 1):
            csvfile.write(str(i) + ",")
        csvfile.write("\n")

    Graph_name = "graph_file/all/2019-10-17-19_day_17-Construct_Filter_result"
    loaded_graph = Graph.from_json(Graph_name + '.json')
    result = loaded_graph.adj_list

    result_list, len_id = Calculate_Flow_Matrix(result, all_len_id)
    # print("result_list", len(result_list), "---->", result_list)
    # save_data(csv_file_path, "0-24", result_list, "0-24")

    date1 = [17,18,19]
    for day in date1:
        for d in range(24):
            print(day)
            date = str(d) + "-" + str(d + 1)
            if d < 9:
                start_time = '0' + str(d) + ':00:00'
                end_time = '0' + str(d + 1) + ':00:00'
                time = '2019-10-' + str(day) +' '+ start_time + '-' + end_time
            elif d == 23:
                start_time = '23:00:00'
                end_time = '23:59:59'
                time = '2019-10-' + str(day) +' '+ start_time + '-' + end_time
            else:
                start_time = str(d) + ':00:00'
                end_time = str(d + 1) + ':00:00'
                time = '2019-10-' + str(day) + ' ' + start_time + '-' + end_time

            print("start_time-end_time:", time)
            # date = "0-1"
            Graph_name = "graph_file/"+str(day)+"/" + date + "/new_2019-10-17-19_day_17-Construct_Filter_result_" + date
            loaded_graph = Graph.from_json(Graph_name + '.json')
            result = loaded_graph.adj_list
            # 遍历原始图
            # for vertex in result:
            #     if result[vertex] != []:
            #         print(vertex, "->", result[vertex])

            result_list, len_id = Calculate_Flow_Matrix(result, all_len_id)
            print("result_list", len(result_list),"---->",result_list)
            save_data(csv_file_path, date, result_list,time)


    # 7595 : [329, 416, 583, 692, 1118, 1572, 1610, 2586, 2890, 3081, 3503, 3870, 4049, 4081, 4638, 4783, 5410, 5557, 5729, 5902, 7564, 7579, 7593, 7638, 8181, 8462]
    # list1 = [0,1,263,2,4,334,110,336,275,48,270,6,131,278,132,276,47,41,16,21,133,102]
    # for index in list1:
    #     print(Centroid_Points_Correspond_Index[index][0], ",", Centroid_Points_Correspond_Index[index][1])


