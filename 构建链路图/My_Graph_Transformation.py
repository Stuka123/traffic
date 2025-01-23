import copy
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
if __name__ == '__main__':
    print("=======================加载统计的链接流量图==============================")
    # loaded_graph = Graph.from_json('graph.json')
    # result = loaded_graph.adj_list
    # print(result)  # 输出加载的图的数据
    # for vertex in result:
    #     if result[vertex] != []:
    #         print(vertex, "->", result[vertex])
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

    Graph_name = "2019-10-17-19_day_17-Construct_Filter_result"
    loaded_graph = Graph.from_json(Graph_name + '.json')
    result = loaded_graph.adj_list

    #遍历原始图
    for vertex in result:
        # for edge in result[vertex]:
        #     if edge[1]>10:
        #         print(vertex, "->", result[vertex])
        if result[vertex] != []:
            # print(Centroid_Points_Correspond_Index[int(vertex)][0],",",Centroid_Points_Correspond_Index[int(vertex)][1])
            print(vertex, "->", result[vertex])
    print("=============================转换成链接网络============================")
    #测试用例
    graph = {
        'A':[['C',1,2],['E',1,9]],
        'C':[['A',2,1],['D',1,5],['B',1,3]],
        'D':[['C',1,6]],
        'B':[['C',4,4],['E',3,10]],
        'E':[['A',4,7],['B',2,8]]
    }
    for vertex in graph:
        if graph[vertex] != []:
            print(vertex, "->", graph[vertex])

    print("---------------------")
    # print(graph)
    graph1 = transform_point_to_edge(result)
    k=0
    for vertex in graph1.adj_list:
        print(vertex, ":", graph1.adj_list[vertex])
        k+=1
    print("k=",k)

    zero_matrix = [[0 for _ in range(k)] for _ in range(k)]
    # print(zero_matrix)

    # 定义CSV文件路径
    csv_file_path = 'zero_matrix1.csv'
    for i in range(1,k+1):
        for j in range(1,k+1):
            if j in graph1.adj_list[i]:
                zero_matrix[i-1][j-1] = 1
        print("i=",i)

    df = pd.DataFrame(zero_matrix, index=[str(i) for i in range(len(zero_matrix))],
                      columns=[str(i) for i in range(len(zero_matrix))])
    df.to_csv(csv_file_path)

    print("---------------------")
    #2: [1, 5, 3]
    target = [2,1,5,3]
    # target = [7595, 329, 416, 583, 692, 1118, 1572, 1610, 2586, 2890, 3081, 3503, 3870, 4049, 4081, 4638, 4783, 5410, 5557, 5729, 5902, 7564, 7579, 7593, 7638, 8181, 8462]
    find_origin_to_target(graph, target)
    print("="*200)
    graph1.to_json("graph1.json")
    # loaded_graph = Graph1.from_json('graph1.json')
    # # 输出加载的图的数据
    # # print(loaded_graph.adj_list)
    # for vertex in loaded_graph.adj_list:
    #     print(vertex, "->", loaded_graph.adj_list[vertex])

    # 7595 : [329, 416, 583, 692, 1118, 1572, 1610, 2586, 2890, 3081, 3503, 3870, 4049, 4081, 4638, 4783, 5410, 5557, 5729, 5902, 7564, 7579, 7593, 7638, 8181, 8462]
    list1 = [24, 424]
    for index in list1:
        print(Centroid_Points_Correspond_Index[index][0], ",", Centroid_Points_Correspond_Index[index][1])


#A->[[C,xx,2]]
#C->[[A,xx,1],[D,xx,5],[B,xx,3]]
#D->[[C,xx,6]]
#B->[[C,xx,4]]
#==================
#1->[[2,xx]]
#2->[[1,xx],[3,xx],[5,xx]]
#3->[[4,xx]]
#4->[[3,xx],[1,xx],[5,xx]]
#5->[[6,xx]]
#6->[[3,xx],[5,xx],[1,xx]]
