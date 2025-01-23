import csv
import json

import numpy as np
import pandas as pd

def load_file():
    # 打开npz文件
    with np.load('PEMS04.npz') as data:
        # 获取文件中的所有数组名称
        array_names = data.keys()
        # 遍历数组名称并获取数组
        print(array_names)
        for name in array_names:
            array = data[name]
            print(f'数组名称: {name}')

        print(array)
        print(array.shape)
        print("len =", len(array), len(array[0]), len(array[0][0]))
        print("len =", len(array[0][0]), "---", array[0][0], array[1][1])

        print(array[0][0])

        # for data in array:
        #     print(data)
if __name__ == "__main__":
    load_file()
    # zero_matrix = "zero_matrix.csv"
    # res = pd.read_csv(zero_matrix, delimiter=',',encoding='utf-8',header=None)
    # # print(res)
    # # print(len(res))
    # result_numpy = np.array(res)
    # print(result_numpy.shape,result_numpy[0].shape)
    # print(result_numpy)
    # print(result_numpy.size(-1))

    # with open(zero_matrix, mode='r',newline='') as outfile:
    #     csv_reader = csv.reader(outfile)
    #     i=0
    #     for row in csv_reader:
    #         i+=1
    #     print(i)
            # print(row)
            # break

    # for i in range(0, len(res)):

    # print("_____name_____")
    # load_file()
    # flow_name1 = "example1----------------------.csv"
    # load_flow1 = pd.read_csv(flow_name1, delimiter=',', encoding='utf-8', )
    # # print(load_flow1)
    # print(len(load_flow1))
    #
    # flow_name2 = "example2----------------------.csv"
    # load_flow2 = pd.read_csv(flow_name2, delimiter=',',encoding='utf-8',)
    # # print(load_flow2)
    # print(len(load_flow2),load_flow2.iloc[0:1])
    #
    # Filter_Inflection_Point_Name = "2019-10-17-19-Filter_Inflection_Point.txt"
    # Filter_Inflection_Point = pd.read_csv(Filter_Inflection_Point_Name, delimiter=',',encoding='utf-8')
    # print(len(Filter_Inflection_Point))
    # #(72,540,2)
    # #(72,拐点数,3)
    #
    # test_num = np.array(
    #     [
    #     [["流量1", "速度1"],
    #     ["流量2", "速度2"],
    #     ["流量3", "速度3"]]
    #     ]
    # )
    #
    # print(test_num,test_num.shape)
    #
    # arr = np.random.rand(4, 3, 2)
    # print(arr)

# class Graph:
#     def __init__(self):
#         # 使用字典来存储扩展邻接表
#         self.adj_list = {}
#     def add_vertex(self, vertex):
#         # 如果顶点不在图中，则添加它
#         if vertex not in self.adj_list:
#             self.adj_list[vertex] = []
#     def add_edge(self, vertex1, vertex2, weight):
#         # 添加一条带权重的无向边
#         if vertex1 not in self.adj_list:
#             self.add_vertex(vertex1)
#         if vertex2 not in self.adj_list:
#             self.add_vertex(vertex2)
#         if (vertex2,weight) not in self.adj_list[vertex1]:
#             self.adj_list[vertex1].append([vertex2, weight])
#     def add_weight(self, vertex1, vertex2,temp_index):
#         if vertex1 not in self.adj_list:
#             return False
#         if vertex2 not in self.adj_list:
#             return False
#
#         self.adj_list[vertex1][temp_index][1] += 1
#         return True
#     def __str__(self):
#         # 打印图的扩展邻接表表示
#         result = ""
#         for vertex in self.adj_list:
#             if len(self.adj_list[vertex])!=0:
#                 edges = self.adj_list[vertex]
#                 edges_str = ", ".join(f"({v}, {w})" for v, w in edges)
#                 result += f"{vertex} -> [{edges_str}]\n"
#         return result
#
#     def to_json(self, filename):
#         with open(filename, 'w') as file:
#             json.dump(self.adj_list, file)
#
#     @staticmethod
#     def from_json(filename):
#         with open(filename, 'r') as file:
#             adj_list = json.load(file)
#         graph = Graph()
#         graph.adj_list = adj_list
#         return graph
# class Graph1:
#     def __init__(self):
#         # 使用字典来存储扩展邻接表
#         self.adj_list = {}
#     def add_vertex(self, vertex):
#         # 如果顶点不在图中，则添加它
#         if vertex not in self.adj_list:
#             self.adj_list[vertex] = []
#     def add_edge(self, vertex1, vertex2):
#         # 添加一条带权重的无向边
#         if vertex1 not in self.adj_list:
#             self.add_vertex(vertex1)
#         if vertex2 not in self.adj_list:
#             self.add_vertex(vertex2)
#         if vertex2 not in self.adj_list[vertex1]:
#             # print("vertex2=",vertex2,"vertex1=",vertex1)
#             self.adj_list[vertex1].append(vertex2)
#
#     def to_json(self, filename):
#         with open(filename, 'w') as file:
#             json.dump(self.adj_list, file)
#
#     @staticmethod
#     def from_json(filename):
#         with open(filename, 'r') as file:
#             adj_list = json.load(file)
#         graph1 = Graph1()
#         graph1.adj_list = adj_list
#         return graph1
# def transform_point_to_edge(graph):
#     graph1 = Graph1()
#     for vertex in graph:
#         # print(vertex,":",graph[vertex])
#         for edge in graph[vertex]:
#             # print("当前这条边=",edge,edge[2],edge[0])
#             # print("hkjdlsam====",graph[str(edge[0])])
#             for neighbor in graph[str(edge[0])]:
#                 # print("neighbor=",neighbor)
#                 # print("当前加入的点为：",edge[2],neighbor[2])
#                 graph1.add_edge(edge[2],neighbor[2])
#     return graph1
# def find_origin_to_target(graph,target):
#     for vertex in graph:
#         for edge in graph[vertex]:
#             if edge[2] in target:
#                 print("起始拐点到目标拐点",vertex, "->", edge[0],"---->边的编号：", edge[2])
# def calculate_len(graph):
#     # 获取最大的编号
#     max = 0
#     for vertex in graph:
#         for edge in graph[vertex]:
#             if edge[2] > max:
#                 max = edge[2]
#     print("len=", max)
#     return max
# def Calculate_Flow_Matrix(graph,all_len_id):
#     result_list = []
#     for i in range(all_len_id+1):
#         result_list.append(0)
#     total = 0
#     for vertex in graph:
#         total += len(graph[vertex])
#     #获取对应编号的流量并保存到result_list集合中
#     for i in range(1,all_len_id + 1):
#         flag1 = 0
#         flag2 = 0
#         for vertex in graph:
#             for edge in graph[vertex]:
#                 if i == edge[2]:
#                     result_list[i] = edge[1]
#                     flag1 += 1
#                 if flag1 != 0:
#                     flag2 += 1
#                     break
#             if flag2 != 0:
#                 break
#     return result_list,all_len_id
#
# def save_data(csv_file_path,result_list,time):
#     with open(csv_file_path, 'a') as csvfile:
#         csvfile.write(time+",")
#         for i in range(1,len(result_list)):
#             csvfile.write(str(result_list[i])+",")
#         csvfile.write("\n")
# if __name__ == "__main__":
#     print("test")
#
#     #通过整个图来构建文件
#     All_graph_name = "graph_file/all/2019-10-17-19_day_17-Construct_Filter_result"
#     loaded_graph = Graph.from_json(All_graph_name + '.json')
#     all_result = loaded_graph.adj_list
#     # 遍历原始图
#     for vertex in all_result:
#         if all_result[vertex] != []:
#             print(vertex, "->", all_result[vertex])
#     all_len_id = calculate_len(all_result)
#     # 指定CSV文件路径
#     csv_file_path = "example3----------------------.csv"
#     print(f"数据已成功保存到 {csv_file_path}")
#     with open(csv_file_path, 'w') as csvfile:
#         csvfile.write("节点编号,")
#         for i in range(1, all_len_id + 1):
#             csvfile.write(str(i) + ",")
#         csvfile.write("\n")
#
#     date = "0-1"
#     Graph_name = "graph_file/17/"+date+"/new_2019-10-17-19_day_17-Construct_Filter_result_"+date
#     loaded_graph = Graph.from_json(Graph_name + '.json')
#     result = loaded_graph.adj_list
#     # 遍历原始图
#     # for vertex in result:
#     #     if result[vertex] != []:
#     #         print(vertex, "->", result[vertex])
#
#     # result_list, len_id = Calculate_Flow_Matrix(result, all_len_id)
#     # print("result_list", len(result_list), "---->", result_list)
#     # save_data(csv_file_path,result_list, "all_time")
#
#     for vertex in result:
#         for edge in result[vertex]:
#             if edge[2] == 290:
#                 print(vertex, "->", result[vertex])
#
#     # print("--------------------------")
#     # Graph_name = "graph_file/17/" + date + "/2019-10-17-19_day_17-Construct_Filter_result_" + date
#     # loaded_graph = Graph.from_json(Graph_name + '.json')
#     # result1 = loaded_graph.adj_list
#     #
#     # for vertex in result1:
#     #     if vertex == str(24):
#     #         print(vertex, "->", result1[vertex])
#         # for edge in result1[vertex]:
