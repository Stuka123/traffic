import json

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
# def adjacency_list_to_matrix(loaded_graph,num_vertices):
#     #初始化邻接矩阵
#     adj_matrix = [[float("inf")] * num_vertices for _ in range(num_vertices)]
#     for i in range(len(adj_matrix)):
#         for j in range(len(adj_matrix)):
#             if i == j and i!=0 and j!=0:
#                 adj_matrix[i][j] = 0
#     print(adj_matrix)
#     #遍历邻接表
#     for vertex in loaded_graph.adj_list:
#         print(vertex,"->",loaded_graph.adj_list[vertex])
#         for edge in loaded_graph.adj_list[vertex]:
#             temp_vertex = int(vertex)
#             temp_edge = int(edge)
#             adj_matrix[temp_vertex][temp_edge] = 1
#
#     return adj_matrix
def adjacency_list_to_matrix(loaded_graph,num_vertices):
    #初始化邻接矩阵
    adj_matrix = [[float(0)] * num_vertices for _ in range(num_vertices)]
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i == j and i!=0 and j!=0:
                adj_matrix[i][j] = 0.0
    print(len(adj_matrix[0]),len(adj_matrix[1]))
    #遍历邻接表
    for vertex in loaded_graph.adj_list:
        print(vertex,"->",loaded_graph.adj_list[vertex])
        for edge in loaded_graph.adj_list[vertex]:
            print("vertex=",vertex,"edge=",edge)
            temp_vertex = int(vertex)
            temp_edge = int(edge[0])
            adj_matrix[temp_vertex][temp_edge] = float(edge[1])
    return adj_matrix
if __name__ == "__main__":
    #加载邻接表
    loaded_graph = Graph1.from_json('2019-10-17-19_day_17-Construct_Filter_result.json')
    num_vertices = len(loaded_graph.adj_list)+1
    print("num_vertices=",num_vertices)

    result_adj_matrix = adjacency_list_to_matrix(loaded_graph,num_vertices)
    print("邻接矩阵的形状",len(result_adj_matrix[0]),len(result_adj_matrix[1]))
    print("======================遍历转换的邻接矩阵========================")
    # for element in result_adj_matrix:
    #     print(element)

    import pandas as pd

    # 定义带权邻接矩阵示例（这里以简单的4x4矩阵为例）
    weighted_adj_matrix = [
        [0, 2, 0, 1],
        [2, 0, 3, 0],
        [0, 3, 0, 4],
        [1, 0, 4, 0]
    ]
    #
    # for element in weighted_adj_matrix:
    #     print(element)

    # 将矩阵转换为DataFrame
    df = pd.DataFrame(result_adj_matrix, index=[str(i) for i in range(len(result_adj_matrix))],
                      columns=[str(i) for i in range(len(result_adj_matrix))])

    # 定义CSV文件路径
    csv_file_path = 'weighted_adj_matrix.csv'
    df.to_csv(csv_file_path)

