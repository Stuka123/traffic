import copy
import json

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

def construct_a_new_traffic_matrix(current_result,copy_all_result):
    for vertex in current_result:
        if current_result[vertex] != []:
            for i in range(len(current_result[vertex])):
                for j in range(len(copy_all_result[vertex])):
                    if current_result[vertex][i][0] == copy_all_result[vertex][j][0]:
                        copy_all_result[vertex][j][1] = current_result[vertex][i][1]
                        break

if __name__ == "__main__":
    # 通过整个图来构建文件
    All_graph_name = "graph_file/all/2019-10-17-19_day_17-Construct_Filter_result"
    loaded_graph = Graph.from_json(All_graph_name + '.json')
    all_result = loaded_graph.adj_list
    # for vertex in all_result:
    #     print(vertex, "->", all_result[vertex])

    date1 = [19]
    for day in date1:
        for k in range(24):
            # 复制一份权重全为0的图
            copy_all_result = copy.deepcopy(all_result)
            for vertex in all_result:
                for i in range(len(all_result[vertex])):
                    copy_all_result[vertex][i][1] = 0
            # 加载当前的时间段的图文件
            # Current_graph_name = "graph_file/17/0-1/2019-10-17-19_day_17-Construct_Filter_result_0-1"
            date = str(k) + "-" + str(k + 1)
            Current_graph_name = "graph_file/" + str(day) + "/" + date + "/2019-10-17-19_day_17-Construct_Filter_result_" + date
            Current_loaded_graph = Graph.from_json(Current_graph_name + '.json')
            current_result = Current_loaded_graph.adj_list
            # 将当前时间段的图文件赋值到整个图文件
            construct_a_new_traffic_matrix(current_result, copy_all_result)
            print("day=",day,"date=",date)
            # 保存新的文件
            graph1 = Graph()
            graph1.adj_list = copy_all_result
            graph1.to_json("graph_file/" + str(day) + "/" + date + "/new_2019-10-17-19_day_17-Construct_Filter_result_" + date + ".json")




