import os
def isPathExist(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    #     print(f"目录 '{directory_path}' 已创建。")
    # else:
    #     print(f"目录 '{directory_path}' 已经存在。")
if __name__ == '__main__':
    load_file = "../../my_result/traffic_wuhan_data/result/"
    file_day_name = '2019-10-'
    part_file = "/sort_time/point_result/merge_point/"

    result_file_path = "../../my_result/traffic_wuhan_data/result/all_day/sort_time/point_result/merge_point/"
    result_file_name = "final_merge_point.txt"
    isPathExist(result_file_path)

    output_file = open(load_file+"all_day/"+ part_file + result_file_name, "w", encoding="utf-8")
    header = "经度,纬度"
    output_file.write(header)
    output_file.write("\n")
    output_file.close()
    points = []
    date = [17,18,19]
    for day in date:
        data_path = load_file + file_day_name +str(day) + part_file + result_file_name
        print(data_path,isPathExist(data_path))
        # if isPathExist(data_path):
        with open(data_path, 'r', encoding='utf-8') as csv_file:
            csv_file.readline()
            for line in csv_file:
                values = line.strip().split(",")
                x, y = map(float, values)  # Convert values to float
                points.append([x, y])
        print(len(points),points[0],points[1],points[-1])

    i = 0
    total = len(points)
    for point in points:
        i+=1
        print(i/total*100,"%")
        # print(point[0],point[1])
        with open(load_file + "all_day/" + part_file + result_file_name, 'a', encoding='utf-8') as file:
            res = str(point[0]) + "," + str(point[1])
            # print(res)
            file.write(res+"\n")
        # break
    output_file.close()




