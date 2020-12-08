import pandas as pd
import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

class Zone():
    def __init__(self, id, expected_flow, max_flow , average_length, first_flow=0, whole_outflow=0, whole_inflow=0, accumulated_vehicles=0,old_accumulated_vehicles=0, updated=False):
        self.id = id
        self.updated = updated
        self.zone_neighbors = [[1, 3], [0, 2, 4], [1, 5], [0, 4, 6], [1, 3, 5, 7], [2, 4, 8], [3, 7], [4, 6, 8], [5, 7]]
        # self.zone_neighbors = [[2,3],[1,4],[1,4],[2,3]]
        self.expected_flow = expected_flow
        self.max_flow = max_flow
        self.average_length = average_length
        self.first_flow = first_flow
        self.outflow_list = []
        self.inflow_list = []
        self.whole_outflow = whole_outflow
        self.whole_inflow = whole_inflow
        self.accumulated_vehicles = accumulated_vehicles
        self.old_accumulated_vehicles = accumulated_vehicles

    def get_zone_id(self):
        return "%s" % (self.id)
    # 得到小区的ID，用字符串形式展示

    def creat_first_flow(self):
        first_flow = random.randint(1000,2000)
        self.first_flow = first_flow

    def create_outflow(self):
        zone_neighbors_id_list = self.zone_neighbors[self.id]
        # 根据self小区的ID，得到其邻居小区的ID
        outflow_list = random.poisson(lam=self.expected_flow, size=(len(zone_neighbors_id_list)))
        # outflow_list = random.randint(1000,2000, size=(len(zone_neighbors_id_list)))
        # 根据泊松分布生成self小区前往邻居各个小区的流量
        for elem in self.outflow_list:
            if elem > self.max_flow:
                elem = self.max_flow
                return elem
            elif elem < self.expected_flow:
                elem = self.expected_flow
                return elem
            else:
                return elem
        self.whole_outflow = sum(outflow_list)
        #self小区的总生成流量
        self.outflow_list = outflow_list
        # 更新self小区的流出列表
        self.updated = True

def get_Zones(number,expected_flow_list,max_flow_list,average_length_list):
    #得到小区（数量、期望交通流的列表、最大交通流的列表）
    zones=[]
    #初始化小区数组为空
    for i in range(number):
        zones.append(Zone(i,expected_flow_list[i],max_flow_list[i],average_length_list[i]))
    return zones
    #为zones数组增加每次增加一个i,expected_flow，max_flow

def simulation_main(endtime):
    #定义一个仿真函数
    zones = get_Zones(9,[1000,1100,1200,1300,1400,1500,1600,1700,1800],[1300,1400,1500,1600,1700,1800,1800,1800,1800],[100,110,120,130,140,150,160,170,180])
    # zones = get_Zones(4,[1000,1100,1200,1300],[1500,1600,1700,1800],[100,110,120,130])

    for elem in zones:
        elem.creat_first_flow()
        elem.old_accumulated_vehicles = elem.first_flow
        # old_accumulated_vehicles.append(first_flow)


    dfindex = []
    #初始化索引为空
    for elem in zones:
        dfindex.append(elem.get_zone_id())
    #对每个zone使用get_zone_id()函数获得其id，再把id加到索引列表dfindex

    df_outflow = pd.DataFrame(index=dfindex)
    #通过pandas的DataFrame函数创建df_inflow,索引为dfindex
    df_inflow = pd.DataFrame(index=dfindex)
    #通过pandas的DataFrame函数创建df_outflow,索引为dfindex
    df_accumulated_vehicles = pd.DataFrame(index=dfindex)
    df_density = pd.DataFrame(index=dfindex)


    for t in range(endtime):
        whole_outflow = []
        whole_inflow = []
        accumulated_vehicles = []
        density = []
        #分别对whole_outflow,whole_inflow定义为空
        for elem in zones:
            elem.create_outflow()
            #调用create_outflow()函数，从小区生成流量流向邻居小区
            elem.inflow_list = []
        for k in range(len(zones)):
            for j in range(len(zones[k].outflow_list)):
                zones[zones[k].zone_neighbors[k][j]].inflow_list.append(zones[k].outflow_list[j])
                #给k小区的j邻居小区的inflow_list,加上k小区流向j邻居小区的outflow
        #k,j循环后，相当于已经把所有的outflow_list里的元素加到对应的inflow_list里
        for elem in zones:

            whole_outflow.append(sum((elem.outflow_list)))
            whole_inflow.append(sum((elem.inflow_list)))

            if sum(elem.outflow_list) > elem.old_accumulated_vehicles:
                outflow = elem.old_accumulated_vehicles
            else:
                outflow = sum(elem.outflow_list)
            elem.accumulated_vehicles = elem.old_accumulated_vehicles + sum(elem.inflow_list) - outflow
            elem.old_accumulated_vehicles = elem.accumulated_vehicles
            accumulated_vehicles.append(elem.accumulated_vehicles)
            density.append((elem.old_accumulated_vehicles + sum((elem.inflow_list)) - outflow) / elem.average_length)


            elem.updated = False

        df_outflow["t%i"%t] = whole_outflow
        df_inflow["t%i"%t] = whole_inflow
        df_accumulated_vehicles["t%i"%t] = accumulated_vehicles
        df_density["t%i"%t] = density

        x = []
        for elem in density:
            x.append(elem)
        step = 3
        d = [x[i:i + step] for i in range(0, len(x), step)]
        print(d)
        #将一次t时间内获得的9个密度分为三组数据

        f, (ax) = plt.subplots(figsize=(6, 4))
        #figsize是绘图窗口，宽4高6英寸，axis是这个绘图窗口的坐标系
        cmap = sns.diverging_palette(200, 20, sep=20, as_cmap=True)
        #从数字到色彩空间的映射，取值是matplotlib包里的colormap名称或颜色对象，或者表示颜色的列表；改参数默认值：根据center参数设定
        pt = d  # pt为数据框或者是协方差矩阵
        sns.heatmap(pt, linewidths=0.05,annot=True, ax=ax, vmax=100, vmin=0, cmap=cmap)
        ax.set_title('density map')
        ax.set_xlabel('')
        ax.set_xticklabels([7,8,9])
        ax.set_ylabel('')
        ax.set_yticklabels([1,4,7])
        plt.show()

        #把t采用十进制整数形式表示t
        print(whole_outflow)
        print(whole_inflow)
        print(accumulated_vehicles)
        print(density)

        df_outflow.to_csv("whole_outflow.csv")
        df_inflow.to_csv("whole_inflow.csv")
        df_accumulated_vehicles.to_csv("accumulated_vehicles.csv")
        df_density.to_csv("density.csv")



    return df_outflow,df_inflow,df_accumulated_vehicles,df_density

if __name__ == "__main__":
    simulation_main(10)
