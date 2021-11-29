import sys
from PyQt5 import QtGui, QtCore, QtWidgets
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') # using pyqt5 backend
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import geopandas as gpd
import networkx as nx
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import time



click_count = 0
startXY, endXY, point_1, point_2, choice = None, None, None, None, None

fun_dict = {"net_d": "networkx Dijkstra", "n_astar": "networkx A*", "poorman": "poor man's A*"}

edges_gpd = gpd.read_file("data/Tartu_edges_gpd.shp")  
del edges_gpd["FID"]

df_nodes = pd.read_csv("data/df_nodes.csv")
df_ways = pd.read_csv("data/df_ways.csv")


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
     
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Path planner")

        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)


    def first_frame(self):

        msg = QtWidgets.QLabel("Please select the path planning algorithm")

        btn1 = QtWidgets.QRadioButton("networkx A*")
        btn1.clicked.connect(self.net_select)  

        btn2 = QtWidgets.QRadioButton("poor man's A*")
        btn2.clicked.connect(self.poor_select)

        btn4 = QtWidgets.QRadioButton("networkx Dijkstra")
        btn4.clicked.connect(self.net_d)

        # frame 2 - clearing the frame, ready to render map and add text 
        btn3 = QtWidgets.QPushButton("Next")
        btn3.clicked.connect(self.second_frame) 



        self.layout.addWidget(msg)
        self.layout.addWidget(btn1)
        self.layout.addWidget(btn2)
        self.layout.addWidget(btn4)
        self.layout.addWidget(btn3)

        self.setLayout(self.layout)

    def net_select(self):
        global choice
        choice = "n_astar"

    def poor_select(self):
        global choice
        choice = "poorman"

    def net_d(self):
        global choice 
        choice =  "net_d"
 

    def second_frame(self):
        
        try:
            self.ax.cla()
        except:
            pass
        finally:
            for i in reversed(range(self.layout.count())):
                item = self.layout.itemAt(i)

                if isinstance(item, QtWidgets.QWidgetItem):
                    #print "widget" + str(item)
                    item.widget().close()
                    # or
                    # item.widget().setParent(None)
                elif isinstance(item, QtWidgets.QSpacerItem):
                    pass

                else:
                    self.layout.clearLayout(item.layout())

            # remove the item from layout
            self.layout.removeItem(item)
            pass    
        #self.layout = QtWidgets.QVBoxLayout(self.main_widget)

 #       except:
 #           pass

        self.fig = Figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        msg = QtWidgets.QLabel(f"You chose: {fun_dict[choice]}. Click on two places in the map and then click on Apply")
        toolbar = NavigationToolbar(self.canvas, self) #MESSES WITH THE CLICKING 
        edges_gpd.plot(ax = self.ax, figsize=(10, 10), linewidth=0.5)
        #minx, miny, maxx, maxy  = edges_gpd.total_bounds
        #self.ax.set_xlim(minx+0.05, maxx-0.05)
        #self.ax.set_ylim(miny+0.05, maxy-0.05)
        self.ax.set_xlim(26.62, 26.82) # for TARTU only
        self.ax.set_ylim(58.33, 58.42)
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.apply) 
        btn_clear = QtWidgets.QPushButton("Clear map")  # optional TODO: add a Back to the algo choice page, splash page 
        btn_clear.clicked.connect(self.count_to_zero) 

        self.layout.addWidget(msg)
        self.layout.addWidget(toolbar)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(btn_apply)
        self.layout.addWidget(btn_clear)

        #widget = QtWidgets.QWidget()
        self.setLayout(self.layout)
        #self.setCentralWidget(widget)
        id = self.canvas.mpl_connect('button_press_event', self.onclick)

        self.cursor = matplotlib.widgets.Cursor(self.ax, color='purple')
        self.canvas.setCursor(QtGui.QCursor(    QtCore.Qt.ArrowCursor))


    def onclick(self, event):

        global click_count, startXY, endXY, point_1, point_2

        if click_count == 0:
            #point_1 = None
            startXY = (event.xdata, event.ydata)
            point_1 = self.ax.scatter(event.xdata, event.ydata, color="red")
        if click_count == 1:
            #point_2 = None
            endXY = (event.xdata, event.ydata)
            point_2 = self.ax.scatter(event.xdata, event.ydata, color="green")        
        click_count += 1
    
    def count_to_zero(self):
        global startXY, endXY, point_1, point_2, click_count
        startXY, endXY = (None,None), (None,None)
        click_count = 0
        point_1.remove() # removing points from map
        point_2.remove()
        point_1, point_2 = None, None

    def apply(self):
        global choice, click_count 

        print("Clicked on Apply")
        print(startXY, endXY)

        dist2coord = df_nodes.copy()

        dist2coord["dist"] = np.linalg.norm(startXY - df_nodes[["lon", "lat"]].apply(pd.to_numeric).values, axis = 1)
        start_node_ID = dist2coord[dist2coord.dist == dist2coord.dist.min()]

        dist2coord["dist"] = np.linalg.norm(endXY - df_nodes[["lon", "lat"]].apply(pd.to_numeric).values, axis = 1)
        end_node_ID = dist2coord[dist2coord.dist == dist2coord.dist.min()]

        start_node_ID = int(start_node_ID["id"])
        end_node_ID = int(end_node_ID["id"])

        # Create a graph
        G = nx.from_pandas_edgelist(df_ways, "A_id", "B_id", ["distance", "edgeID"])

        # Add coordinates to each node so Astar algorithm can calculate heuristic
        node_attributes = df_nodes.set_index('id').T.to_dict('list')
        nx.set_node_attributes(G, node_attributes, name="coords")
        click_count = 0

        if choice == "n_astar":
            
            current_time = time.time()

            shortest_path = nx.astar_path(G, start_node_ID, end_node_ID, weight="distance")
  
            shortest_path_IDs = []
            for i in range(len(shortest_path) - 1):
                shortest_path_IDs.append(G.get_edge_data(shortest_path[i], shortest_path[i+1]).get("edgeID"))


            # Get locations to zoom into for the general area where a path was found
            margin_percent = 0.05

            min_lat = min([G.nodes()[node].get("coords")[0] for node in shortest_path])
            max_lat = max([G.nodes()[node].get("coords")[0] for node in shortest_path])
            lat_margin = (max_lat - min_lat) * margin_percent

            min_lon = min([G.nodes()[node].get("coords")[1] for node in shortest_path])
            max_lon = max([G.nodes()[node].get("coords")[1] for node in shortest_path])
            lon_margin = (max_lon - min_lon) * margin_percent

            self.ax.cla() ## clear the map
            for i in reversed(range(self.layout.count())):
                item = self.layout.itemAt(i)

                if isinstance(item, QtWidgets.QWidgetItem):
                    #print "widget" + str(item)
                    item.widget().close()
                    # or
                    # item.widget().setParent(None)
                elif isinstance(item, QtWidgets.QSpacerItem):
                    pass
                    #print "spacer " + str(item)
                    # no need to do extra stuff
                else:
                    self.layout.clearLayout(item.layout())

            # remove the item from layout
            self.layout.removeItem(item)    

            self.canvas = FigureCanvas(self.fig)
            msg = QtWidgets.QLabel(f"Networkx A* based on the chosen points")
            edges_gpd.plot(ax=self.ax, linewidth=0.5)
            edges_gpd.loc[shortest_path_IDs].plot(ax = self.ax, linewidth=5, cmap = "viridis")
            end_time = time.time() 
            processing = end_time - current_time
            msg1 = QtWidgets.QLabel(f"It took about {round(processing,2)} seconds to render this map.")
            btn_back = QtWidgets.QPushButton("Back to the map")
            btn_back.clicked.connect(self.second_frame) 

            self.ax.set_xlim(min_lat - lat_margin, max_lat + lat_margin)# for TARTU only
            self.ax.set_ylim(min_lon - lon_margin, max_lon + lon_margin)   

            self.layout.addWidget(msg)
            self.layout.addWidget(self.canvas)
            self.layout.addWidget(msg1)
            self.layout.addWidget(btn_back)
            
            self.setLayout(self.layout)


        elif choice == "poorman":

            current_time = time.time()
            # Self-implemented
            shortest_path, edge_history = poor_mans_a_star(start_node_ID, end_node_ID, astar_heuristic)

            #Convert OSM node IDs in found shortest path to IDs of edges
            shortest_path_IDs = []
            for i in range(len(shortest_path) - 1):
                shortest_path_IDs.append(G.get_edge_data(shortest_path[i], shortest_path[i+1]).get("edgeID"))

            # Generate colors for path-finding visualisation
            colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(edge_history)))

            # Append shortest path to edge_history so it would also be drawn during animation
            for i in shortest_path_IDs:
                edge_history.append([i])
                
            # Get locations to zoom into for the general area where a path was found
            margin_percent = 0.05

            min_lat = min([G.nodes()[node].get("coords")[0] for node in shortest_path])
            max_lat = max([G.nodes()[node].get("coords")[0] for node in shortest_path])
            lat_margin = (max_lat - min_lat) * margin_percent

            min_lon = min([G.nodes()[node].get("coords")[1] for node in shortest_path])
            max_lon = max([G.nodes()[node].get("coords")[1] for node in shortest_path])
            lon_margin = (max_lon - min_lon) * margin_percent

            self.ax.cla() ## clear the map
            for i in reversed(range(self.layout.count())):
                item = self.layout.itemAt(i)

                if isinstance(item, QtWidgets.QWidgetItem):
                    #print "widget" + str(item)
                    item.widget().close()
                    # or
                    # item.widget().setParent(None)
                elif isinstance(item, QtWidgets.QSpacerItem):
                    pass
                    #print "spacer " + str(item)
                    # no need to do extra stuff
                else:
                    self.layout.clearLayout(item.layout())

            # remove the item from layout
            self.layout.removeItem(item)    

            self.canvas = FigureCanvas(self.fig)
            msg = QtWidgets.QLabel("Poor man's A* based on the chosen points")
            edges_gpd.plot(ax=self.ax, linewidth=0.5)


            btn_back = QtWidgets.QPushButton("Back to the map")
            btn_back.clicked.connect(self.second_frame) 

            self.ax.set_xlim(min_lat - lat_margin, max_lat + lat_margin)
            self.ax.set_ylim(min_lon - lon_margin, max_lon + lon_margin)   

            # Draw visualisation of how shortest path was found and then also plot shortest path itself
            def animate(i):
                # This raw plt.plot() vs gpd.plot() is a bit faster.
                art = []
                color = colors[i] if i < len(colors) else "red"
                for x in edges_gpd.iloc[edge_history[i]].values:
                    x1 = x[0].coords[0][0]
                    x2 = x[0].coords[1][0]
                    y1 = x[0].coords[0][1]
                    y2 = x[0].coords[1][1]
                    
                    art = self.ax.plot((x1, x2), (y1, y2), color = color, linewidth = 4)
                return art

            # Animation initialiser
            self.anim = FuncAnimation(self.fig, animate, interval=0.005, repeat=False, blit=False, frames=len(edge_history))

            end_time = time.time()
            processing = end_time - current_time
            msg1 = QtWidgets.QLabel(f"It took about {round(processing,2)} seconds to render this map.")

            self.layout.addWidget(msg)
            self.layout.addWidget(self.canvas)
            self.layout.addWidget(msg1)
            self.layout.addWidget(btn_back)
            
            self.setLayout(self.layout)

        elif choice == "net_d":

            current_time = time.time()
            shortest_path = nx.bidirectional_dijkstra(G, start_node_ID, end_node_ID, weight="distance")
  
            shortest_path_IDs = []
            for i in range(len(shortest_path[1]) - 1):
                shortest_path_IDs.append(G.get_edge_data(shortest_path[1][i], shortest_path[1][i+1]).get("edgeID"))


            # Get locations to zoom into for the general area where a path was found
            margin_percent = 0.05

            min_lat = min([G.nodes()[node].get("coords")[0] for node in shortest_path[1]])
            max_lat = max([G.nodes()[node].get("coords")[0] for node in shortest_path[1]])
            lat_margin = (max_lat - min_lat) * margin_percent

            min_lon = min([G.nodes()[node].get("coords")[1] for node in shortest_path[1]])
            max_lon = max([G.nodes()[node].get("coords")[1] for node in shortest_path[1]])
            lon_margin = (max_lon - min_lon) * margin_percent

            self.ax.cla() ## clear the map
            for i in reversed(range(self.layout.count())):
                item = self.layout.itemAt(i)

                if isinstance(item, QtWidgets.QWidgetItem):
                    #print "widget" + str(item)
                    item.widget().close()
                    # or
                    # item.widget().setParent(None)
                elif isinstance(item, QtWidgets.QSpacerItem):
                    pass
                    #print "spacer " + str(item)
                    # no need to do extra stuff
                else:
                    self.layout.clearLayout(item.layout())

            # remove the item from layout
            self.layout.removeItem(item)    

            self.canvas = FigureCanvas(self.fig)
            msg = QtWidgets.QLabel("Networkx Dijkstra based on the chosen points")
            edges_gpd.plot(ax=self.ax, linewidth=0.5)
            edges_gpd.loc[shortest_path_IDs].plot(ax = self.ax, linewidth=5, cmap = "viridis")

            end_time = time.time()
            processing = end_time - current_time
            msg1 = QtWidgets.QLabel(f"It took about {round(processing,2)} seconds to render this map.")

            btn_back = QtWidgets.QPushButton("Back to the map")
            btn_back.clicked.connect(self.second_frame) 

            self.ax.set_xlim(min_lat - lat_margin, max_lat + lat_margin)# for TARTU only
            self.ax.set_ylim(min_lon - lon_margin, max_lon + lon_margin)   

            self.layout.addWidget(msg)
            self.layout.addWidget(self.canvas)
            self.layout.addWidget(msg1)
            self.layout.addWidget(btn_back)
            
            self.setLayout(self.layout)

     
  
        return 


        

G = nx.from_pandas_edgelist(df_ways, "A_id", "B_id", ["distance", "edgeID"])
# Add coordinates to each node so Astar algorithm can calculate heuristic
node_attributes = df_nodes.set_index('id').T.to_dict('list')
nx.set_node_attributes(G, node_attributes, name="coords")

def astar_heuristic(start_node_id, end_node_id):
    start_coord = G.nodes()[start_node_id].get("coords")
    end_coord = G.nodes()[end_node_id].get("coords")
    return np.linalg.norm(np.asarray(start_coord) - np.asarray(end_coord))

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.insert(0, current)
    return total_path

def poor_mans_a_star(start, end, h):
    open_set = set([start])
    came_from_dict = dict()

    g_score_dict = {node_id: np.inf for node_id in list(G.nodes())}
    g_score_dict[start] = 0

    f_score_dict = {node_id: np.inf for node_id in list(G.nodes())}
    f_score_dict[start] = h(start, end)
    
    edges_per_step = []

    while len(open_set) != 0:
        current = None
        currentMin = np.inf
        for x in open_set:
            if f_score_dict.get(x) < currentMin:
                currentMin = f_score_dict.get(x)
                current = x

        if current == end:
            return reconstruct_path(came_from_dict, current), edges_per_step

        open_set.remove(current)
        edges_this_step = []
        for neighbor in G.neighbors(current):
            #edges_this_step.append(G.get_edge_data(current, neighbor).get("edgeID"))
            tentative_g_score = g_score_dict[current] + \
                                G.get_edge_data(current, neighbor).get("distance")
            if tentative_g_score < g_score_dict[neighbor]:
                edges_this_step.append(G.get_edge_data(current, neighbor).get("edgeID"))
                came_from_dict[neighbor] = current
                g_score_dict[neighbor] = tentative_g_score
                f_score_dict[neighbor] = g_score_dict[neighbor] + h(neighbor, end)

                if neighbor not in open_set:
                    open_set.add(neighbor)
        edges_per_step.append(edges_this_step)
                    
    raise Exception('No path found.')



def mains():
    qApp = QtWidgets.QApplication([])
    aw = ApplicationWindow()
    aw.first_frame()
    aw.show()
    sys.exit(qApp.exec_())

if __name__ == "__main__":
    mains()
