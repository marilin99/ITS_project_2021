import overpy
import pandas as pd
import numpy as np

from shapely.geometry import Polygon
from shapely.ops import polygonize
from shapely.geometry import LineString
from shapely.geometry import Point

import geopandas as gpd
import pyproj





def nodelist_to_edges(list_of_nodes):
    '''
    Takes a list of nodes in a way and converts them into node pairs (including ID and coord. of each node), imitating an edge
    
    In the following example, nodes [11228194, 8223498, 322959398] turn into
    [[11228194, 8223498], [8223498, 322959398]]
    
    Example input: 
    [<overpy.Node id=11228194 lat=58.4218993 lon=26.5053693>, 
     <overpy.Node id=8223498 lat=58.4217486 lon=26.5060318>, 
     <overpy.Node id=322959398 lat=58.4215384 lon=26.5069523>]
     
    Example output:
    [[[11228194, Decimal('26.5053693'), Decimal('58.4218993')], [8223498, Decimal('26.5060318'), Decimal('58.4217486')]],
    [[8223498, Decimal('26.5060318'), Decimal('58.4217486')], [322959398, Decimal('26.5069523'), Decimal('58.4215384')]]]
    '''
    
    x = list_of_nodes
    index_pairs = [0, len(x) - 1]
    index_pairs[1:1] = np.repeat(range(1, len(x) - 1), 2)
    index_pairs = list(zip(index_pairs[::2], index_pairs[1::2]))
    return [[[x[a].id, x[a].lon, x[a].lat], [x[b].id, x[b].lon, x[b].lat]] for a, b in index_pairs]


def make_line(row):
    '''
    Converts two coordinates into a LineString
    TODO: Fix warning "Convert the '.coords' to a numpy array instead."
    '''
    coordinates = np.asarray([(row["A_lon"], row["A_lat"]), (row["B_lon"], row["B_lat"])])
    return LineString(coordinates)



api = overpy.Overpass()
query_input = [58.3405, 26.6445, 58.4046, 26.8481]

result = api.query("""
[out:json][timeout:25];
(
    way["highway"](58.3405, 26.6445, 58.4046, 26.8481);
    relation["highway"](58.3405, 26.6445, 58.4046, 26.8481);
);
(._;>;);
out;
""")


### FOR QUERYING OTHER CITIES

#result = api.query("""
#    [out:json][timeout:25];
#    area[name = "Tartu linn"];
#    (
#      way(area)["highway"];
#      relation(area)["highway"];
#    );
#    (._;>;);
#    out;
#    """)

#min_lat = float(min(result.nodes, key=lambda x: x.lat).lat)
#max_lat = float(max(result.nodes, key=lambda x: x.lat).lat)
#min_lon = float(min(result.nodes, key=lambda x: x.lon).lon)
#max_lon = float(max(result.nodes, key=lambda x: x.lon).lon)
#bbox = [min_lon, min_lat, max_lon, max_lat]
#bbox

node_cols = ["id", "lon", "lat"]
df_nodes = pd.DataFrame([[getattr(node, att) for att in node_cols] for node in result.nodes])\
    .rename(columns = dict(zip(range(0, 3), node_cols)))

# Convert coordinates into whatever that is not Decimal()
df_nodes[["lon", "lat"]] = df_nodes[["lon", "lat"]].apply(pd.to_numeric)

way_cols = ["roadID", "tag", "nodelist"]

# Get all edges alongside the two nodes they consist of
df_ways = pd.DataFrame([[way.id, way.tags.get("highway"), nodelist_to_edges(way.nodes)] for way in result.ways])\
    .rename(columns = dict(zip(range(0, 3), way_cols))).explode("nodelist").reset_index(drop = True)

# Separate start and end point of each edge into new column A and B
df_ways[["A","B"]] = pd.DataFrame(df_ways["nodelist"].tolist())
df_ways = df_ways.drop(columns = ["nodelist"])

# Separate the ID and longitude-latitude coordinates of each _start_ point into new columns
df_ways[["A_id", "A_lon", "A_lat"]] = pd.DataFrame(df_ways["A"].tolist())
df_ways = df_ways.drop(columns = ["A"])

# Separate the ID and longitude-latitude coordinates of each _end_ point into new columns
df_ways[["B_id", "B_lon", "B_lat"]] = pd.DataFrame(df_ways["B"].tolist())
df_ways = df_ways.drop(columns = ["B"])

# Convert coordinates into whatever that is not Decimal()
df_ways[["A_lon", "A_lat", "B_lon", "B_lat"]] = df_ways[["A_lon", "A_lat", "B_lon", "B_lat"]].apply(pd.to_numeric)

# Calculate distances (weights) of each edge
df_ways["distance"] = np.linalg.norm(df_ways[["A_lon", "A_lat"]].values - df_ways[["B_lon", "B_lat"]].values, axis = 1)

# Get separate ID for each edge
df_ways = df_ways.reset_index().rename(columns={'index': 'edgeID'})

# Convert coordinates into linestrings under geometry column
df_ways["geometry"] = df_ways.apply(make_line, axis=1)

# All roads into GeoDF
proj_epsg_4326 = pyproj.CRS.from_string("epsg:4326")
edges_gpd = gpd.GeoDataFrame(geometry=df_ways["geometry"], crs=proj_epsg_4326)


edges_gpd.to_file("data/Tartu_edges_gpd.shp")
df_ways.to_csv('data/df_ways.csv', index=False)
df_nodes.to_csv('data/df_nodes.csv', index=False)