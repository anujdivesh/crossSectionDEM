import geopandas as gpd
import numpy as np
import pandas as pd 
import rasterio
import matplotlib.pyplot as plt 
from osgeo import gdal
import rioxarray
from pyproj import Transformer

#start_coords = [176.309881,-6.283988]
#end_coords = [ 176.331339,-6.281685]

start_coords = [176.316061,-6.281002]
end_coords = [  176.325502,-6.285183]

lon = [start_coords[0]]
lat = [start_coords[1]]

n_points = 500


for i in np.arange(1, n_points+1):
    x_dist = end_coords[0] - start_coords[0]
    y_dist = end_coords[1] - start_coords[1]
    point  = [(start_coords[0] + (x_dist/(n_points+1))*i), (start_coords[1] + (y_dist/(n_points+1))*i)]
    lon.append(point[0])
    lat.append(point[1])
    
lon.append(end_coords[0])
lat.append(end_coords[1])



df = pd.DataFrame({'Latitude': lat, 
                    'Longitude': lon})

gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))

gdf = gdf.set_crs('epsg:4326')
gdf_pcs = gdf.to_crs(epsg = 3857)

gdf_pcs['h_distance'] = 0

for index, row in gdf_pcs.iterrows():
    gdf_pcs['h_distance'].loc[index] = gdf_pcs.geometry[0].distance(gdf_pcs.geometry[index])

#Extracting DEM Elevation
gdf_pcs['Elevation'] = 0
rds = rioxarray.open_rasterio("Nanumaga_MB_Lidar.tif")
transformer = Transformer.from_crs("EPSG:4326", rds.rio.crs, always_xy=True)

for index, row in gdf_pcs.iterrows():
    lon = row['Longitude']
    lat = row['Latitude']

    xx, yy = transformer.transform(lon, lat)
    value = rds.sel(x=xx, y=yy, method="nearest").values
    gdf_pcs['Elevation'].loc[index] = value

x_y_data = gdf_pcs[['h_distance', 'Elevation']]
x_y_data.plot(x='h_distance',y='Elevation')

min_valu = x_y_data['Elevation'].min()
plt.fill_between(x_y_data['h_distance'],x_y_data['Elevation'],min_valu,alpha=0.4)
plt.show()

