# 8 cores 64 GB
import numpy as np
import xarray as xr
import rasterio.features
import stackstac
import pystac_client
import planetary_computer
import xrspatial.multispectral as ms
from dask_gateway import GatewayCluster
import time
import pickle
from osgeo import gdal,ogr,osr

# Load the SVM model
model = pickle.load(open("classification.pkl", 'rb'))
def predict(array,model=model):
    try:
        np_array = array
        im_lines = None
        for i in range(np_array.shape[0]):
            im_line = np_array[i,:,:].flatten()
            if im_lines is None:
                im_lines = im_line
            else:
                im_lines = np.vstack((im_lines, im_line))
                print(im_line.shape)
                print(im_lines.shape)
        im_lines = im_lines.transpose((1, 0))
        output_array = model.predict(im_lines).reshape(np_array.shape[1], np_array.shape[2]).astype("int16")
        res = xr.DataArray(output_array,dims=["x", "y"])
        return res
    except Exception:
        return np.zeros(shape=array[0].shape)

# Define the fucntion of a single test
def test(chunk_size):
    try:
        # Create a dask cluster
        start_time = time.time()
        cluster = GatewayCluster()
        client = cluster.get_client()
        cluster.scale(24)
        print(cluster.dashboard_link)

        
        # search data from stac
        x1 = 116
        x2 = 117
        y1 = 40.5
        y2 = 39.5
        area_of_interest = {
            "type": "Polygon",
            "coordinates": [
                [
                  [x1,y1],
                  [x1,y2],
                  [x2,y2],
                  [x2,y1],
                  [x1,y1]
                ]
            ],
        }
        bbox = rasterio.features.bounds(area_of_interest)
        stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = stac.search(
            bbox=bbox,
            datetime="2020-05-01/2020-06-30",
#             datetime="2020-06-01/2020-07-30",
            collections=["sentinel-2-l2a"],
            limit=500,  # fetch items in batches of 500
            query={"eo:cloud_cover": {"lt": 10}},
        )

        items = list(search.get_items())
        signed_items = [planetary_computer.sign(item).to_dict() for item in items]
        print(len(items))
        
        # Create datasets
        data = (
            stackstac.stack(
                signed_items,
                assets=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
                epsg=4326,
                resolution=0.00016,
                resampling=rasterio.enums.Resampling.bilinear
            )
            .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
            .assign_coords(band=lambda x: x.common_name.rename("band"))  # use common names
        )
        
        # Compute the key parametre to cut the ROI
        xoff = round((x1 - data.transform[2])/data.transform[0])
        yoff = round((y1 - data.transform[5])/data.transform[4])
        x_pixel = round((x2-x1)/data.transform[0])
        y_pixel = round((y2-y1)/data.transform[4])

        gt_output = [data.transform[2] + xoff * data.transform[0],data.transform[0],0,
                    data.transform[5] + yoff * data.transform[4],0 , data.transform[4]]

        data_roi = data[:,
                    :,
                    yoff:yoff+y_pixel,
                    xoff:xoff+x_pixel,
                   ].chunk(chunk_size)
        
        # Take the max value
        median = data_roi.max(dim="time")
        
        divide_time = time.time()
        
        # Execute
        run = median.data.map_blocks(predict, drop_axis=0)
        res = run.compute()
        
        # Write the result into file
        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create("dask.tif", res.shape[1], res.shape[0], 1, 2, ["BIGTIFF=YES"])
        dst.GetRasterBand(1).WriteArray(res)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dst.SetProjection(srs.ExportToWkt())
        dst.SetGeoTransform(gt_output)
        del dst

        end_time = time.time()
        cluster.close()

        divide_cost = divide_time - start_time
        process_cost = end_time - divide_time
        total_cost = end_time - start_time
        print(chunk_size, divide_cost, process_cost, total_cost)
        return [chunk_size, divide_cost, process_cost, total_cost]
    except Exception as e:
        cluster.close()
        print(e)
        return None

# Execute in loop
res = []
for chunk_size in [4096, 2048, 1024, 512, 256, 128, 64]:
    for i in range(4):
        res.append(test(chunk_size))
