// import the training points
var train_data = ee.FeatureCollection("users/blackmhw/Guangzhou/sample_training"),
    validate_data = ee.FeatureCollection("users/blackmhw/Guangzhou/sample_validate");

// // Extract sample data
// Mask Sentinel2-L2A data with cloud 
function maskS2clouds(image) {
  var qa = image.select('QA60')

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

  // Return the masked and scaled data, without the QA bands.
  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"])
}

// Map the function over one year of data and take the minium.
// Load Sentinel-2 TOA reflectance data.
var collection = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2019-05-01', '2020-01-01')
    // Pre-filter to get less cloudy granules.
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .map(maskS2clouds)

var composite = collection.median()

// Use these bands for classification.
var bands = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"];

// The name of the property on the points storing the class label.
var classProperty = 'land_use';

// Sample the composite to generate training adn validating data.
var training = composite.select(bands).sampleRegions({
  collection: train_data,
  properties: [classProperty],
  scale: 10
});
var validating = composite.select(bands).sampleRegions({
  collection: validate_data,
  properties: [classProperty],
  scale: 10
});


// Train a SVM classifier.
var classifier = ee.Classifier.libsvm().train({
  features: training,
  classProperty: classProperty,
});


// Classify the test FeatureCollection.
var test = validating.classify(classifier);


// Print the confusion matrix.
var confusionMatrix = test.errorMatrix(classProperty, 'classification');
print('Confusion Matrix', confusionMatrix);


// // Classify the composite.
// Get Collection
var collection_predict = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2020-05-01', '2020-06-30')
    // Pre-filter to get less cloudy granules.
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .map(maskS2clouds)

// Predict
var composite_predict = collection_predict.max()
var classified = composite_predict.classify(classifier);

// study area Beijing, China
var geometry = ee.Geometry.Polygon({
coords: [
            [116,39.5],
            [117,39.5],
            [117,40.5],
            [116,40.5],
            [116,39.5]
        ],
});

// Execute and output
Export.image.toDrive({
  image: classified,
  description: "gee",
  scale: 20,
  crs:"EPSG:4326",
  region: geometry,
  maxPixels: 10000000000000,
  shardSize:1024
  })

// loop for different chunk size
var chunk_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
for (var chunk_size in chunk_sizes){
  for (var i in [0,1,2,3]){
    print(chunk_sizes[chunk_size], i)
    Export.image.toDrive({
      image: classified,
      description: (chunk_sizes[chunk_size] + '_' + i),
      scale: 20,
      crs:"EPSG:4326",
      region: geometry,
      maxPixels: 10000000000000,
      shardSize:chunk_sizes[chunk_size]
      })
  }
}
