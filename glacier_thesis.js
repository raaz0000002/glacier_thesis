// Load the study area (Dudh Koshi basin)
var studyArea = ee.FeatureCollection('projects/carbonstudybyraj/assets/dudhkoshi');
Map.centerObject(studyArea, 8);

// ===== SENTINEL-2 IMAGERY AND WATER DETECTION =====
// Load and preprocess Sentinel-2 imagery for 2024
var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(studyArea)
    .filterDate("2024-01-01", "2024-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
    .median()
    .clip(studyArea);

// Compute NDWI for water body detection
var NDWI = sentinel2.normalizedDifference(["B3", "B8"]).rename("NDWI");
var Lakes = NDWI.gt(0.3); // Threshold for water bodies
var LakeVector = Lakes.selfMask().reduceToVectors({
    geometry: studyArea.geometry(),
    geometryType: "Polygon",
    reducer: ee.Reducer.countEvery(),
    scale: 10
});

// ===== DEM AND TERRAIN ANALYSIS =====
// Load and clip DEM (SRTM)
var DEM = ee.Image("USGS/SRTMGL1_003").clip(studyArea);

// Compute terrain products
var terrain = ee.Terrain.products(DEM);
var slope = terrain.select("slope");
var aspect = terrain.select("aspect");

// ===== PRECIPITATION ANALYSIS =====
// Load CHIRPS precipitation data for 2024
var chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD");
var year = 2024;
var months = ee.List.sequence(1, 12);

// Compute monthly precipitation
var monthlyPrecip = months.map(function(month) {
    var startdate = ee.Date.fromYMD(year, month, 1);
    var enddate = startdate.advance(1, 'month');
    var monthlySum = chirps.filterDate(startdate, enddate).mean().clip(studyArea);
    return monthlySum.set({'month': month, 'system:time_start': startdate.millis()});
});
var monthlyPrecipCollection = ee.ImageCollection.fromImages(monthlyPrecip);

// Compute monthly statistics
var monthlyStats = monthlyPrecipCollection.map(function(image) {
    var meanPrecip = image.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: studyArea.geometry(),
        scale: 5000,
        bestEffort: true
    });
    return ee.Feature(null, {
        'month': ee.Number(image.get('month')),
        'mean_precipitation': ee.Number(meanPrecip.get('precipitation')).multiply(100) // Convert to mm
    });
});
var monthlyStatsCollection = ee.FeatureCollection(monthlyStats);

// Compute annual average precipitation
var annualAvgPrecip = monthlyPrecipCollection.mean().clip(studyArea).multiply(100);

// ===== TEMPERATURE ANALYSIS =====
// Load MODIS LST data for 2024
var modis = ee.ImageCollection("MODIS/061/MOD11A1")
    .filterDate("2024-01-01", "2024-12-31")
    .select('LST_Day_1km');
var modcel = modis.map(function(img) {
    return img.multiply(0.02).subtract(273.15).copyProperties(img, ['system:time_start']);
});
var meanLST = modcel.mean().clip(studyArea);

// Prepare LST time series
var lstTimeSeries = modcel.map(function(img) {
    var meanLST = img.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: studyArea.geometry(),
        scale: 1000
    });
    return ee.Feature(null, {
        'date': img.date().format('YYYY-MM-dd'),
        'mean_LST_Celsius': meanLST.get('LST_Day_1km')
    });
});

// ===== GLACIER AND SNOW ANALYSIS =====
// Load GLIMS glacier data
var glaciers = ee.FeatureCollection("GLIMS/20230607").filterBounds(studyArea);

// Estimate snowline and thickness
var snowlineElevation = 3000; // Adjust based on your study area
var snowlineMask = DEM.gte(snowlineElevation).clip(studyArea);
var glacierSlope = ee.Terrain.slope(DEM);
var estimatedThickness = glacierSlope.multiply(snowlineElevation).divide(100).updateMask(snowlineMask);
var velocityFactor = 0.02;
var glacierVelocity = estimatedThickness.multiply(velocityFactor);

// ===== ROCKFALL CLASSIFICATION =====
// Load Sentinel-2 for 2023
var sentinel2023 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(studyArea)
    .filterDate("2023-01-01", "2023-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
    .median()
    .clip(studyArea);
var bands = ["B2", "B3", "B4", "B8", "B11", "B12"];
var imageRockfall = sentinel2023.select(bands);

// Define training data for rockfall
var rockfall = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([86.92, 27.89]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.95, 27.92]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.8542, 27.9748]), {class: 1})
]);
var stableGround = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([86.85, 27.85]), {class: 0}),
    ee.Feature(ee.Geometry.Point([86.88, 27.87]), {class: 0})
]);
var trainingRockfall = imageRockfall.sampleRegions({
    collection: rockfall.merge(stableGround),
    properties: ["class"],
    scale: 10
});
var classifierRockfall = ee.Classifier.smileRandomForest(50).train({
    features: trainingRockfall,
    classProperty: "class",
    inputProperties: bands
});
var classifiedRockfall = imageRockfall.classify(classifierRockfall);

// ===== GLOF RISK CLASSIFICATION =====
// Define training data for GLOF risk
var dangerousAreas = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([86.6255, 27.7752]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9233, 27.9000]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.8460, 27.7388]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.8613, 27.6869]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9135, 27.7950]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9435, 27.8053]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9377, 27.8362]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9658, 27.7987]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9775, 27.8050]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9537, 27.7810]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.9555, 27.7533]), {class: 1}),
    ee.Feature(ee.Geometry.Point([86.6102, 27.8735]), {class: 1})
]);
var stableAreas = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([86.5000, 27.7000]), {class: 0}),
    ee.Feature(ee.Geometry.Point([86.5500, 27.7500]), {class: 0}),
    ee.Feature(ee.Geometry.Point([86.6000, 27.8000]), {class: 0}),
    ee.Feature(ee.Geometry.Point([86.6500, 27.8500]), {class: 0})
]);
var trainingGLOF = imageRockfall.sampleRegions({
    collection: dangerousAreas.merge(stableAreas),
    properties: ["class"],
    scale: 10
});
var classifierGLOF = ee.Classifier.smileRandomForest(50).train({
    features: trainingGLOF,
    classProperty: "class",
    inputProperties: bands
});
var classifiedGLOF = imageRockfall.classify(classifierGLOF);

// ===== VISUALIZATION =====
// Define visualization parameters
var visParams = {
    sentinelRGB: {bands: ["B4", "B3", "B2"], min: 0, max: 3000},
    ndwi: {min: -1, max: 1, palette: ["blue", "white", "green"]},
    lakes: {palette: ["blue"]},
    dem: {min: 50, max: 4500, palette: ["blue", "green", "yellow", "brown", "white"]},
    slope: {min: 0, max: 60, palette: ["green", "yellow", "orange", "red"]},
    aspect: {min: 0, max: 360, palette: ["red", "yellow", "green", "cyan", "blue", "purple"]},
    precip: {min: 0, max: 500, palette: ['001137', '0db39e', 'e7eb05', 'd00000']},
    lst: {min: 20, max: 50, palette: ['blue', 'green', 'yellow', 'red']},
    thickness: {min: 0, max: 500},
    velocity: {min: 0, max: 10, palette: ['blue', 'cyan', 'green', 'yellow', 'red']},
    classVis: {min: 0, max: 1, palette: ['green', 'red']}
};

// Add layers to the map (only once per feature)
Map.addLayer(studyArea, {}, 'Study Area');
Map.addLayer(sentinel2, visParams.sentinelRGB, "Sentinel-2 RGB");
Map.addLayer(NDWI, visParams.ndwi, "NDWI");
Map.addLayer(Lakes.updateMask(Lakes), visParams.lakes, "Lakes");
Map.addLayer(DEM, visParams.dem, "DEM");
Map.addLayer(slope, visParams.slope, "Slope");
Map.addLayer(aspect, visParams.aspect, "Aspect");
Map.addLayer(annualAvgPrecip, visParams.precip, "Average Precipitation (Yearly)");
Map.addLayer(meanLST, visParams.lst, "Mean LST (°C)");
Map.addLayer(glaciers.style({color: 'cyan', width: 1}), {}, "Glacier Area");
Map.addLayer(snowlineMask, {palette: ['blue', 'white']}, "Snowline Mask");
Map.addLayer(estimatedThickness, visParams.thickness, "Estimated Glacier Thickness");
Map.addLayer(glacierVelocity, visParams.velocity, "Computed Glacier Velocity");
Map.addLayer(classifiedRockfall, visParams.classVis, "Rockfall Classification");
Map.addLayer(classifiedGLOF, visParams.classVis, "Potential GLOF Risk Areas");

// ===== PRINT OUTPUTS =====
print('Monthly Precipitation Collection:', monthlyPrecipCollection);
print('Monthly Precipitation Stats:', monthlyStatsCollection);
print(ui.Chart.feature.byFeature(monthlyStatsCollection, 'month', ['mean_precipitation'])
    .setOptions({title: 'Monthly Precipitation in ' + year, hAxis: {title: 'Month'}, vAxis: {title: 'Precipitation (mm)'}}));
print(ui.Chart.image.series({
    imageCollection: modcel,
    region: studyArea.geometry(),
    reducer: ee.Reducer.mean(),
    scale: 1000,
    xProperty: 'system:time_start'
}).setOptions({title: 'LST Temporal Analysis (2024)', vAxis: {title: 'LST (°C)'}, hAxis: {title: 'Date'}}));
print("GLIMS Glacier Dataset:", glaciers);
print('Rockfall classification completed.');
print('GLOF risk classification completed.');

// ===== EXPORTS =====
// Export water bodies
Export.table.toDrive({
    collection: LakeVector,
    folder: "Dudhkoshi",
    description: "Lakes_Shapefile",
    fileFormat: "SHP"
});

// Export terrain data
Export.image.toDrive({image: DEM, description: "DEM", folder: "Dudhkoshi", scale: 10, region: studyArea.geometry(), fileFormat: "GeoTIFF", maxPixels: 1e13});
Export.image.toDrive({image: slope, description: "Slope", folder: "Dudhkoshi", scale: 10, region: studyArea.geometry(), fileFormat: "GeoTIFF", maxPixels: 1e13});
Export.image.toDrive({image: aspect, description: "Aspect", folder: "Dudhkoshi", scale: 10, region: studyArea.geometry(), fileFormat: "GeoTIFF", maxPixels: 1e13});

// Export precipitation data
Export.table.toDrive({collection: monthlyStatsCollection, description: 'Precipitation_' + year, fileFormat: 'CSV'});
Export.image.toDrive({image: annualAvgPrecip, description: 'AnnualPrecipitationMap_' + year, folder: 'Dudhkoshi', scale: 5000, region: studyArea.geometry(), fileFormat: "GeoTIFF"});

// Export temperature data
Export.image.toDrive({image: meanLST, description: 'Mean_LST_2024', folder: 'temperature', scale: 1000, region: studyArea.geometry(), fileFormat: "GeoTIFF", maxPixels:10e13});
Export.table.toDrive({collection: lstTimeSeries, description: 'LST_Time_Series_2024', folder: 'GoogleEarthEngineExports', fileFormat: 'CSV'});

// Export glacier data
Export.table.toDrive({collection: glaciers, description: 'Glacier_Area_Geometry', fileFormat: 'GeoJSON'});
Export.image.toDrive({image: estimatedThickness, description: 'Glacier_Thickness', folder: 'GlacierAnalysis', scale: 30, region: studyArea.geometry(), fileFormat: 'GeoTIFF', maxPixels: 1e13});
Export.image.toDrive({image: glacierVelocity, description: 'Glacier_Velocity', folder: 'GlacierAnalysis', scale: 30, region: studyArea.geometry(), fileFormat: 'GeoTIFF', maxPixels: 1e13});

// Export classification results
Export.image.toDrive({image: classifiedRockfall, description: "Rockfall_Classification", folder: "RockfallDetection", scale: 10, region: studyArea.geometry(), fileFormat: "GeoTIFF", maxPixels: 1e13});
Export.image.toDrive({image: classifiedGLOF, description: "GLOF_Risk_Classification", folder: "GLOF_Detection", scale: 10, region: studyArea.geometry(), fileFormat: "GeoTIFF", maxPixels: 1e13});
