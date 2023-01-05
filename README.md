# Repository to host code scripts for manuscript "Sentinel and Google Street View data predict household income in New York City, Greater Sydney, and Amsterdam"

This repository contains Google Earth Engine code scripts to calculate income prediction features from satellite imagery (links to the scripts are provided as follows), code to calculate income prediction features from Google Street View images (images are not provided as they are subject to copyright), TensorFlow image classification models for Google Street View scenes.

Folders 'Amsterdam model', 'New York City model', and 'Greater Sydney model' contain Jupyter notebooks for CatBoost income prediction models based on combined sets of features. To use remote sensing- or Google Street View-only sets of features, use .csv files with extensions 'rs' and 'gsv', respectively instead of defaults. Resulting .bin and .json CatBoost models are provided.


## Google Earth Engine scripts
### Amsterdam
Sentinel-1 mosaic: https://code.earthengine.google.com/d6eb2228f9aacab3886f4c2364c62c43?noload=true

Sentinel-2 mosaic: https://code.earthengine.google.com/16d850e348da586ad7b7d233f0c674a2?noload=true

Building classification: https://code.earthengine.google.com/77031078d399c359c23259c3d98029df?noload=true

Hue-Saturation-Value bands: https://code.earthengine.google.com/332d5f07e52e2a02aafda5b011d7964c?noload=true

Spectral indices: https://code.earthengine.google.com/16129a13beb14e2f5836efae41094341?noload=true

Tasseled Cap transformation: https://code.earthengine.google.com/5b225d2a83645788d41cfda61b42539e?noload=true

First-order entropy: https://code.earthengine.google.com/7a73e0b56383301acd7c5b9f33e3aae5?noload=true

Geary's C statistics: https://code.earthengine.google.com/1a0725e9a16d59bb40b6404f4af6a31c?noload=true

Sentinel-1 GLCM indices: https://code.earthengine.google.com/1cdb102e0b52360d54e08c96ea187b5b?noload=true

Colour hue GLCM indices: https://code.earthengine.google.com/5b084259beb88898a5bd5694a1c14613?noload=true

Colour saturation GLCM indices: https://code.earthengine.google.com/5d06ae36606c7642c5ab88e568bd57cc?noload=true

Colour value GLCM indices: https://code.earthengine.google.com/2409e1330425a9d0539b403b24956f8e?noload=true

Luminance GLCM indices: https://code.earthengine.google.com/70b070f087343480eb75b1afc97416c7?noload=true

NDBI GLCM indices: https://code.earthengine.google.com/0c02feb0428b1f19290fc1463b8cd1c5?noload=true

NDVI GLCM indices: https://code.earthengine.google.com/677baa3b0230fa0199e35d2cc48fa324?noload=true

NDWI GLCM indices: https://code.earthengine.google.com/7cf5774aa3e481c87d919652cdd00018?noload=true

Tasseled Cap Brightness GLCM indices: https://code.earthengine.google.com/7d19ebcf730d26059abbd89745fd030c?noload=true

Tasseled Cap Greenness GLCM indices: https://code.earthengine.google.com/c0d6f85b90a48c01f506f840ac0b9917?noload=true

Tasseled Cap Wetness GLCM indices: https://code.earthengine.google.com/97304f4298a7f591973fe76827df2cb5?noload=true

### New York City
Sentinel-1 mosaic: https://code.earthengine.google.com/99d32f9c7eee4fb1c6eb09ff5ff719ed?noload=true

Sentinel-2 mosaic: https://code.earthengine.google.com/6c942fc8b8164e5238ee3d6f24029ed6?noload=true

Building classification: https://code.earthengine.google.com/f750a139311644e64c5b678f7f163d18?noload=true

Hue-Saturation-Value bands: https://code.earthengine.google.com/786e24257f672e52ab99604e755584b3?noload=true

Spectral indices: https://code.earthengine.google.com/8312ecd9d665d392f226c31a8e6dc457?noload=true

Tasseled Cap transformation: https://code.earthengine.google.com/f4e4058b2d8dd0d00920ff07849d5aae?noload=true

First-order entropy: https://code.earthengine.google.com/7d72bfac1b7ec3120b31fcdf52801d78?noload=true

Geary's C statistics: https://code.earthengine.google.com/740bf70381af39196ecf3d499ddff610?noload=true

Sentinel-1 GLCM indices: https://code.earthengine.google.com/48ef487ce4660017d4388372b5add7c6?noload=true

Colour hue GLCM indices: https://code.earthengine.google.com/3fc9df7eba82fd12138e4e867402a236?noload=true

Colour saturation GLCM indices: https://code.earthengine.google.com/72fad06d29bdbf7e69f6b34e1303bd77?noload=true

Colour value GLCM indices: https://code.earthengine.google.com/14dffe8eadce04ae5a0fcfea09e1a39e?noload=true

Luminance GLCM indices: https://code.earthengine.google.com/1856e2f96ca9c2378f2dfa9cacdc2479?noload=true

NDBI GLCM indices: https://code.earthengine.google.com/50d21408f4d705454d67bdfe7856763e?noload=true

NDVI GLCM indices: https://code.earthengine.google.com/949f685ab01403d5968d140fbd0c05ca?noload=true

NDWI GLCM indices: https://code.earthengine.google.com/6c4d9fd7f428305503ef468e8fa3cd88?noload=true

Tasseled Cap Brightness GLCM indices: https://code.earthengine.google.com/55c0e7ab605b103c502cb983d1a6c0b8?noload=true

Tasseled Cap Greenness GLCM indices: https://code.earthengine.google.com/a81d6d68772345c692f6475b25b8f77f?noload=true

Tasseled Cap Wetness GLCM indices: https://code.earthengine.google.com/6ad26a2961ff6a49148698713ffaf447?noload=true


### Greater Sydney

Sentinel-1 mosaic: https://code.earthengine.google.com/64bdd04f2cea6865e1fc13159edc9ccd?noload=true

Sentinel-2 mosaic: https://code.earthengine.google.com/75a4bc73b585e6fd68eba1cc9d204501?noload=true

Building classification: https://code.earthengine.google.com/d2d9b6531cf55d3e0141f68a14da7a1d?noload=true

Hue-Saturation-Value bands: https://code.earthengine.google.com/c3a766818708155366ada7bb397db94f?noload=true

Spectral indices: https://code.earthengine.google.com/a84dbf44eb6c1a33012fb972459b399e?noload=true

Tasseled Cap transformation: https://code.earthengine.google.com/9337910be1845ca1a7c74733c5f9f211?noload=true

First-order entropy: https://code.earthengine.google.com/fb034bd1e9c38110b97b75304e7d21c2?noload=true

Geary's C statistics: https://code.earthengine.google.com/3ce0b84dd8a7b1193a5785b6bb2de51c?noload=true

Sentinel-1 GLCM indices: https://code.earthengine.google.com/99cae0c63f17f0ca35e538ac53c73d42?noload=true

Colour hue GLCM indices: https://code.earthengine.google.com/53d9c76e953064d9444a02bb0f890b30?noload=true

Colour saturation GLCM indices: https://code.earthengine.google.com/f878ebccab140d85be30021957e13677?noload=true

Colour value GLCM indices: https://code.earthengine.google.com/93b57d6e428c5189793c180982cdde65?noload=true

Luminance GLCM indices: https://code.earthengine.google.com/ca5998bc48bd77bc659cc5bd8c3202d1?noload=true

NDBI GLCM indices: https://code.earthengine.google.com/ed36242ef041b0aad884ac7b9519e5cb?noload=true

NDVI GLCM indices: https://code.earthengine.google.com/2007a3a3b7ea4f61bc78d2733d9caab4?noload=true

NDWI GLCM indices: https://code.earthengine.google.com/4c0ad8442d4ad3d5d90a789d7a8a8767?noload=true

Tasseled Cap Brightness GLCM indices: https://code.earthengine.google.com/ec2e1047b41d1ec606dcdded9492cef9?noload=true

Tasseled Cap Greenness GLCM indices: https://code.earthengine.google.com/554348e565ab204cf9c1a81c2d1fd4b3?noload=true

Tasseled Cap Wetness GLCM indices: https://code.earthengine.google.com/2f7ed24b07655d27e6815002f384a09b?noload=true

## Credits and references

Building classification code scripts were adapted from Tassi A., Vizzari M., 2020, “Object-oriented LULC classification in Google Earth Engine combining SNIC, GLCM, and Machine Learning algorithms”, Remote Sens. 2020, 12(22), 3776; https://doi.org/10.3390/rs12223776

We used IVPY tool for extraction of colour and morphological features from Google Street View images by David Crockett (2019). Ivpy: Iconographic visualization inside computational notebooks. International Journal for Digital Art History, (4), 3-60; https://journals.ub.uni-heidelberg.de/index.php/dah/article/download/66401/73328
GitHub page: https://github.com/damoncrockett/ivpy

Code for income modelling was adopted from CatBoost notebooks here: https://github.com/catboost

Google Street View data were downloaded using google_streetview tool for Google Street View Image API by Richard Wen: https://rrwen.github.io/google_streetview/index.html
