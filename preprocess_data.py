#!/usr/bin/python
# -*- encoding: utf-8 -*-


labels_info = [
   {
       'name': 'unlabeled',
       'id':   0,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (0, 0, 0)
   },
   {
       'name': 'ego vehicle',
       'id':  1 ,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (0, 0, 0)
   },
   {
       'name': 'rectification border' ,
       'id':  2 ,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (0, 0, 0)
   },
   {
       'name': 'out of roi',
       'id':  3 ,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (0, 0, 0)
   },
   {
       'name': 'static',
       'id':  4 ,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (0, 0, 0)
   },
   {
       'name': 'dynamic',
       'id':  5 ,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (111, 74, 0)
   },
   {
       'name': 'ground',
       'id':  6 ,
       'trainId': 19 ,
       'category': 'void',
       'catId': 0,
       'hasInstance': False,
       'ignoreInEval': True,
       'color': (81, 0, 81)
   },
   {
       'name': 'road',
       'id':  7 ,
       'trainId':  0 ,
       'category': 'flat',
       'catId': 1,
       'hasInstance': False,
       'ignoreInEval': False ,
       'color': (128, 64, 128)
   },
   {  'name': 'sidewalk'             , 'id':  8 , 'trainId':  1 , 'category': 'flat', 'catId': 1, 'hasInstance': False, 'ignoreInEval': False , 'color': (244, 35, 232)},
   {  'name': 'parking'              , 'id':  9 , 'trainId': 19 , 'category': 'flat', 'catId': 1, 'hasInstance': False, 'ignoreInEval': True, 'color': (250, 170, 160)},
   {  'name': 'rail track'           , 'id': 10 , 'trainId': 19 , 'category': 'flat', 'catId': 1, 'hasInstance': False, 'ignoreInEval': True, 'color': (230, 150, 140)},
   {  'name': 'building'      , 'id': 11,  'trainId':   2,  'category': 'construction'  , 'catId':  2    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': ( 70, 70, 70)},
   {  'name': 'wall'          , 'id': 12,  'trainId':   3,  'category': 'construction'  , 'catId':  2    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (102,102,156)},
   {  'name': 'fence'         , 'id': 13,  'trainId':   4,  'category': 'construction'  , 'catId':  2    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (190,153,153)},
   {  'name': 'guard rail'    , 'id': 14,  'trainId':  19,  'category': 'construction'  , 'catId':  2    , 'hasInstance': False   , 'ignoreInEval': True    ,  'color': (180,165,180)},
   {  'name': 'bridge'        , 'id': 15,  'trainId':  19,  'category': 'construction'  , 'catId':  2    , 'hasInstance': False   , 'ignoreInEval': True    ,  'color': (150,100,100)},
   {  'name': 'tunnel'        , 'id': 16,  'trainId':  19,  'category': 'construction'  , 'catId':  2    , 'hasInstance': False   , 'ignoreInEval': True    ,  'color': (150,120, 90)},
   {  'name': 'pole'          , 'id': 17,  'trainId':   5,  'category': 'object'        , 'catId':  3    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (153,153,153)},
   {  'name': 'polegroup'     , 'id': 18,  'trainId':  19,  'category': 'object'        , 'catId':  3    , 'hasInstance': False   , 'ignoreInEval': True    ,  'color': (153,153,153)},
   {  'name': 'traffic light' , 'id': 19,  'trainId':   6,  'category': 'object'        , 'catId':  3    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (250,170, 30)},
   {  'name': 'traffic sign'  , 'id': 20,  'trainId':   7,  'category': 'object'        , 'catId':  3    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (220,220,  0)},
   {  'name': 'vegetation'    , 'id': 21,  'trainId':   8 , 'category': 'nature'        , 'catId':  4    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (107,142, 35)},
   {  'name': 'terrain'       , 'id': 22,  'trainId':   9 , 'category': 'nature'        , 'catId':  4    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': (152,251,152)},
   {  'name': 'sky'           , 'id': 23,  'trainId':  10 , 'category': 'sky'           , 'catId':  5    , 'hasInstance': False   , 'ignoreInEval': False   ,  'color': ( 70,130,180)},
   {  'name': 'person'        , 'id': 24,  'trainId':  11 , 'category': 'human'         , 'catId':  6    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (220, 20, 60)},
   {  'name': 'rider'         , 'id': 25,  'trainId':  12 , 'category': 'human'         , 'catId':  6    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (255,  0,  0)},
   {  'name': 'car'           , 'id': 26,  'trainId':  13 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (  0,  0,142)},
   {  'name': 'truck'         , 'id': 27,  'trainId':  14 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (  0,  0, 70)},
   {  'name': 'bus'           , 'id': 28,  'trainId':  15 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (  0, 60,100)},
   {  'name': 'caravan'       , 'id': 29,  'trainId':  19 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': True    ,  'color': (  0,  0, 90)},
   {  'name': 'trailer'       , 'id': 30,  'trainId':  19 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': True    ,  'color': (  0,  0,110)},
   {  'name': 'train'         , 'id': 31,  'trainId':  16 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (  0, 80,100)},
   {  'name': 'motorcycle'    , 'id': 32,  'trainId':  17 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (  0,  0,230)},
   {  'name': 'bicycle'       , 'id': 33,  'trainId':  18 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': True    , 'ignoreInEval': False   ,  'color': (119, 11, 32)},
   {  'name': 'license plate' , 'id': -1,  'trainId':  -1 , 'category': 'vehicle'       , 'catId':  7    , 'hasInstance': False   , 'ignoreInEval': True    ,  'color':  (0,  0, 142)},
]
