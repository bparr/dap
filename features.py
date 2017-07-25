#!/usr/bin/env python3

"""
Enum of features. Currently just for 2016 dataset.
"""

from enum import Enum


# With enums, the order listed is the order used for iteration.
# TODO(bparr): 2016_09_penetrometer_robot_Large_Stalks.csv has two lines for
#              Rw22 Ra32 which seem to describe completely different plants. So
#              ignoring.
# TODO(bparr): Reconsider using these row94 files?
#      - 2016_07_13_leaf_segmentation_leaf_fill_row94.csv
#      - 2016_09_penetrometer_manual_Row_94.csv
class Features(Enum):
  ROW = 'row'
  COLUMN = 'range'
  PLANT_ID = 'plant_id'

  # Harvest data.
  HARVEST_SF16h_HGT1_120 = 'SF16h_HGT1_120'
  HARVEST_SF16h_HGT2_120 = 'SF16h_HGT2_120'
  HARVEST_SF16h_HGT3_120 = 'SF16h_HGT3_120'
  HARVEST_SF16h_TWT_120 = 'SF16h_TWT_120'
  HARVEST_SF16h_WTP_120 = 'SF16h_WTP_120'
  HARVEST_SF16h_WTL_120 = 'SF16h_WTL_120'
  HARVEST_SF16h_PAN1_120 = 'SF16h_PAN1_120'
  HARVEST_SF16h_PAN2_120 = 'SF16h_PAN2_120'
  HARVEST_SF16h_PAN3_120 = 'SF16h_PAN3_120'

  # Composition data.
  COMPOSITION_ADF = 'ADF'
  COMPOSITION_AD_ICP = 'AD_ICP'
  COMPOSITION_ADJ_CP = 'Adj_CP'
  COMPOSITION_ANDFOM = 'aNDFom'
  COMPOSITION_ASH = 'Ash'
  COMPOSITION_CRUDE_PROTEIN = 'Crude_protein'
  COMPOSITION_DCAD = 'DCAD'
  COMPOSITION_DRY_MATTER = 'Dry_Matter'
  COMPOSITION_EE_FAT = 'EE_Fat'
  COMPOSITION_LIGNIN = 'Lignin'
  COMPOSITION_NEG_OARDC = 'NEG_OARDC'
  COMPOSITION_NEL3X_ADF = 'NEL3x_ADF'
  COMPOSITION_NEL3X_OARDC = 'NEL3x_OARDC'
  COMPOSITION_NEM_OARDC = 'NEM_OARDC'
  COMPOSITION_NFC = 'NFC'
  COMPOSITION_SPCP = 'SPCP'
  COMPOSITION_STARCH = 'Starch'
  COMPOSITION_TDN_OARDC = 'TDN_OARDC'
  COMPOSITION_WSC_SUGAR = 'WSC_Sugar'
  COMPOSITION_CELLULOSE = 'Cellulose'
  COMPOSITION_HEMICELLULOSE = 'Hemicellulose'

  # Robot data.
  ROBOT_LEAF_NECROSIS_07 = '2016_07_13-14_Leaf_Necrosis'
  ROBOT_VEGETATION_INDEX_07 = '2016_07_13-14_vegetation_index'
  ROBOT_VEGETATION_INDEX_08 = '2016_08_05-08_vegetation_index'
  ROBOT_LEAF_AREA_07 = '2016_07_13_BAP_Leaf_Area'
  ROBOT_LASER_PLANT_HEIGHT_07 = '2016_07_13_laser_plant_height'
  ROBOT_LIGHT_INTERCEPTION_07 = '2016_07_light_interception'
  ROBOT_LIGHT_INTERCEPTION_08 = '2016_08_light_interception'
  ROBOT_LIGHT_INTERCEPTION_09 = '2016_09_light_interception'

  # Synthetically created data.
  SYNTHETIC_HARVEST_SF16h_HGT_120_MEAN = 'SF16h_HGT_120_MEAN'
  SYNTHETIC_HARVEST_SF16h_HGT_120_STD = 'SF16h_HGT_120_STD'
  SYNTHETIC_HARVEST_SF16h_PAN_120_MEAN = 'SF16h_PAN_120_MEAN'
  SYNTHETIC_HARVEST_SF16h_PAN_120_STD = 'SF16h_PAN_120_STD'

  # GPS location, in UTM format.
  GPS_EASTINGS = 'gps_eastings_UTMzone17N'
  GPS_NORTHINGS = 'gps_northings_UTMzone17N'

  # Accessions Data.
  ACCESSION_PHOTOPERIOD = 'accession_photoperiod'
  ACCESSION_TYPE = 'accession_type'
  ACCESSION_ORIGIN = 'accession_origin'
  ACCESSION_RACE = 'accession_race'

  # Other lower priority data.
  PLOT_ID = 'plot_id'
  NOTES = 'Notes'
  X_OF_Y = 'x_of_y'
  PLOT_PLAN_TAG = 'plot_plan_tag'
  PLOT_PLAN_CON = 'plot_plan_con'
  PLOT_PLAN_BARCODE = 'plot_plan_barcode'
  PLOT_PLAN_END = 'plot_plan_end'

