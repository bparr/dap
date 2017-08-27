Ben Parr (bparr) Data Analysis Project (https://www.ml.cmu.edu/current-students/Data%20Analysis%20Project.html).


Run tests: &nbsp; `find -name \*_test.py -exec {} \;`

Reproduce random forest results: &nbsp; `./predict.py --rf_only`

The predict script can write the fully processed and ready-for-regression datasets to the dataviews/ directory. This is useful for using the AutonLab's random forest implementation, for example. To do this, just run: &nbsp; `./predict.py --write_dataviews_only`

Use `./predict.py --help` for more details on the different supported flags. For example, the script currently supports five different datasets. The 2014 dataset. The full 2016 dataset. The 2016 dataset without the harvest phenotypes (2016.noHarvest). And the 2016 dataset without the harvest phenotypes or the robot/aerial data (2016.noRobotAerialHarvest).

Regenerate merged dataset: &nbsp; `./merge_data.py && ./merge_data_test.py`


Directory Descriptions:
  * 2014/ contains the original 2014 dataset.
  * 2016/ contains the original 2016 data files, saved from Excel as .csv files.
  * dataviews/ contains the outputs from using the `--write_dataviews_only` flag (see above, or `./predict.py --help`).
  * predictions/ contains the predicted values for each regressor.
  * reports/ contains PDFs, and images used in reports.
  * results/ contains the outputs of `./predict.py` such as r^2 scores, and feature importance figures.
  * spatial/ contains the output of the spatial correlation work using Mantel tests.
