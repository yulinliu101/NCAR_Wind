# Download NCAR_WIND
1. Go to this link: http://rda.ucar.edu/datasets/ds090.0/

2. Click on the DATA ACCESS tab (second tab from left)

3. Click the "Internet Download Files" link.

4. Click "Dynamic List" link.

5. In Temporal (Valid Time) Selection, enter the date range for which you want the wind files.

6. Select the checkboxes: u-component of wind, v-component of wind, and Pressure

In the Subset Selection combo box, select "2.5 degree analyses on 17 pressure

levels, plus other surfaces and layers (pgb.f00)

7. Click the "Create List" button. You get a list of files in GRIB 1 format that can be downloaded either individually (one for each month) or collectively as a tar file

8. (if date range spans multiple months). Individual file names will look like: A23202-0801.pgb.f00, where 0801 is January 2008.


# NCAR_Wind
Decode NCAR wind data and match with trajectories

This module has been tested on both Linux and Windows machine. Please refer to the DemoWindMatching.ipynb for further instruction.

File structure:
The file structure is similar to the older version of wind processing file. The only difference is that in this module, you need to put the raw wind data in a folder "/raw".

In the Get_Wind_Matching.py file, only the function "DecondeWind" requires Linux environment, mainly because of the package "pygrib". However, this function will only be called once at the beginning of all the other runs. That is to say, after we download the raw wind data for a whole year and put them in the folder '/raw', you only need to call this function once to extract the wind data of the US for the whole year. The extracted data will be stored as numpy array and be dumped to another folder in a zip file. This process will take roughly 10 minutes. After extracting and dumping the files, you don't have to stick to the Linux machine, but just to copy the zip file to the windows envirnment for further matching purpose. (But if you want to keep Linux env, you can keep going with the jupyter notebook file). The script has been tested on both platform with py35 and py27.

The final step of the module is to merge the matching results with the MNL file. In the demo, I merged the matching wind speed and wind distance with the "MA_MNL_DEPARRYEAR.csv" for convinience and comparison purpose. However, if you do not want to merge with MA_MNL_DEPARRYEAR.CSV, you can change the source code in the file "Get_Wind_Matching.py" at line 324.

With the new MNL file (you should set Overwrite = True), you should use the new "GetProcessMNL_LR.py" (uploaded here) to run the choice model and linear model.
