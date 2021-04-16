# Written by Bingdian, Edited by Mok

# imports

import pathlib
import os
import sys

print("\nscanning data folder... \n\n--------------------\n")

# prints out all the files in the data folder

for path in pathlib.Path("data").iterdir():
    if path.is_file():       
        old_name = path.stem
        print(old_name)

print ("\n--------------------\n")

x = input("Confirm all your files are in the data folder?[y/n]: ")

count = 0

if x == "y":

    for path in pathlib.Path("data").iterdir():
        if path.is_file():       
            old_name = path.stem
            old_extension = path.suffix
            directory = path.parent

#------------- edit this part to change filename to whatever you need --------------- 

            set1 = old_name[3:11]
            set2 = old_name[21:29]
            set3 = old_name[11:15]

            new_name = f"PILOT [{set2}] [{set1}] [{set3}]" + old_extension 

#------------- end editable part ----------------------------------------------------

            path.rename(pathlib.Path(directory, new_name))
            
            count +=1

    sys.exit(f"\nCompleted! {count} file(s) renamed!")

else:
    sys.exit("\nOperation aborted!")
