import os

# Lets change working directory to the pictures folder
os.chdir(r"P:\Projects\Office of Community Development\Working Files\Sensitivity Testing Oct2021\ST4_Gap_Analysis\NonTropicalStorms")

# confirm working directory by printing it out
print (os.getcwd())

# loop over the files in the working directory and printing them out
for dir in os.listdir(os.getcwd()):
#  print (dir)
 os.rename(dir, 'NTS ' + dir)