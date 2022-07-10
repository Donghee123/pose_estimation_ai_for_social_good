import os
import shutil
import random

strDataFolderPath = os.path.join('sample_image_folder', 'skeleton_npy', 'test')

file_destination = os.path.join('sample_image_folder', 'skeleton_npy', 'test', 'test')
     

dictOfLabes = {'good' : 0, 'left' : 1, 'right' : 2, 'turtleneck' : 3}
strDataFolderlist = os.listdir(strDataFolderPath)
        
X = []
Y = []

for strLabelFolderPath in strDataFolderlist:
    strOneLabelDataPath = os.path.join(strDataFolderPath, strLabelFolderPath)
    listOfOneLabelDataPath = os.listdir(strOneLabelDataPath)
    for strNPDataPath in listOfOneLabelDataPath:
        if random.random() <= 0.2:
            shutil.move(os.path.join(strOneLabelDataPath,strNPDataPath), os.path.join(file_destination,strLabelFolderPath))