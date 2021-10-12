from zipfile import ZipFile
from glob import glob
import os, re
import shutil
from tqdm import tqdm
import pandas as pd
pd.set_option('chained_assignment', None)


if __name__ == '__main__':
    
    ## Note: all of this processing was done one google collab so all the paths here are with respect to google collab

    #### To copy only relevant file from dataset ####

    # Create a empty CSV file to store our file paths and aneursym location
    dataset = pd.DataFrame(columns=['tof_file', 'aneurysm_file', 'aneurysm', 'location'])
    for files in tqdm(glob('/content/drive/MyDrive/ADAM-Dataset-Full/*.zip')):
        name = files.split('.')[0].split('/')[-1]
        with ZipFile(files, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall('temp')
        og_file = f'./temp/{name}/orig/TOF.nii.gz'
        aneuryms_file = f'./temp/{name}/aneurysms.nii.gz'

        flag = 0

        with open(f'./temp/{name}/location.txt', 'r') as f:
            if f.read() != '':
                flag = 1
            location = f.readlines()
        if not os.path.exists('/content/drive/MyDrive/ADAM-Filtered-Data'):
            os.makedirs('/content/drive/MyDrive/ADAM-Filtered-Data')
        dataset.loc[dataset.shape[0]] = [f'/content/drive/MyDrive/ADAM-Filtered-Data/{name}_TOF.nii.gz', f'/content/drive/MyDrive/ADAM-Filtered-Data/{name}_aneurysms.nii.gz', flag, location]
        
        # Copy the selected files fo the new folder
        shutil.copyfile(src=og_file, dst=f'/content/drive/MyDrive/ADAM-Filtered-Data/{name}_TOF.nii.gz')
        shutil.copyfile(src=aneuryms_file, dst=f'/content/drive/MyDrive/ADAM-Filtered-Data/{name}_aneurysms.nii.gz')

    # To shuffle dataset randomly before splitting
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    if not os.path.exists('../dataset/'):
        os.mkdir('../dataset/')

    dataset.to_csv('../dataset/dataset.csv', index=False)
    train_df = dataset.iloc[:int(dataset.shape[0] * 0.8)]
    val_df = dataset.iloc[int(dataset.shape[0] * 0.8):]

    train_df.to_csv('../dataset/train_set.csv', index=False)
    val_df.to_csv('../dataset/val_set.csv', index=False)

    print("Files Extracted Successfully")