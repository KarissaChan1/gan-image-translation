from utils import remove_slices, save_slices
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


rootpath = 'E:/Image Synthesis/Data/Registered_ALL/'

image_types = ['FA','MD','FLAIR']

names_df = pd.DataFrame()
id_df = pd.DataFrame()

# save filenames in dataframe for each image type
for t in image_types:
    holder = pd.DataFrame()

    file_list = os.listdir(rootpath+t)
    im_dict = sio.loadmat(rootpath+t+"/"+file_list[1])
    vol_name = list(im_dict.keys())[-1]
    vol = im_dict[vol_name]

    for f in file_list:
        holder = holder.append({t:f},ignore_index = True)

    names_df = pd.concat([names_df,holder],axis=1)
 
    del holder

patient_id = []
for x in range(len(names_df)):
    id = names_df['FA'][x][0:-8]
    patient_id.append(id)

names_df['Patient_ID']=patient_id


# split train test val samples
train,test = train_test_split(names_df,test_size = 0.2, random_state=42)
train,val = train_test_split(train,test_size=0.2, random_state=42)

print("Number of training vols: ",len(train))
print("Number of validation vols: ",len(val))
print("Number of test vols: ",len(test))

train=train.reset_index(drop=True)
val=val.reset_index(drop=True)
test = test.reset_index(drop=True)

# make directories to save samples
experiment = 'experiment4'
save_root = 'E:/Image Synthesis/experiments/' + experiment

train_dir = save_root + '/train/'
val_dir = save_root + '/val/'
test_dir = save_root + '/test/'
print(train_dir)
os.mkdir(save_root)
os.mkdir(train_dir)
os.mkdir(val_dir)
os.mkdir(test_dir)


# save slices

train_list = save_slices(train,train_dir)
df = pd.DataFrame(train_list)
df.to_excel(save_root+'/data_sampling.xlsx', sheet_name='train', index=False)
n_train = len(train_list)
print(n_train)

val_list = save_slices(val,val_dir)
print(len(val_list))

test_list = save_slices(test,test_dir)
print(len(test_list))

path = save_root+'/data_sampling.xlsx'
writer = pd.ExcelWriter(path, engine='openpyxl')
df2 = pd.DataFrame(val_list)
df3 = pd.DataFrame(test_list)
df2.to_excel(writer, sheet_name='val', index=False)
df3.to_excel(writer, sheet_name='test', index=False)

writer.save()
writer.close()








