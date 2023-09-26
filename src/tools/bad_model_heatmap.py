import shutil
from os.path import join,exists
from os import makedirs

bad_baseline_file=open('/home/unav/Desktop/Resolution_Agnostic_VPR/log/fail_baseline.txt','r')
bad_pipeline_file=open('/home/unav/Desktop/Resolution_Agnostic_VPR/log/fail_pipeline.txt','r')
model_names=['Baseline','Triplet','MSE','MSE+ICKD']

# Initialize empty dictionaries for the data
data = {}
def data_extract(lines,data):
    for line in lines:
        line = line.strip()  # Remove any leading or trailing whitespace
        model,path=line.split("_",1)
        if model not in data:
            data[model]=[path]
        else:
            data[model].append(path)
    return data

for k,v in data.items():
    data[k]=set(v)

data=data_extract(bad_baseline_file,data)
data=data_extract(bad_pipeline_file,data)

sample_model='Baseline'
example_files=data[sample_model]
save_path='/home/unav/Desktop/Resolution_Agnostic_VPR/log/bad_model'
heat_path='/mnt/data/Resolution_Agnostic_VPR/logs/heatmap/unav/Lighthouse6_6'

for image in example_files:
    if (image not in data['MSE+ICKD']) and (image in data['Triplet']) and (image in data['MSE']):
        key_list=image.split('/')
        pitch=key_list[7]
        yaw=key_list[8]
        resolution=key_list[10]
        qp=key_list[11]
        image_id=key_list[-1]
        save_folder=join(save_path,resolution,qp,pitch,yaw,image_id.replace('.png',''))
        if not exists(save_folder):
            makedirs(save_folder)
        shutil.copy(image,save_folder)
        for model in model_names:
            heatmap_path=join(heat_path,model,pitch,yaw,resolution,qp,image_id)
            if not exists(heatmap_path):
                shutil.rmtree(save_folder)
                break
            output_path=join(save_folder,model+'_'+image_id)
            shutil.copy(heatmap_path,output_path)    




