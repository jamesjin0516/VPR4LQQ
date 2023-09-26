from os import makedirs
from os.path import join,exists
import subprocess

def extract_frames(config,image_path,video_path,PITCH,YAW):
    if not exists(image_path):
        makedirs(image_path)
        images=join(image_path,f'{PITCH}_{YAW}_%05d.png')
        cmd = [
        'ffmpeg',
        '-i', video_path,
        '-r', str(config['rate']),
        images]
        subprocess.call(cmd)