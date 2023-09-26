from os.path import exists
import subprocess

def compressor(src_video,qp,v,out_video_path):
    ## The function will compress the video to each combinations of QP and resolution
    if not exists(out_video_path):
        vcodec = 'libx265'
        width, height=v
        ffmpeg_cmd = ['/usr/bin/ffmpeg', '-i', src_video, '-vcodec', vcodec, '-x265-params', 'qp=' + qp, '-pix_fmt', 'yuv422p',
                      '-bf', '0', '-g', '60', '-preset', 'veryfast',
                      '-vf', 'scale={}:{}'.format(height, width), '-sws_flags', 'bicubic', out_video_path, '-y']
        subprocess.call(ffmpeg_cmd)