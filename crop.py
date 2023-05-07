import os
import subprocess
import tqdm
import argparse

def crop(path):
    paths = os.listdir(path)
    for p in tqdm.tqdm(paths):
        file = p.split('.')[0]
        file = '%s/%s' % (path, file)
        cmd = 'ffmpeg -y -loglevel quiet -i %s.mp4 -f wav -ar 16000 %s.wav' % (file, file)
        # print(cmd)
        subprocess.run(cmd, shell=True)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("av-yolo")
    
    parser.add_argument('--path', type=str, default='')

    args = parser.parse_args()

    crop(args.path)