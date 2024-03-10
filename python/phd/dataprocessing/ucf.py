import cv2
from typing import Dict, List
from time import sleep
import numpy as np
from scipy.interpolate import interp1d

DATASET_DIR = '/media/data/Datasets/action_youtube_naudio/'
SAVE_DIR = '/media/data/Datasets/ucf-processed/'

import sys
import os

def get_paths() -> Dict[str, Dict[int, str]]:
    dirs = os.scandir(DATASET_DIR)
    class_folders = {}
    for d in dirs:
        if d.is_dir(): 
            path = d.path
            class_name = d.name
            video_dirs = os.scandir(path)            
            video_files_paths = {}
            for i, vd in enumerate(video_dirs):     
                if vd.name != 'Annotation':  
                    vid_paths = [v.path for v in os.scandir(vd.path) if v.path.endswith('.avi')]
                    video_files_paths[i] = vid_paths
            class_folders[class_name] = video_files_paths
    return class_folders

def display_vid(x: np.ndarray, fps):
    N = x.shape[-1]
    T = 1000//int(fps)
    for n in range(N):
        cv2.imshow('Frame',x[..., n].astype(np.uint8))
        cv2.waitKey(T)        
    cv2.destroyAllWindows()

def resample_video(x, orig_fps, target_fps):
    t = np.arange(x.shape[-1])/orig_fps
    interp = interp1d(t, x, kind='nearest', axis=-1)
    T = x.shape[-1]/orig_fps
    N = int(T*target_fps) - 1
    t_new = np.arange(N)/target_fps
    return np.clip(interp(t_new), 0.0, 255.0)

def write_video(fname, x, fps):
    x = x.astype(np.uint8)
    out = cv2.VideoWriter(SAVE_DIR + fname, cv2.VideoWriter_fourcc(*"MJPG"), fps, (x.shape[1], x.shape[0]), isColor=len(x.shape) == 4)
    N = x.shape[-1]
    for n in range(N):
        out.write(x[..., n])
    out.release()

def read_video(file_path: str, display = False, save = False):
    cap = cv2.VideoCapture(file_path)
    frames = []
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    d = 4
    wd, hd = int(w//d), int(h//d)
    fps = cap.get(cv2.CAP_PROP_FPS)
    play_speed = 1000//int(fps)
    print(f'{file_path=}, {wd=}, {hd=}, {fps=}')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (wd, hd), interpolation=cv2.INTER_CUBIC)
            frame = np.array(frame)
            frames.append(frame[..., np.newaxis])                       
        else:
            break
    cap.release()
    
    print(f'{len(frames)=}')
    x = np.concatenate(frames, axis=-1)
    
    if x.shape[-1] < 60: return None
    
    target_fps = 10
    print(x.shape)
    x = resample_video(x.astype(np.float32), fps, target_fps)
    
    h_pad = 270//d - x.shape[0]
    w_pad = 320//d - x.shape[1]
    x = np.pad(x, ((h_pad//2, h_pad - h_pad//2), (w_pad//2, w_pad - w_pad//2), (0, 0))) 
    
    
    if display: display_vid(x, target_fps)    
    if save:
        save_name = file_path[file_path.rfind('/')+1:]
        write_video(save_name, x, target_fps)
    
    return x
    

def load_videos(save = False) -> Dict[str, Dict[int, np.ndarray]]:
    paths = get_paths()
    videos = {}
    for label in paths.keys():
        exs = paths[label]
        examples = {}
        for ex_num in exs.keys():
            p_list = exs[ex_num]
            vid_entries = []
            for vid_path in p_list:
                vid = read_video(vid_path, save=save)
                if vid is not None: vid_entries.append(vid)
            examples[ex_num] = vid_entries
        videos[label] = examples
    return videos


import pickle as pkl
def save_train_test(p = 0.2):
    vids = load_videos(save=True)
    X_train, y_train, X_test, y_test = [],[],[],[]
    for label in vids.keys():
        n_examples = len(vids[label].keys())
        n_train = int((1 - p) * n_examples)
        for i, vs in enumerate(vids[label].values()):
            if i < n_train:
                for v in vs:
                    X_train.append(v)
                    y_train.append(label)
            else:
                for v in vs:
                    X_test.append(v)
                    y_test.append(label)
    print(f'{len(X_train)=}, {len(X_test)=}')
    with open('data/ucf.pkl', 'wb') as file:
        pkl.dump((X_train, y_train, X_test, y_test), file)
        
def read_train_test():
    with open('data/ucf.pkl', 'rb') as file:
        X_train, y_train, X_test, y_test = pkl.load(file)
    return X_train, y_train, X_test, y_test
        
            

if __name__ == '__main__':
    import pprint
    # pprint.pprint(get_paths())
    # x = read_video('/media/data/Datasets/action_youtube_naudio/horse_riding/v_riding_09/v_riding_09_01.avi', False, True)
    save_train_test()

