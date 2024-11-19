import os
from PIL import Image

dataset = "../../vincent-dataset-vol-1/ruralscapes/src_full"
dataset_gt = "../../vincent-dataset-vol-1/ruralscapes/gt"
videos = os.listdir(dataset)

clip_length = 50
k = 0

label_name = lambda x: x[:-4] + '.png'

for v in videos:
    frame_dir = os.path.join(dataset, v)
    seg_dir = os.path.join(dataset_gt, v)
    frames = os.listdir(frame_dir)
    frames.sort()
    frames = frames[1:-1]
    clips = [frames[i:i+clip_length] for i in range(0, len(frames), clip_length)]
    for (i, clip) in enumerate(clips):
        if len(clip) >= 10:
            clip_name = f"{k}_{v}_{i}"
            k += 1
            print(f"{clip_name}: {len(clip)} frames")
            clip_dir = os.path.join('../../vincent-dataset-vol-1/ruralscapes/data', clip_name)
            if not os.path.exists(clip_dir):
                os.mkdir(clip_dir)
                os.mkdir(os.path.join(clip_dir, 'origin'))
                os.mkdir(os.path.join(clip_dir, 'mask'))
                for f_name in clip:
                    frame = Image.open(os.path.join(frame_dir, f_name))
                    if label_name(f_name) in os.listdir(seg_dir):
                        seg = Image.open(os.path.join(seg_dir, label_name(f_name)))
                        seg.save(os.path.join(clip_dir, 'mask', label_name(f_name)))
                        seg.close()
                    frame.save(os.path.join(clip_dir, 'origin', f_name))
                    frame.close()


path = "../../vincent-dataset-vol-1/ruralscapes"

def get_split_indices(split):
    with open(os.path.join(path, split)) as f:
        indices = f.readlines(-1)

    video_indices = []
    for idx in indices:
        v_idx = idx[:-1] # remove \n
        if v_idx not in video_indices:
            video_indices.append(v_idx)

    return video_indices

video_train_idx = get_split_indices("train_old.txt")
video_val_idx = get_split_indices("val_old.txt")

new_video_train_idx = []
new_video_val_idx = []

videos = os.listdir(path + '/data')
for v in videos:
    base_name = v.split("_")[1] + "_" + v.split("_")[2]
    if base_name in video_train_idx:
        new_video_train_idx.append(v)
    elif base_name in video_val_idx:
        new_video_val_idx.append(v)
    else:
        print(f"{base_name}: error")

key = lambda x: int(x.split("_")[0])
new_video_train_idx.sort(key=key)
new_video_val_idx.sort(key=key)

with open(path + '/train.txt', 'w') as f:
    for v in new_video_train_idx:
        f.write(v + '\n')

with open(path + '/val.txt', 'w') as f:
    for v in new_video_val_idx:
        f.write(v + '\n')