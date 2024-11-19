import os

dataset = "../../vincent-dataset-vol-1/UAVid/original_data"
ref_vids = os.listdir("../../vincent-sess-dataset/UAVid/downsampled_original_data")
videos = os.listdir(dataset)

clip_length = 100
k = max([int(v.split("_")[0]) for v in os.listdir(os.path.join("../../vincent-dataset-vol-1/UAVid/data"))]) + 1

label_name = lambda x: x[:-4] + '.png'

v_iter = tqdm(videos)
for v_name in v_iter:
    v_iter.set_description(v_name)
    if v_name not in ref_vids:
        frame_dir = os.path.join(dataset, v_name, 'origin')
        seg_dir = os.path.join(dataset, v_name, 'mask_orig')
        frames = os.listdir(frame_dir)
        frames.sort()
        frames = frames[1:]
        clips = [frames[i:i+clip_length] for i in range(0, len(frames), clip_length)]
        for (i, clip) in enumerate(clips):
            clip_name = f"{k}_{v_name}_{i}"
            k += 1
            clip_dir = os.path.join('../../vincent-dataset-vol-1/UAVid/data', clip_name)
            if not os.path.exists(clip_dir):
                os.mkdir(clip_dir)
                os.mkdir(os.path.join(clip_dir, 'origin'))
                os.mkdir(os.path.join(clip_dir, 'mask_orig'))
                for f_name in clip:
                    frame = Image.open(os.path.join(frame_dir, f_name))
                    if f_name in os.listdir(seg_dir):
                        seg = np.array(Image.open(os.path.join(seg_dir, f_name)))
                        new_seg_img = np.zeros(seg.shape[:-1], np.uint8)
                        new_seg_img[np.all(seg == np.array([0, 0, 0]), axis=-1)] = 0
                        new_seg_img[np.all(seg == np.array([128, 0, 0]), axis=-1)] = 1
                        new_seg_img[np.all(seg == np.array([128, 64, 128]), axis=-1)] = 2
                        new_seg_img[np.all(seg == np.array([192, 0, 192]), axis=-1)] = 3
                        new_seg_img[np.all(seg == np.array([0, 128, 0]), axis=-1)] = 4
                        new_seg_img[np.all(seg == np.array([128, 128, 0]), axis=-1)] = 5
                        new_seg_img[np.all(seg == np.array([64, 64, 0]), axis=-1)] = 6
                        new_seg_img[np.all(seg == np.array([64, 0, 128]), axis=-1)] = 7
                        new_seg_img = Image.fromarray(new_seg_img)
                        new_seg_img.save(os.path.join(clip_dir, 'mask_orig', label_name(f_name)))
                        new_seg_img.close()
                    frame.save(os.path.join(clip_dir, 'origin', f_name))
                    frame.close()
            #for f_name in clip:
            #    if f_name in os.listdir(os.path.join(dataset, v_name, "mask")):
            #        seg = Image.open(os.path.join(seg_dir, f_name))
            #       seg.save(os.path.join(clip_dir, 'mask', label_name(f_name)))
            #       seg.close()