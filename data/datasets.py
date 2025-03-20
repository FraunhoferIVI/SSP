# UAVid: A Semantic Segmentation Dataset for UAV Imagery
# VDD: Varied Drone Dataset for Semantic Segmentation
# Arxiv Link UAVid: https://arxiv.org/abs/1810.10438
# Arxiv Link VDD: https://arxiv.org/pdf/2305.13608

# Here we use the extended classes from the VDD dataset

class UAVID:
    name = "UAVid"
    n_classes = 8
    img_size = (1080, 1920)
    fps = 20
    n_frames_vc = 8 #16
    path = "./datasets/UAVid" # Path to the dataset
    frame_folder = "origin"
    mask_folder = "mask"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".png"
    classes = {
        0: "background",
        1: "road",
        2: "vegetation",
        3: "tree",
        4: "person",
        5: "vehicle",
        6: "water",
        7: "building",
        8: "roof"
    }
    colors = {
        0: (0,0,0),
        1: (128, 0, 128),
        2: (112, 148, 32),
        3: (64, 64, 0),
        4: (255, 16, 255),
        5: (0, 128, 128),
        6: (0, 0, 255),
        7: (255, 0, 0),
        8: (64, 160, 120)
    }
    ignore_index = 255
    def convert_labels(label):
        label[label==0]=255
        label = label-1
        label[label>=8]=255
        return label

# Introduced in ACCV 2020: [Semantics through Time: Semi-supervised Segmentation of Aerial Videos with Iterative Label Propagation]
# Arxiv Link: https://arxiv.org/pdf/2010.01910v1
    
class RURALSCAPES:
    name = "RuralScapes"
    n_classes = 12
    img_size = (2160, 4096)
    fps = 10
    n_frames_vc = 8 #16
    path = "./datasets/ruralscapes"
    frame_folder = "origin"
    mask_folder = "mask"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".jpg"
    classes = {
        0: "background",
        1: "residential",
        2: "land",
        3: "forest",
        4: "sky",
        5: "fence",
        6: "road",
        7: "hill",
        8: "church",
        9: "car",
        10: "person",
        11: "haystack",
        12: "water"
    }
    colors = {
        0: (0,0,0),
        1: (255,255,0),
        2: (0,255,0),
        3: (0,127,0),
        4: (0,255,255),
        5: (127,127,0),
        6: (255,255,255),
        7: (127,127,63),
        8: (255,0,255),
        9: (127,127,127),
        10: (255,0,0),
        11: (255,127,0),
        12: (0,0,255)
    }
    ignore_index = 255
    def convert_labels(label):
        label[label==0]=255
        label = label-1
        label[label>=12]=255
        return label
    