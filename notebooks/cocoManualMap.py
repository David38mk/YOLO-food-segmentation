import os
from pathlib import Path

#these paths should be changed !
SRC_ROOT = Path("datasets/coco/labels")         
DST_ROOT = Path("datasets/food_binary/labels")   

splits = ["train", "val"]   

coco_to_binary = {
    0: 0,   # person
    1: 0,   # bicycle
    2: 0,   # car
    3: 0,   # motorcycle
    4: 0,   # airplane
    5: 0,   # bus
    6: 0,   # train
    7: 0,   # truck
    8: 0,   # boat
    9: 0,   # traffic light
    10: 0,  # fire hydrant
    11: 0,  # stop sign
    12: 0,  # parking meter
    13: 0,  # bench
    14: 0,  # bird
    15: 0,  # cat
    16: 0,  # dog
    17: 0,  # horse
    18: 0,  # sheep
    19: 0,  # cow
    20: 0,  # elephant
    21: 0,  # bear
    22: 0,  # zebra
    23: 0,  # giraffe
    24: 0,  # backpack
    25: 0,  # umbrella
    26: 0,  # handbag
    27: 0,  # tie
    28: 0,  # suitcase
    29: 0,  # frisbee
    30: 0,  # skis
    31: 0,  # snowboard
    32: 0,  # sports ball
    33: 0,  # kite
    34: 0,  # baseball bat
    35: 0,  # baseball glove
    36: 0,  # skateboard
    37: 0,  # surfboard
    38: 0,  # tennis racket
    39: 0,  # bottle
    40: 0,  # wine glass
    41: 0,  # cup
    42: 0,  # fork
    43: 0,  # knife
    44: 0,  # spoon
    45: 0,  # bowl

    46: 1,  # banana
    47: 1,  # apple
    48: 1,  # sandwich
    49: 1,  # orange
    50: 1,  # broccoli
    51: 1,  # carrot
    52: 1,  # hot dog
    53: 1,  # pizza
    54: 1,  # donut
    55: 1,  # cake

    56: 0,  # chair
    57: 0,  # couch
    58: 0,  # potted plant
    59: 0,  # bed
    60: 0,  # dining table
    61: 0,  # toilet
    62: 0,  # tv
    63: 0,  # laptop
    64: 0,  # mouse
    65: 0,  # remote
    66: 0,  # keyboard
    67: 0,  # cell phone
    68: 0,  # microwave
    69: 0,  # oven
    70: 0,  # toaster
    71: 0,  # sink
    72: 0,  # refrigerator
    73: 0,  # book
    74: 0,  # clock
    75: 0,  # vase
    76: 0,  # scissors
    77: 0,  # teddy bear
    78: 0,  # hair drier
    79: 0   # toothbrush
}
DEFAULT_NO_FOOD = 0

for split in splits:
    src_dir = SRC_ROOT / split
    dst_dir = DST_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    for label_path in src_dir.rglob("*.txt"):
        rel = label_path.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  

            old_class_id = int(parts[0])
            new_class_id = coco_to_binary.get(old_class_id, DEFAULT_NO_FOOD)
            parts[0] = str(new_class_id)

            new_lines.append(" ".join(parts))

        if new_lines:
            with open(out_path, "w") as f:
                f.write("\n".join(new_lines))
                
#in data.yaml - this should be changed since we are tranforming this taskk from multiclass to binary classification

# nc: 2
# names:
#   0: no_food
#   1: food

# train: datasets/food_binary/images/train (path from the dataset)
# val: datasets/food_binary/images/val