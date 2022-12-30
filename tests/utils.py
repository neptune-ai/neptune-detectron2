import os
import shutil


def get_images():
    img_dir = "./datasets/coco/train2014"
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) == 0:
        os.makedirs(img_dir, exist_ok=True)
        os.system("wget http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg")
        os.system("wget http://images.cocodataset.org/train2014/COCO_train2014_000000384029.jpg")
        shutil.move("COCO_train2014_000000057870.jpg", img_dir)
        shutil.move("COCO_train2014_000000384029.jpg", img_dir)
