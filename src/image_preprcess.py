import glob
import re
from torchvision import transforms
from PIL import Image

CENTER_CROP = 160
RESIZE = 64
# D:\Aoyama\SSRW2022\src\train\ROHAN4600_0001-0500\ROHAN4600_0001
def image_preprocess():
    all_dirs = glob.glob('train/ROHAN4600*0')
    all_speaking = []
    for dir in all_dirs:
        all_speaking.extend(glob.glob(dir+'/*'))
    transform = transforms_maker()
    for speaking in all_speaking:
        dir_name = 'train/transformed_train/' + speaking[-14:]
        filenames = glob.glob(speaking+'/*')
        for filename in filenames:
            img = Image.open(filename)
            img = transform(img)
            img.save(dir_name+'/'+filename[-9:])
            print(filename)

def transforms_maker():
    transform = transforms.Compose(
        [transforms.CenterCrop(CENTER_CROP), transforms.Resize(RESIZE)]
    )
    return transform

if __name__ == '__main__':
    image_preprocess()