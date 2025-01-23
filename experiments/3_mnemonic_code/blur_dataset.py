import PIL.Image
import PIL.ImageFilter
from pathlib import Path
from tqdm import tqdm


train_set = "/home/tomek/datasets/MnemonicCode/mnemonic_blur/train"
radius = 0.5

def blur_gauss_image(img: PIL.Image.Image, filter: PIL.ImageFilter.GaussianBlur):
    return img.filter(filter)


def main():
    filter = PIL.ImageFilter.GaussianBlur(radius=radius)
    for image in tqdm(Path(train_set).glob("*")):
        img1 = PIL.Image.open(image)
        img1.show()

        img2 = blur_gauss_image(img1, filter)
        img2.show()

        #img2.save(image)


if __name__=="__main__":
    main()
