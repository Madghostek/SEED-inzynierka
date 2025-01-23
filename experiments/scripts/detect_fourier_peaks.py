import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def calculate_average_spectrum(folder_path: Path, slice=None) -> np.ndarray:
    """
    Calculate the average spectrum of PNG images in the given folder.
    """
    spectra = []

    images = os.listdir(folder_path)
    if slice:
        images=images[slice[0]:slice[1]]

    for filename in tqdm(images):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')  # use grayscale (maybe should check spectrum of each color?)
            image_array = np.array(image)

            fft_image = np.fft.fft2(image_array)
            fft_shifted = np.fft.fftshift(fft_image)

            magnitude_spectrum = np.log1p(np.abs(fft_shifted))
            spectra.append(magnitude_spectrum)

    average_spectrum = np.mean(spectra, axis=0)

    return average_spectrum

def plot_spectrum(spectrum, title="Uśrednione spektrum", name: str = "diff.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(spectrum, cmap='viridis')  # log scale
    plt.title(title)
    plt.colorbar(label="Amplituda")
    plt.xlabel("Częstotliwość w osi X")
    plt.ylabel("Częstotliwość w osi Y")
    #plt.show()
    plt.savefig(name,bbox_inches='tight')


def diffs1():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",type=str)
    args = parser.parse_args()

    folder_path = Path(args.path)
    avg_spectrum = calculate_average_spectrum(folder_path)
    #clean = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/clean/train"))
    clean = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/clean/train"),(0,30000))
    clean2 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/clean/train"),(30000,60000))

    #plot_spectrum(clean)
    #plot_spectrum(avg_spectrum)
    
    #plot_spectrum(clean2)
    plot_spectrum(clean-clean2, name="clean_diff.png")
    #plot_spectrum(clean-avg_spectrum, name="attack_diff.png")


def diffs2():
    plt.rcParams['figure.max_open_warning'] = 50
    parser = argparse.ArgumentParser()
    parser.add_argument("path",type=str)
    args = parser.parse_args()

    #folder_path = Path(args.path)
    #avg_spectrum = calculate_average_spectrum(folder_path)
    task0 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/class0/train0"))
    # plt.imshow(task0)
    # plt.show()
    task1 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/class0/train1")) 
    # plt.imshow(task1)
    # plt.show()
    task2 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/class0/train2"))
    task3 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/class0/train3"))
    task4 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/class0/train4"))

    tasks = (task0,task1,task2,task3,task4)
    labels = ("0","1","2","3","4")

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
    fig.suptitle("Różnice spektralne między zadaniami")
    for i in range(4):
        for j in range(4):
            ax = axes[i][j]
            ax.axis("off")
            if i+1>j:
                im = ax.imshow(tasks[j]-tasks[i+1], cmap='viridis')
                ax.set_title(f"T{j+1}-T{i+2}")

    fig.colorbar(im, label="Amplituda", ax=axes.ravel().tolist())
    plt.savefig("fig53_grid.png",bbox_inches='tight')


if __name__ == "__main__":
    diffs2()
