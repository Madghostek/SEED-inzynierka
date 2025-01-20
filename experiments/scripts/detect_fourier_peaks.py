import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def calculate_average_spectrum(folder_path: Path, slice: tuple=(0,60000)) -> np.ndarray:
    """
    Calculate the average spectrum of PNG images in the given folder.
    """
    spectra = []

    for filename in tqdm(os.listdir(folder_path)[slice[0]:slice[1]]):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')  # use grayscale (maybe should see spectrum of each color?)
            image_array = np.array(image)

            fft_image = np.fft.fft2(image_array)
            fft_shifted = np.fft.fftshift(fft_image)

            magnitude_spectrum = np.abs(fft_shifted)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path",type=str)
    args = parser.parse_args()

    folder_path = Path(args.path)
    avg_spectrum = calculate_average_spectrum(folder_path)
    clean = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/clean/train"))
    #clean = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/clean/train"),(0,30000))
    #clean2 = calculate_average_spectrum(Path("/home/tomek/datasets/WhiteSquare/clean/train"),(30000,60000))

    #plot_spectrum(clean)
    #plot_spectrum(avg_spectrum)
    
    #plot_spectrum(clean2)
    #plot_spectrum(clean-clean2, name="clean_diff.png")
    plot_spectrum(clean-avg_spectrum, name="attack_diff.png")

