from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

separator = "*"*108


def parse(path: Path):
    """Parses output file from FACIL.
    return[0] - TAw Acc
    return[1] - TAg Acc
    return[2] - TAw Forg
    return[3] - TAg Forg"""
    with path.open() as f:
        lines = f.readlines()

    sections = []
    cur_section = []

    for line in reversed(lines):
        if line.strip()==separator:
            sections.append(cur_section[::-1])
            cur_section=[]
            if len(sections)==5:
                break
        else:
            line = line.strip()
            if str.isnumeric(line[0]):
                cur_section.append(line.split("\t")[0])

    sections_ordered = sections[1:][::-1]

    converted_secitons = []

    for section in sections_ordered:
        numbers_in_section = []
        for line in section:
            nums_strings = [x for x in line.strip().split("%") if x]
            #print(line,nums_strings)
            nums = list(map(float,nums_strings))
            numbers_in_section.append(nums)
        converted_secitons.append(numbers_in_section)
    return converted_secitons

    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path",type=str)
    args = parser.parse_args()
    sections = parse(Path(args.path))
    for s in sections:
        for line in s:
            print(line)
        print("")

    data_array = np.array(sections[1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(data_array, annot=True, fmt=".1f", cmap="Reds", cbar=True, linewidths=.5)

    plt.title("Task-agnostic accuracy (5% vis) 50% dataset touched")
    plt.xlabel("Time point")
    plt.ylabel("Task")

    plt.savefig(Path(args.path).name)
