from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

plt.rcParams.update({'font.size': 22,'axes.titlesize':22, 'axes.labelsize':22, 'xtick.labelsize':22, 'ytick.labelsize':22})
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
    for appr in ["finetuning","lwf"]:
        to_compare = [f"3_mnemonic_code/previous_interesting/{appr}-mnemonic-1op", 
                    f"3_mnemonic_code/previous_interesting/{appr}-mnemonic-2op",
                    f"3_mnemonic_code/previous_interesting/{appr}-mnemonic-3op",
                    f"3_mnemonic_code/previous_interesting/{appr}-mnemonic-5op"]
        
        fig,axes = plt.subplots(2,2,figsize=(15,15))
        for file,ax in zip(to_compare,axes.flat):
            print(file)
            attack = parse(Path(file))
            baseline = parse(Path(file+"-noattack"))
            # for s in attack:
            #     for line in s:
            #         print(line)
            #     print("")

            data_array = np.array(attack[1])
            data_array_baseline = np.array(baseline[1])
            diff = data_array_baseline-data_array

            # make triangle
            for i in (0,1,2,3,4):
                for j in (0,1,2,3,4):
                    if i<j:
                        diff[i][j]=np.nan

            sns.heatmap(diff, ax=ax, annot=True, fmt=".1f", cmap="Reds", cbar=False, linewidths=.5)


            ax.set_title(f"alfa={file[-3:-2]}%")
            ax.set_xlabel("Zadanie")
            ax.set_ylabel("Etap")
            ax.set_xticklabels([1,2,3,4,5])
            ax.set_yticklabels([1,2,3,4,5])

        name = "Dostrajanie z buforem" if appr=="finetuning" else "LwF"
        fig.tight_layout(pad=2.0)
        fig.suptitle(name)
        plt.savefig(f"mnemonic_comparison_{appr}.png",bbox_inches='tight')
