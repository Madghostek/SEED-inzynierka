from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme()

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
    folder1 = "opacitygrid2-finetune"
    folder2 = "opacitygrid2-lwf"
    
    plt.tight_layout()
    fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(13,6))
    axes[0].set_ylabel("Celność dla zadania 2")



    for ax,approach in zip(axes,[folder1,folder2]):
        variant_series = {}

        files = {}

        for file in sorted(Path(approach).glob("*")):
            files[file.name] = parse(file)

        for file in files:
            tag_accuracies = files[file][1] #TAg
            tag_accuracies = np.array(tag_accuracies)
            
            # get only task 2
            print(tag_accuracies[:,1])
            task2 = np.concatenate([[np.nan],tag_accuracies[:,1]])
            print(task2)
            task2[1]=np.nan
            variant_series[f"alfa={file[7:-10]}%"]=task2

       
        sns.lineplot(data=variant_series, ax=ax)

        ax.set_title("LwF" if approach==folder2 else "finentuning")
        ax.set_xticks([1,2,3,4,5])
        ax.set_ylim(50,100)
        ax.set_xlabel("Uczone zadanie")
        #ax.set_ylim([0,100])

    plt.savefig(f"task2_opacities.png",bbox_inches='tight')