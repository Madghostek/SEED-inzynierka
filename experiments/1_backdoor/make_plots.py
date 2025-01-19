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
    base = Path("1data_1op_mnist")
    #approach = "lwf"
    approaches = ["finetune","lwf"]

    plt.tight_layout()

    fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(13,6))
    axes[0].set_ylabel("Średnia celność dla wszystkich zadań")

    for approach,ax in zip(approaches,axes):

        files = {}

        labels = ["mnist-task1","mnist-task2","mnist-task3","mnist-task4","mnist-task5"]

        for file in sorted(base.glob("*")):
            files[file.name] = parse(file)

        average_series = {}
        for i,file in enumerate(x for x in files if approach in x):
            series = [None] #1-index the tasks
            for tag_accuracies in files[file][1]: #TAg
                tag_accuracies = np.array(tag_accuracies)
                # trick to ignore zeros
                tag_accuracies[tag_accuracies==0]=np.nan
                temp = np.nanmean(tag_accuracies,axis=0)
                series.append(temp)
            average_series[labels[i]]=series
        

        sns.lineplot(data=average_series, ax=ax)
 
        ax.set_title(approach if approach=="lwf" else "finentuning")
        ax.set_xticks([1,2,3,4,5])
        ax.set_xlabel("Uczone zadanie")

    plt.savefig(f"fig51.png",bbox_inches='tight')