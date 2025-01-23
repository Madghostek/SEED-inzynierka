from pathlib import Path

from parse_outputs import parse

clean_acc = 81.2 # just taken from `blending-lwf-clean` run 

def main():
    forgettings = []
    for file in sorted(Path("2_blending/grid_results").glob("*")):
        data = parse(file)
        forgettings.append(clean_acc-data[1][1][0])

    i = 0
    for ratio in (0.2,0.4,0.6,0.8,1):
        for op in (0.2,0.3,0.4):
            print(op,ratio,round(forgettings[i],2))
            i+=1

if __name__=="__main__":
    main()
    