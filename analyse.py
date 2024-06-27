import argparse
from colorama import init, Fore, Back, Style


def analyse(path, file_path):
    with open(path, "r") as f:
        lines = f.readlines()
        succeeded = []
        failed = []
        for i in range(len(lines)):
            line = lines[i]
            if line[0] == "=":
                pos_line = lines[i + 1]
                cut_pos = pos_line.find("(")
                temp_list = pos_line[cut_pos + 1 :].split(",")
                x, y = int(temp_list[0]), int(temp_list[1])
                result_line = lines[i + 3]

                if result_line.find("true") != -1:
                    succeeded.append((x, y))
                else:
                    failed.append((x, y))
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].find("map:") != -1:
                j = i + 1
                while len(lines[j]) > 6:
                    j += 1
                map_lines = lines[i + 1 : j]
                map_lines = [line[5:] for line in map_lines]
                for x in range(len(map_lines)):
                    for y in range(len(map_lines[x])):
                        if (x + 1, y + 1) in succeeded:
                            print(Fore.GREEN + map_lines[x][y], end="")
                        elif (x + 1, y + 1) in failed:
                            print(Fore.RED + map_lines[x][y], end="")
                    print(Style.RESET_ALL)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, help="path of a maude file that you want to evaluate"
    )

    return parser.parse_args()


args = get_args()
analyse("out.txt", args.file_path)
