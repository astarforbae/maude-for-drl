import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, help="path of a maude file that you want to evaluate"
    )
    parser.add_argument(
        "--checker_path",
        type=str,
        help="path of MODEL-CHECKER.maude file",
        default="/home/ZhangXingYi/local/maude/model-checker.maude",
    )

    return parser.parse_args()


def enter(line):
    return line + "\n"


if __name__ == "__main__":
    args = get_args()
    with open("evaluate.maude", "w") as f:
        wt = f.write
        wt(enter("load " + args.checker_path + " ."))
        wt(enter("load " + args.file_path + " ."))
        map_name = args.file_path.split("/")[-2]
        n, m = map_name.split("x")[0], map_name.split("x")[1]
        for i in range(1, int(n) + 1):
            for j in range(1, int(m) + 1):
                wt(enter(f"red modelCheck(({i}, {j}), [] <> Success) ."))
        wt(enter("quit ."))
