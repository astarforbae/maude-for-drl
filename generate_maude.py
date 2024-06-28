import os


def get_file_name(file_dir):
    # 遍历文件夹中所有文件，获得最大的数字 ，每个文件的命名格式为：数字.maude
    max_num = 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            num = int(file.split(".")[0])
            if num > max_num:
                max_num = num
    return str(max_num + 1) + ".maude"


def annotate(line, line_break=True):
    return "---  " + line + ("\n" if line_break else "")


def get_map_in_lines(map_dir_path):
    with open(os.path.join(map_dir_path, "grid.txt"), "r") as f:
        lines = f.readlines()
    return lines


def get_actions_in_lines(agent_path):
    with open(os.path.join(agent_path, "actions.txt"), "r") as f:
        lines = f.readlines()
    return lines


def indents(line, num=1):
    return "\t" * num + line + "\n"


def get_holes(map_dir_path):
    with open(os.path.join(map_dir_path, "grid.txt"), "r") as f:
        lines = f.readlines()
    poses = []
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j] == "X":
                poses.append((i + 1, j + 1))
    return poses


def generate(agent_path):
    splitted = agent_path.split("/")
    map_name = splitted[-3]
    map_path = "/".join(splitted[:-1])
    file_dir = os.path.join("/home/ZhangXingYi/codes/CLIFF/maude_files/", map_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = get_file_name(file_dir)
    # file_name = "test.maude"
    file_path = os.path.join(file_dir, file_name)
    n = map_name.split("x")[0]
    m = map_name.split("x")[1]
    holes = get_holes(map_path)
    actions = get_actions_in_lines(agent_path)
    # 开始写文件
    with open(file_path, "w") as f:
        wt = f.write
        wt(annotate("agent path: " + agent_path))
        wt(annotate("map dir path: " + map_path))
        wt(annotate(""))
        wt(annotate("map size: " + map_name))
        wt(annotate("A: start point."))
        wt(annotate("B: end point."))
        wt(annotate("X: hole."))
        wt(annotate(".: road."))
        wt(annotate(""))
        # 打印地图
        wt(annotate("map: "))
        for line in get_map_in_lines(map_path):
            wt(annotate(line, line_break=False))
        wt(annotate(""))
        # 打印智能体策略
        wt(annotate("agent strategy:"))
        for line in actions:
            wt(annotate(line, line_break=False))
        wt("\n")
        wt("mod CHECKER-" + file_name.split(".")[0] + " is\n")
        wt(indents("protecting NAT ."))
        wt(indents("including MODEL-CHECKER ."))
        wt(indents("including LTL-SIMPLIFIER ."))
        wt(indents("op _,_ : Nat Nat -> State [ctor] ."))
        wt(indents("vars x y : Nat ."))
        wt(indents("var S : State ."))
        wt(indents("var P : Prop ."))
        wt(indents("op init : -> State ."))
        wt(indents("eq init = (1, 1) ."))
        wt(indents("op BeyondBoundary : -> Prop ."))
        wt(indents("op InHole : -> Prop ."))
        wt(indents("op Success : -> Prop ."))
        wt(
            indents(
                f"ceq (x, y) |= BeyondBoundary =  true if (x > {n}) or (x == 0) or (y == 0) or (y > {m}) ."
            )
        )
        if len(holes) > 0:
            wt(
                indents(
                    "ceq (x, y) |= InHole = true if "
                    + " or ".join([f"(x == {i} and y == {j})" for i, j in holes])
                    + " ."
                )
            )
        wt(indents(f"ceq (x, y) |= Success = true if x == {n} and y == {m} ."))
        wt(indents("eq S |= P = false [owise] ."))
        wt(indents(""))
        # 定义迁移规则
        for i in range(1, int(n) + 1):
            for j in range(1, int(m) + 1):
                if i == int(n) and j == int(m):
                    wt(indents(f"rl [{i}-{j}] : ({i}, {j}) => ({i}, {j}) ."))
                    continue
                ch = actions[i - 1][j - 1]
                if ch == "X":
                    wt(indents(f"rl [{i}-{j}] : ({i}, {j}) => ({i}, {j}) ."))
                elif ch == "→":
                    wt(indents(f"rl [{i}-{j}] : ({i}, {j}) => ({i}, {j + 1}) ."))
                elif ch == "←":
                    wt(indents(f"rl [{i}-{j}] : ({i}, {j}) => ({i}, {j - 1}) ."))
                elif ch == "↑":
                    wt(indents(f"rl [{i}-{j}] : ({i}, {j}) => ({i - 1}, {j}) ."))
                elif ch == "↓":
                    wt(indents(f"rl [{i}-{j}] : ({i}, {j}) => ({i + 1}, {j}) ."))
        wt("endm")


generate("/home/ZhangXingYi/codes/CLIFF/map/6x6/m4/2")
