```bash
# load
load /home/ZhangXingYi/local/maude/model-checker.maude
```

```bash
# 搜索
search init =>! S:State .

# 验证 - 从init状态出发

# 不会到边界
red modelCheck(init, [] ~ BeyondBoundary) . 
# 不会掉进坑里
red modelCheck(init, [] ~ InHole) .
# 胜利
red modelCheck(S:State, [] <> Success) .
red modelCheck(init,  ~ <> Success) .


# 验证 - 从所有状态出发

# 自动化脚本
./eval.sh /home/ZhangXingYi/codes/CLIFF/maude_files/4x4/3.maude /home/ZhangXingYi/local/maude/model-checker.maude /home/ZhangXingYi/local/maude/maude.linux64
```


