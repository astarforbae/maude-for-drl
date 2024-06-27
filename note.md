```bash
# load
load /home/ZhangXingYi/local/maude/model-checker.maude
```

```bash
# 搜索
search init =>! S:State .

# 验证

# 不会到边界
red modelCheck(init, [] ~ BeyondBoundary) . 
# 不会掉进坑里
red modelCheck(init, [] ~ InHole) .
# 胜利
red modelCheck(init, [] <> Success) .
red modelCheck(init,  ~ <> Success) .

# 没有违规操作时胜利 [] (~(InHole \/ BeyondBoundary) \/ Success)

```


