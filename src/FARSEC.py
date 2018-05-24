import pre

# 初始化过滤器
Filter = pre.Filtering()

# 读取数据并预处理
Filter.treatment(dir = "Ambari2.csv")
# Filter.readDateFromFile(date = "date.csv")


# 找到安全相关关键词
Filter.findSRW()
print("find security related keywords succeed")

# 过滤NSBR并训练
Filter.farsec(support='farsecsq',train='knn')