import pre

Filter = pre.Filtering()

br,label = Filter.readcsv(dir = "Ambari.csv",summary_col=1,description_col =2,security_col=3)

print( "find security related keywords succeed ? ",Filter.findSRW(br,label))

Filter.farsec(support='farsecsq',train='knn')