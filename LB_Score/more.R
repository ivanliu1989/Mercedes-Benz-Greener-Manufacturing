# ID = 409, y = 91.00760
# 
# ID = 437, y = 85.96960
# 
# ID = 493, y = 108.40135
# 
# ID = 1664, y = 112.93977
# 
# ID: 2129, y = 112.03
# 
# ID: 2342, y = 93.06
# 
# ID: 7055, y = 91.549
# 
# ID: 4958, rho: -59.19183, y: 113.58711 (tested with new submission)
# ID: 4960, rho: -59.22558, y: 89.83957 (not tested, out of ammo)
# 
# ID = 488, y = 113.39009
# 
# ID=253 indeed has y=115.93724.


library(jsonlite)
LBScore = fromJSON('./LB_Score/all_questions.json')

LBScore.split = rbindlist(LBScore$answers, idcol = TRUE)
setDT(LBScore.split)
LBScore.split = LBScore.split[inside_public_lb==TRUE,]
LBScore.split$ID = LBScore.split$.id
LBScore.split$y = ifelse(LBScore.split$`_id` == 0, LBScore.split$y_value, LBScore.split$`_id`)
LBScore.split = LBScore.split[,.(ID, y)]

LBScore.extra = data.table(ID = c(409,437,493,1664,2129,2342,7055,4958,4960,488,253),
                           y= c(91.00760,85.96960,108.40135,112.93977,112.03,93.06,91.549,113.58711,89.83957,113.39009,115.93724))
LBScore.split$ID = LBScore$id[LBScore.split$ID]
LBScore = rbind(LBScore.split, LBScore.extra) 
setorder(LBScore, ID)

LBScore
write.csv(unique(LBScore), file = "./data/LBScore.csv", row.names = F)


# mean when there was a duplicate, it helped really, really a lot!
# I have introduced a variable counting the repetitions: for each row, I include (as a new column) a variable indicating how many 'clones' does this row have.