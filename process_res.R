library(data.table)
setwd('~/Documents/Princeton/Senior/thesis/code/mycode')
res <- fread('object_trials.csv')

res[, t_perturb := abs(x_perturb) + abs(y_perturb)]
res <- res[successes > 48]
res[t_perturb==0][, unique(Object)]
obj_list <- res[, grepl('-e', Object), by=Object][V1==FALSE][order(Object)]

fwrite(obj_list, 'obj_list.csv')

