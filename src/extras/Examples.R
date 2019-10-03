# Title     : Examples in Examples.md
# Objective : To have a concrete stuff rather than in pdf
# Created by: mayukh
# Created on: 03.10.19

library(abc)
ss_predict=read.csv('ss_predicted.csv.gz')
target=read.csv('ss_target.csv.gz')
index=as.vector(factor(as.matrix(read.table('model_index.csv.gz',header=TRUE))))

cvmodsel=cv4postpr(as.vector(index),as.matrix(ss_predict[,-1]),nval=100,method='rejection',tols=.01)
summary(cvmodsel)


params=read.csv('params.csv.gz')
ss=read.csv('ss_predicted.csv.gz')
target=read.csv('ss_target.csv.gz')
res=abc(target = target,param=params,sumstat = ss,tol=.01,method='neuralnet',transf = 'log')
summary(res)
plot(res,param=params)