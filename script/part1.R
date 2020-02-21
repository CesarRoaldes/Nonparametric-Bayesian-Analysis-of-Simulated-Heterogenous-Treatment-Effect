install.packages('estimatr')
install.packages('dummies')
install.packages('dplyr')
install.packages('matrixStats')
install.packages('mixtools')
install.packages('pracma')
install.packages('MASS')
install.packages('readr')
install.packages('matlib')
install.packages('ggplot2')
install.packages('EnvStats')



library(EnvStats)
library(estimatr)
library(dummies)
library(dplyr)
library(matrixStats)
library(mixtools) 
library(pracma)
library(MASS)
library(readr)
library(matlib)
library(ggplot2)



path <- getwd() 

########################################### SIMULATION DES DONNEES #####################################
########## 1. Simulation des données 
set.seed(20031996)
N <- 1000000   ## Taille de l'échantillon 
P=10            ## Nombre de covariates 
P_cont = 8      ## Nombre de variables continues 
P_cate = 2      ## Nombre de variables discrètes (dummy)
X_cont <- matrix(rexp(N*P_cont),N,P_cont) # Variables continues  
X_cate <- data.frame(X_Female=rbinom(N,1,prob=0.7) ,X_New=rbinom(N,1,prob=0.8)) # Variables discrètes  
t <- sample(0:1, N, replace=T)              # Variable de traitement
inter <- rep(1, N)                          # Intercept

X <- data.frame(X_cont,X_cate)
for (i in 1:ncol(X)){
  colnames(X)[i] <- paste("X",sep="_",i)
} # Nom des colonnes 

full_data2 <- data.frame(t,inter,X)

############ Dummies transform 
X_cont = data.frame(X_cont)
X_cont <-X_cont %>% mutate( quantile1 = as.integer(ntile(X_cont[,1],5))) %>%
  mutate( quantile2= as.integer(ntile(X_cont[,2],5))) %>%
  mutate( quantile3= as.integer(ntile(X_cont[,3],5))) %>%
  mutate( quantile4= as.integer(ntile(X_cont[,4],5))) %>%
  mutate( quantile5= as.integer(ntile(X_cont[,5],5))) %>%
  mutate( quantile6= as.integer(ntile(X_cont[,6],5))) %>%
  mutate( quantile7= as.integer(ntile(X_cont[,7],5))) %>%
  mutate( quantile8= as.integer(ntile(X_cont[,8],5)))

X_cont <- cbind(X_cont, dummy(X_cont$quantile1, sep = "_"), dummy(X_cont$quantile2, sep = "_"), dummy(X_cont$quantile3, sep = "_"),
                dummy(X_cont$quantile4, sep = "_"),
                dummy(X_cont$quantile5, sep = "_"),
                dummy(X_cont$quantile6, sep = "_"),
                dummy(X_cont$quantile7, sep = "_"),
                dummy(X_cont$quantile8, sep = "_"))


X_cont<- X_cont[,c((2*P_cont+1):ncol(X_cont))]  
colnum = as.character(rep(c(1:5),P_cont))
colnum2 = as.character(rep(1:P_cont, each=5))

for (i in 1:ncol(X_cont)){
  colnames(X_cont)[i] = paste("X",colnum2[i],colnum[i],sep="_")
}

X_cont <-X_cont %>% dplyr::select(-ends_with('1')) 
full_data1<- data.frame(t,inter,X_cont,X_cate)

############ Creation de l'effet de traitement
library(EnvStats)
beta = rep(2,10) 
gamma = c(1,1)
alpha=1
mu=1
set.seed(20031996)
erreur1= rnorm(nrow(full_data2))

set.seed(19061994)
erreur2 = rpareto(nrow(full_data2),0.5,2)

beta_x = as.matrix(full_data2[,c(3:(P+2))])%*%as.matrix(beta)

y_0 = alpha + beta_x +erreur1
y_1 = y_0+ mu + gamma[1]*full_data2$X_9 + gamma[2]*full_data2$X_1+ erreur2

full_data2$y_hete[full_data2$t==1] = y_1[full_data2$t==1]
full_data2$y_hete[full_data2$t==0] = y_0[full_data2$t==0]

zero <- rbinom(N,1,prob=0.3)            # set 0 to 70% of variables
full_data2$y_hete = full_data2$y_hete*zero

hist(full_data2$y_hete,breaks=50,include.lowest = T)

full_data1$y_hete =full_data2$y_hete

########################################### ENREGISTREMENT POUR PYTHON ##############################################
simulated_data <-full_data2
save=paste(path,'/../data//simulated_data.csv',sep='')
write.table(simulated_data, save, row.names=FALSE, sep=",")

########################################### AVERAGE TREATMENT EFFECT #####################################
library(pracma)
library(MASS)
require(dplyr)
full_data <- full_data1
X_2 <- full_data %>% dplyr :: select(-t,-y_hete,-inter)
y<- data.frame(full_data$y_hete)
colnames(y)<- "y"
t<- data.frame(full_data$t)
colnames(t)<- "t"
data_2 <-data.frame(y,t,X_2)

#### Estimateur 1 : SIMPLE USUAL ATE 

fit<-lm(data_2$y~data_2$t)
summary(fit)

estimateur1= fit$coefficients
sd1=summary(fit)$coefficients[2,2]


#### Estimateur 2 : Regression adjuste ATE 

fit<-lm(data_2$y~data_2$t+data_2$X_1_2+data_2$X_1_3+data_2$X_1_4+data_2$X_1_5
        +data_2$X_2_2+data_2$X_2_3+data_2$X_2_4+data_2$X_2_5
        +data_2$X_3_2+data_2$X_3_3+data_2$X_3_4+data_2$X_3_5
        +data_2$X_4_2+data_2$X_4_3+data_2$X_4_4+data_2$X_4_5
        +data_2$X_5_2+data_2$X_5_3+data_2$X_5_4+data_2$X_5_5
        +data_2$X_6_2+data_2$X_6_3+data_2$X_6_4+data_2$X_6_5
        +data_2$X_7_2+data_2$X_7_3+data_2$X_7_4+data_2$X_7_5
        +data_2$X_8_2+data_2$X_8_3+data_2$X_8_4+data_2$X_8_5
        +data_2$X_Female+data_2$X_New
)
summary(fit)

estimateur2= fit$coefficients[2]
sd2=summary(fit)$coefficients[2,2]

#### Estimateur 3 : BAYESIAN ADJUSTED ATE
m=100
P=ncol(X_2)
N=nrow(X_2)
coef_T <- matrix( rep(NA, m*(P+1)) , ncol = P+1, byrow = F)
coef_N <- matrix( rep(NA, m*(P+1)) , ncol = P+1, byrow = F)
moy_X <-matrix(rep(NA, m*(P+1)) , ncol = P+1, byrow = F)
estimateur3<- matrix(rep(NA,m),nrow=m)

#### Creation de la formule avec paste 
names =colnames(X_2)
names= paste('data_T$',names,sep='')
features_T='data_T$X_1_2'
for (i in 1:(P-1)){
  features_T = paste(features_T,names[i+1],sep='+')
}
names =colnames(X_2)
names= paste('data_N$',names,sep='')
features_N='data_N$X_1_2'
for (i in 1:(P-1)){
  features_N = paste(features_N,names[i+1],sep='+')
}
fmla_T=''
fmla_N=''
fmla_T <- as.formula(paste('data_T$y~', features_T ,sep=''))
fmla_N <- as.formula(paste('data_N$y~', features_N ,sep=''))

#### Boucle 
library(matrixStats)
for (b in 1:m){
  data <- data_2
  proba_ <- rexp(N,1)
  data <- cbind(data,proba_)
  sum_theta = sum(data$proba_)
  moy_X[b,]<-(t(as.matrix(data$proba_))%*%as.matrix(cbind(inter,data[,c(4:ncol(data)-1)])))/sum_theta
  data_T<- data %>% filter(t==1) 
  coef_T[b,] <- coefficients(lm(fmla_T,weights=data_T$proba_))
  data_N<- data %>% filter(t==0) 
  coef_N[b,] <- coefficients(lm(fmla_N,weights=data_N$proba_))
  print(b)
}
estimateur3<- matrix(rep(NA,m),nrow=m)
for ( i in 1:m){
  estimateur3[i,1]<- (moy_X[i,]%*%t((coef_T-(coef_N)))[,i])[1,1]
}
hist(estimateur3)
sd3=colSds(estimateur3)
estimateur3=colMeans(estimateur3)


##### Table 2 (3 first columns)
estimateur1[2] 
sd1
estimateur2 
sd2
estimateur3 
sd3


######################################### HETEROGENOUS TREATMENT EFFECT (PYTHON GRAPH) ###################################
#### Effet X_sex 
library(tidyr)
library(dplyr)
library(readr)
library(matlib)
require(MASS)
require(dplyr)
library(pracma)
full_data<- full_data1 %>% dplyr:: select(y_hete,t,X_Female)
full_data<- data.frame(full_data)
full_data$X_Male = 1-full_data$X_Female
colnames(full_data)=c('y_hete','t','X_Female','X_Male')
data1<-full_data[,c("y_hete","t","X_Male","X_Female")]
t1<-full_data[,c("t")]

m=1000
coef_TM <- cbind(rep(NA, m))
coef_TF <- cbind(rep(NA, m))
coef_CM <- cbind(rep(NA, m))
coef_CF <- cbind(rep(NA, m))

for (b in 1:m){
  data_loop <- data1
  data_loop <- data_loop %>% mutate(theta=rexp(nrow(data1),1))
  coef_TM[b] <- unlist(data_loop %>% filter(t==1&X_Male==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_TF[b] <- unlist(data_loop %>% filter(t==1&X_Male==0) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_CM[b] <- unlist(data_loop %>% filter(t==0&X_Male==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_CF[b] <- unlist(data_loop %>% filter(t==0&X_Male==0) %>% summarize(mean=weighted.mean(y_hete,theta)))
}

coef_plot_sex= data.frame(cbind(coef_TF-coef_CF,coef_TM-coef_CM))
colnames(coef_plot_sex) = c('sex1','sex2')

#### Effet X_spend

full_data<- full_data1 %>% dplyr:: select(y_hete,t,X_1_2,X_1_3,X_1_4,X_1_5)
full_data<- data.frame(full_data)
full_data$X_1_1 = 1-(full_data$X_1_2+full_data$X_1_3+full_data$X_1_4+full_data$X_1_5)
data1<-full_data[,c('y_hete','t','X_1_1','X_1_2','X_1_3','X_1_4','X_1_5')]

m=1000
coef_T1 <- cbind(rep(NA, m))
coef_C1 <- cbind(rep(NA, m))
coef_T2 <- cbind(rep(NA, m))
coef_C2 <- cbind(rep(NA, m))
coef_T3 <- cbind(rep(NA, m))
coef_C3 <- cbind(rep(NA, m))
coef_T4 <- cbind(rep(NA, m))
coef_C4 <- cbind(rep(NA, m))
coef_T5 <- cbind(rep(NA, m))
coef_C5 <- cbind(rep(NA, m))

for (b in 1:m){
  data_loop <- data1
  data_loop <- data_loop %>% mutate(theta=rexp(nrow(data1),1))
  coef_T1[b] <- unlist(data_loop %>% filter(t==1&X_1_1==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_T2[b] <- unlist(data_loop %>% filter(t==1&X_1_2==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_C1[b] <- unlist(data_loop %>% filter(t==0&X_1_1==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_C2[b] <- unlist(data_loop %>% filter(t==0&X_1_2==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_T3[b] <- unlist(data_loop %>% filter(t==1&X_1_3==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_T4[b] <- unlist(data_loop %>% filter(t==1&X_1_4==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_C3[b] <- unlist(data_loop %>% filter(t==0&X_1_3==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_C4[b] <- unlist(data_loop %>% filter(t==0&X_1_4==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_T5[b] <- unlist(data_loop %>% filter(t==1&X_1_5==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_C5[b] <- unlist(data_loop %>% filter(t==0&X_1_5==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
}

coef_plot_spend= data.frame(cbind(coef_T1-coef_C1,coef_T2-coef_C2,coef_T3-coef_C3,coef_T4-coef_C4,coef_T5-coef_C5))
colnames(coef_plot_spend) = c('quant1','quant2','quant3','quant4','quant5')

#### Effet X_new
full_data<- full_data1 %>% dplyr:: select(y_hete,t,X_New)
full_data<- data.frame(full_data)
full_data$X_Old = 1-full_data$X_New
colnames(full_data)=c('y_hete','t','X_new','X_old')
data1<-full_data[,c('y_hete','t','X_new','X_old')]
t1<-full_data[,c("t")]

m=1000
coef_TO <- cbind(rep(NA, m))
coef_TN <- cbind(rep(NA, m))
coef_CO <- cbind(rep(NA, m))
coef_CN <- cbind(rep(NA, m))

for (b in 1:m){
  data_loop <- data1
  data_loop <- data_loop %>% mutate(theta=rexp(nrow(data1),1))
  coef_TO[b] <- unlist(data_loop %>% filter(t==1&X_new==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_TN[b] <- unlist(data_loop %>% filter(t==1&X_new==0) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_CO[b] <- unlist(data_loop %>% filter(t==0&X_new==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
  coef_CN[b] <- unlist(data_loop %>% filter(t==0&X_new==0) %>% summarize(mean=weighted.mean(y_hete,theta)))
}

coef_plot_new= data.frame(cbind(coef_TN-coef_CN,coef_TO-coef_CO))
colnames(coef_plot_new)= c('new','old')

#### SAVE FOR PYTHON CODE 
data_boxplot= data.frame(coef_plot_spend,coef_plot_sex,coef_plot_new)
save=paste(path,'/../data//OLS_est.csv',sep='')
write.table(data_boxplot, save, row.names=FALSE, sep=",")

# 
# ############################# RUN WITH 100 000 observations only ##########################
# ######################## ANNEXES (tests for the first graphs of the paper)######################
# #### Effet X_sex
# library(tidyr)
# library(dplyr)
# library(readr)
# library(matlib)
# require(MASS)
# require(dplyr)
# library(pracma)
# full_data<- full_data1 %>% dplyr:: select(y_hete,t,X_Female)
# full_data<- data.frame(full_data)
# full_data$X_Male = 1-full_data$X_Female
# colnames(full_data)=c('y_hete','t','X_Female','X_Male')
# 
# data1<-full_data[,c("y_hete","t","X_Male","X_Female")]
# t1<-full_data[,c("t")]
# 
# m=1000
# coef_TM <- cbind(rep(NA, m))
# coef_TF <- cbind(rep(NA, m))
# coef_CM <- cbind(rep(NA, m))
# coef_CF <- cbind(rep(NA, m))
# 
# for (b in 1:m){
#   data_loop <- data1
#   data_loop <- data_loop %>% mutate(theta=rexp(nrow(data1),1))
#   coef_TM[b] <- unlist(data_loop %>% filter(t==1&X_Male==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
#   coef_TF[b] <- unlist(data_loop %>% filter(t==1&X_Male==0) %>% summarize(mean=weighted.mean(y_hete,theta)))
#   coef_CM[b] <- unlist(data_loop %>% filter(t==0&X_Male==1) %>% summarize(mean=weighted.mean(y_hete,theta)))
#   coef_CF[b] <- unlist(data_loop %>% filter(t==0&X_Male==0) %>% summarize(mean=weighted.mean(y_hete,theta)))
# }
# 
# moyenneM= mean(coef_TM) - mean(coef_CM)
# moyenneF = mean(coef_TF) - mean(coef_CF)
# var(coef_TF-coef_CF)
# 
# ### Calculate variance
# data_TM=data1 %>% filter(t==1&X_Male==1)
# data_CM= data1 %>% filter(t==0&X_Male==1)
# data_TF= data1 %>% filter(t==1&X_Male==0)
# data_CF= data1 %>% filter(t==0&X_Male==0)
# 
# library(estimatr)
# funct_var_taylor<- function(X,Y){
#   beta = coefficients(lm(Y~0+X))
#   print(1)
#   R=Diag(c(matrix(Y)-matrix(X)%*%beta))
#   print(2)
#   inverse = ginv(crossprod(matrix(X),matrix(X)))
#   print(3)
#   RR=crossprod(R,R)
#   print(4)
#   return(inverse%*%t(matrix(X))%*%RR%*%matrix(X)%*%inverse)
# }
# 
# varianceM=funct_var_taylor(data_CM$X_Male,data_CM$y_hete)+funct_var_taylor(data_TM$X_Male,data_TM$y_hete)
# varianceF=funct_var_taylor(data_TF$X_Female,data_TF$y_hete)+funct_var_taylor(data_CF$X_Female,data_CF$y_hete)
# 
# ################# Ellipse plot
# library(mixtools)
# N=1000# Number of random samples
# # Target parameters for univariate normal distributions
# rho <- cov(coef_TF-coef_CF,coef_TM-coef_CM)
# mu1 <- 0; s1 <- sqrt(varianceF)
# mu2 <- 0; s2 <- sqrt(varianceM)
# # Parameters
# mu <- c(mu1,mu2) # Mean
# sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2),2) # Covariance matrix
# #Ellipse function
# ellipse_bvn <- function(bvn, alpha){
#   Xbar <- apply(bvn,2,mean)
#   S <- cov(bvn)
#   ellipse(Xbar, S, alpha = alpha, col="red",lwd=2)
# }
# library(MASS)
# bvn <- mvrnorm(N, mu = mu, Sigma = sigma )
# var(bvn[,c(1)])
# colnames(bvn) <- c("bvn_M","bvn_F")
# coefF_centre = coef_TF-coef_CF- mean(coef_TF- coef_CF)
# coefM_centre = coef_TM-coef_CM-mean(coef_TM-coef_CM)
# plot(coefF_centre,coefM_centre,
#      xlab = 'Effect on Female',
#      ylab = 'Effect on Male'
# )
# ellipse_bvn(bvn,0.1)
# ellipse_bvn(bvn,0.01)
# ellipse_bvn(bvn,0.001)
# 
# ### Box plot
# CoefF=rep('Coef_F',1000)
# CoefM=rep('Coef_M',1000)
# coef_plot= data.frame(c(CoefF,CoefM),rbind(coef_TF-coef_CF,coef_TM-coef_CM))
# ggplot(coef_plot, aes(x=coef_plot$c.CoefF..CoefM., y=coef_plot$rbind.coef_TF...coef_CF..coef_TM...coef_CM.)) + geom_boxplot()
# 
# 
# # Graphique de la distribution de la dif des coefficients
# library(tidyr)
# library(plyr)
# 
# results= data.frame(coef_TF-coef_CF,coef_TM-coef_CM)
# colnames(results)= c("Effet-Femme","Effet-Homme")
# results_plot<- gather(results,key ="effet", value ="value")
# 
# library(ggplot2)
# mean <- ddply(results_plot, "effet", summarise, grp.mean=mean(value))
# ggplot(results_plot, aes(x=value,color=effet))+geom_density()+
#   geom_vline(data=mean, aes(xintercept=grp.mean, color=effet),
#              linetype="dashed")

