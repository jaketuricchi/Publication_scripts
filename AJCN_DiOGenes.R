###################
# DioGenes script #
###################
# Aim: to test the effect of proportionate changes in body composition during weight loss on (a) weight regain after 8 weeks
# and (b) changes in appetite (by visual analouge scale) during the study.

#accompanies the publication of the manuscript in The American Journal of Clinical Nutrition (AJCN) found at:
# https://academic.oup.com/ajcn/article/111/3/536/5707675 


library('dplyr')
library('plyr')
library('ggplot2')
library('tidyverse')
library('car')
library('magrittr')
library('qwraps2')
library('psych')
library('data.table')
library('stargazer')
library('reshape2')
library('ggpubr')
library('multcomp')
library('tableone')
library('jtools')
library(gridExtra)
library('tableone')
library(stringr)
options(scipen=999)

#set wd
#setwd("~/Dropbox/PhD/Diogenes/New Data")
setwd("C:/Users/jaket/Dropbox/PhD/Diogenes/New Data")

#load data
dt<-read.csv("diog_adult_v02_data.csv")

#select vars relevant for the primary analysis
BCdata<-subset(dt, select=c(partner,family, member, aci10650, aci10640, aci10550, aci10540, aci20710, aci20700, 
                   aci20610, aci20600,
                   aci20660, aci30650, aci30560, aci30550, aci40740, aci40730, aci40640, aci40630, 
                   aci10353, aci20370, aci30350, aci40400, dfmet1, dfmet2, dfmet3, dfmet4,
                   dfm1, dfm2, dfm3, dfm4, dffm1, dffm2, dffm3, dffm4, dfp1,dfp2,dfp3,dfp4,
                   qdp0010, sc0950, sc0250, alcd0020, sc0370))

#rename from the variable codes
colnames(BCdata)<-c("partner", "family", "member", "DXAFM1", "DXAFFM1", "BIAFM1", "BIAFFM1", "DXAFM2", "DXAFFM2", "BIAFM2", "BIAFFM2",
                    "DXAFM3", "DXAFFM3", "BIAFM3", "BIAFFM3", "DXAFM4", "DXAFFM4", "BIAFM4", "BIAFFM4", 
                    "CID1weight", "CID2weight", "CID3weight", "CID4weight", "BCmethod1", "BCmethod2", "BCmethod3", "BCmethod4",
                    "BCFM1", "BCFM2", "BCFM3", "BCFM4", "BCFFM1", "BCFFM2", "BCFFM3", "BCFFM4", "BCFMP1", "BCFMP2",
                    "BCFMP3", "BCFMP4","gender", "BMI", "age", "centre", "race")

## Create unique identifier
BCdata$ID<- with(BCdata, paste0("X", partner,".", family,".", member))

#some background exploration was done on these variables to test which we should use. Lets now filter to what we need
df<-subset(BCdata, select=c(ID, DXAFM1, DXAFFM1, DXAFM2, DXAFFM2,
                            CID1weight, CID2weight, CID3weight,
                            gender, BMI, age, centre))

#filter out those without the necessary data for the primary analysis
df<-na.omit(df) #leaves us with n=219

#calculate weight changes
df$WL<-df$CID2weight-df$CID1weight #weight loss during the 8-week diet
df$WRG26<-df$CID3weight-df$CID2weight #weight regain during the 26 week follow-up

#generate the primary independent variable to test: percentage of weight lost as FFM (%FFML)
df<-df%>%mutate(PFFML=((DXAFFM2-DXAFFM1)/WL)*100, BF1=(DXAFM1/CID1weight)*100,
                FFML=DXAFFM2-DXAFFM1, FML=DXAFM2-DXAFM1)%>%
  filter(PFFML<75 & PFFML>-30) #remove supraphysiological outliers
df<-df%>%mutate(gender=as.factor(gender))

# load second df which contains other variables
df_psych<-read.csv("DIOGenes_James_data.csv", na.strings=c(" ","NA"))
df_psych$ID<- with(df_psych, paste0("X", partner,".", family,".", member))

#for now all we need is trial arm
arm<-subset(df_psych, select = c(ID, randiet))
colnames(arm)[2]<-'arm'
df<-merge(df, arm, by='ID')

###############
### EDA #######
###############

# produce descriptive tables
# in all ppts
descriptives_all<-CreateTableOne(vars=c('age', 'centre', 'arm', 'CID1weight', 'DXAFM1', 'DXAFFM1', 
                                        'BF1', 'WL', 'PFFML', 'WRG26'), data=df)
print(descriptives_all)

# by gender
descriptives_gender<-CreateTableOne(vars=c('age', 'centre', 'arm', 'CID1weight', 'DXAFM1', 'DXAFFM1', 
                                        'BF1', 'WL', 'PFFML', 'WRG26'), strata='gender',data=df)
print(descriptives_gender)

# consider the distribution of key variables
qqPlot(df$WL)
qqPlot(df$FFML)
qqPlot(df$WRG26)

# correlation of key variables
library(corrplot)
cor_df<-subset(df, select = c(age, CID1weight, DXAFFM1, DXAFM1, BF1, WL, FFML, FML, PFFML, WRG26))
cor_key<-cor(cor_df) #correlation table for key variables (supplementary materials)
corrplot.mixed(cor_key)

#correlations per gender#
#men
cor_df_men<-subset(df, select = c(age, CID1weight, DXAFFM1, DXAFM1, BF1, WL, FFML, FML, PFFML, WRG26, gender))%>%
  filter(gender==1)%>%dplyr::select(-gender)
cor_key_men<-cor(cor_df_men) #correlation table for key variables (supplementary materials)
corrplot.mixed(cor_key_men)

#women
cor_df_women<-subset(df, select = c(age, CID1weight, DXAFFM1, DXAFM1, BF1, WL, FFML, FML, PFFML, WRG26, gender))%>%
  filter(gender==2)%>%dplyr::select(-gender)
cor_key_women<-cor(cor_df_women) #correlation table for key variables (supplewomentary materials)
corrplot.mixed(cor_key_women)

#EDA - Association between baseline body fat and %FFML (testing Forbes rule)
ggplot(df, aes(x=BF1, y=PFFML))+geom_point(aes(col=gender))+geom_smooth(method='lm', se=F, aes(col=gender))+
  theme_light()+labs(title='Association between baseline body fat and change in body composition',
                     x='Baseline bodyfat (%)', y='%FFML') #supplementary materials figure

ggplot(df, aes(x=DXAFM1, y=PFFML))+geom_point(aes(col=gender))+geom_smooth(method='lm', se=F, aes(col=gender))+
  theme_light()+labs(title='Association between baseline fat mass and change in body composition',
                     x='Baseline fat mass (kg)', y='%FFML') #note the difference in direction using % vs kg FM.

ggplot(df, aes(x=PFFML, y=WL))+geom_point(aes(col=gender))+geom_smooth(method='lm', se=F, aes(col=gender))+
  theme_light()+labs(title='Association between weight loss and change in body composition',
                     x='%FFML', y='Weight loss (kg)') #%FFML increases with WL.

#what was the WL and WRG across different centres and trial arms?
WL_arm_gg<-ggplot(df, aes(x=arm, y=WL, col=gender))+geom_point()+
  theme_light()+labs(title='Association between trial arm and WL',
                     x='Trial Arm', y='Weight loss (kg)')+
  geom_boxplot()

WL_centre_gg<-ggplot(df, aes(x=centre, y=WL, col=gender))+geom_point()+
  theme_light()+labs(title='Association between trial arm and WL',
                     x='Trial Arm', y='Weight loss (kg)')+
  geom_boxplot()

WRG_arm_gg<-ggplot(df, aes(x=arm, y=WRG26, col=gender))+geom_point()+
  theme_light()+labs(title='Association between trial arm and WL',
                     x='Trial Arm', y='Weight loss (kg)')+
  geom_boxplot()

WRG_centre_gg<-ggplot(df, aes(x=arm, y=WRG26, col=gender))+geom_point()+
  theme_light()+labs(title='Association between trial arm and WL',
                     x='Trial Arm', y='Weight loss (kg)')+
  geom_boxplot()

grid.arrange(WL_arm_gg, WL_centre_gg, WRG_arm_gg, WRG_centre_gg)




#plot the main association.
ggplot(df, aes(x=FFML, y=WRG26))+geom_point()+geom_smooth(method='lm', se=F)+
  theme_light()+labs(title='Association between percentage of weight lost as FFM and weight regain at 26 weeks',
                     x='%Fat free mass loss', y='Weight regain 26 weeeks (kg)')+
  geom_point(aes(color=gender, group=gender))+geom_smooth(aes(color=gender, group=gender), method='lm', se=F)
#evidence of a positive association - main manuscript figure


#####################
# analysis ##########
####################

# univariate linear regressions for each key variable predicting weight change at 26 wk
#we'll do these in all and by gender

#function to get and organise results from lms
extract_univar_results<-function(y){
  sum<-data.frame(t(summary(y)$coefficients[2,])) #extract main results
  r2<-as.data.frame(summary(y)$r.squared) #get R2
  sum<-bind_cols(sum, r2) 
  sum$predictor<-as.character(summary(y)$terms[[3]]) #get the predictor
  colnames(sum)<-c('Beta', 'Error', 't_value', 'p_value', 'r_squared', 'predictor')
  sum<-sum[,c(6, 1:5)]
  return(sum)
}
  
#run over each predictor seperately for men and women
uni_lm<-list()
univariate_lms<-function(x){
  uni_lm[[1]]<-lm(WRG26~age, data=x)
  uni_lm[[2]]<-lm(WRG26~CID1weight, data=x)
  uni_lm[[3]]<-lm(WRG26~DXAFFM1, data=x)
  uni_lm[[4]]<-lm(WRG26~DXAFM1, data=x)
  uni_lm[[5]]<-lm(WRG26~WL, data=x)
  uni_lm[[6]]<-lm(WRG26~PFFML, data=x)
  univariate_results<-lapply(uni_lm, extract_univar_results)%>%bind_rows()%>%
    mutate(gender=x$gender[1])
  return(univariate_results)
}

#Run, bind, round.
univariate_results_gender<-dlply(df, 'gender', univariate_lms)%>%bind_rows() #by gender
univariate_results_both<-univariate_lms(df)%>%mutate(gender='both')

univariate_results_all<-bind_rows(univariate_results_both, univariate_results_gender)
univariate_results_all[2:6]<-lapply(univariate_results_all[2:6], round, digits=3)

# multivariate models
df$centre<-as.factor(df$centre)
multivariate_lms<-list()

summary(multivariate_lms[[1]]<-lm(WRG26 ~ centre+arm+CID1weight+BF1+WL+PFFML, data=df)) #all ppts
summary(multivariate_lms[[2]]<-lm(WRG26 ~ centre+arm+CID1weight+BF1+WL+PFFML, data=filter(df, gender==1)))
summary(multivariate_lms[[3]]<-lm(WRG26 ~ centre+arm+CID1weight+BF1+WL+PFFML, data=filter(df, gender==2)))

y<-multivariate_lms[[2]]
extract_multivar_results<-function(y){
  sum<-data.frame(summary(y)$coefficients) #extract main results
  sum$predictor<-rownames(sum)
  colnames(sum)<-c('Beta', 'Error', 't_value', 'p_value', 'predictor')
  sum<-sum[,c(5, 1:4)]
  return(sum)
}

#results of multivaraite model
multivariate_results_all<-lapply(multivariate_lms, extract_multivar_results)%>%bind_rows()

#main result: %FFML predicts WRG in men but not women (with a statistical tendency in the full group)


######################
#     analysis 2     #
######################
#Aim 2: consider the association between %FFML and changes in appetite (by visual analouge scale)

#get VAS vars
vas<-subset(df_psych, select=c(ID, dvas11, dvas12, dvas13, dvas14,  dvas21, dvas22, dvas23, dvas24))

colnames(vas)<-c('ID', "hungerVAS1", "fullnessVAS1", "desireVAS1", "prospectiveVAS1", 
                 "hungerVAS2", "fullnessVAS2", "desireVAS2", "prospectiveVAS2")
vas<-vas[-1,]
vas<-na.omit(vas)
vas[2:9]<-lapply(vas[2:9], as.numeric)

#calculate change scores
vas$hunger_change<-vas$hungerVAS2-vas$hungerVAS1
vas$fullness_change<-vas$fullnessVAS2-vas$fullnessVAS1
vas$desire_change<-vas$desireVAS2-vas$desireVAS1
vas$prospective_change<-vas$prospectiveVAS2-vas$prospectiveVAS1

#merge in with main df
df2<-merge(df, vas, by='ID') #leaves us with only n=40 for secondary aim


##############
#   EDA     ##
##############
#how does appetite change over time?
vas_long<-vas%>%melt(id.var='ID')%>%subset(ID %in% df2$ID)%>%
  mutate(appetite_measure=as.factor(ifelse(grepl('hunger', variable), 'hunger',
                                 ifelse(grepl('fullness', variable), 'fullness',
                                        ifelse(grepl('desire', variable), 'desire',
                                               ifelse(grepl('prospective', variable), 'prospective',NA))))),
         CID=as.factor(ifelse(grepl('1', variable), 1,
                              ifelse(grepl('2', variable), 2, 'change'))))

ggplot(subset(vas_long, CID !='change'), aes(x=appetite_measure, y=value, fill=CID))+geom_boxplot()+
  theme_light() #here we see that appetite decreases from CID1 to CID2

#summarise vas numerically to report by table in manuscript
vas_results<-vas_long%>%group_by(appetite_measure, CID)%>%
  dplyr::summarise(mean_score=mean(value), sd_score=sd(value))
  

vas_wide<-reshape(vas_long, dir='w', timevar = 'appetite_measure', idvar=c('ID', 'CID'))
colnames(vas_wide)<-str_replace(colnames(vas_wide), 'value.', '')
vas_means<-CreateTableOne(vars= c('hunger', 'fullness', 'desire', 'prospective'), strata=c('CID'), data=vas_wide)
print(vas_means) 

#write.csv(vas_means, 'vas_means_by_cid.csv', row.names=F)


## EDA - linear associations

#bind pffml to appetite measures
vas_long2<-merge(vas_long, subset(subset(df2, select=c(ID, PFFML, gender))), by='ID')

#in all
ggplot(subset(vas_long2,CID=='change'), aes(x=PFFML, y=value, color=appetite_measure))+
  geom_point()+geom_smooth(method='lm', se=F)+theme_light()

#by gender
ggplot(subset(vas_long2,CID=='change'), aes(x=PFFML, y=value, color=appetite_measure))+
  geom_point()+geom_smooth(method='lm', se=F)+theme_light()+
  facet_grid(.~gender, scales='free')

#here we see associations between increased %FFML and general increase in appetite in favour of the primary hypothesis.

#these are all the materials needed for the primary manuscript.

