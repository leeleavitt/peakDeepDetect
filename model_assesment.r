#First wee need to look into each treatment foler and find
#any file that begins with exp and ends with .csv

main_dir <- getwd()
dirsToLoop <- list.dirs("./data_in_csv/", recursive = F)

onesDataSummary <- data.frame()
zerosDataSummary <- data.frame()
totalDataSummary <- data.frame()

for(i in 1:length(dirsToLoop)){
    setwd(main_dir)
    setwd(dirsToLoop[i])

    filesToAsses <- list.files(pattern = "^exp.*csv$")

    for(j in 1:length(filesToAsses)){
        experiment <- read.csv(filesToAsses[j], row.names = 1)

        #Calculate how many were mis classified
        totalMisClassified <- length(which(experiment[,1] != experiment [,2], arr.ind = T))
        totalDataSummary[i,j] <- round(totalMisClassified / dim(experiment)[1] * 100, digits = 2)

        #calculate the 1's that were mis classified
        onesExperiment <- experiment[experiment[,1] == 1, ]
        totalMisClassified <- length(which(onesExperiment[,1] != onesExperiment[,2], arr.ind = T))
        onesDataSummary[i,j] <- round(totalMisClassified / dim(onesExperiment)[1] * 100, digits = 2)

        #calculate the 0's that were mis classified
        zerosExperiment <- experiment[experiment[,1] == 0, ]
        totalMisClassified <- length(which(zerosExperiment[,1] != zerosExperiment[,2], arr.ind = T))
        zerosDataSummary[i,j] <- round(totalMisClassified / dim(zerosExperiment)[1] * 100, digits = 2)
    }
}

#Make Bar Plots
applicationNames <- Reduce(c, lapply(dirsToLoop, function(x) strsplit(x, '//')[[1]][2]))
experiemntNames <-  c('exp_1', 'exp_2', 'exp_3', 'exp_4')

colnames(onesDataSummary) <- applicationNames
row.names(onesDataSummary) <- experiemntNames

colnames(zerosDataSummary) <- applicationNames
row.names(zerosDataSummary) <- experiemntNames

colnames(totalDataSummary) <- applicationNames
row.names(totalDataSummary) <- experiemntNames

require(RColorBrewer)
par(bg = 'gray80')
cols <- brewer.pal(4, 'Dark2')
barplot(as.matrix(zerosDataSummary), beside=T, col = cols, ylim = c(0,3), main = 'Misclassified 0\'s', ylab = '% Misclassified', border=NA)
legend('topleft', experiemntNames, fill=cols, bty = 'n', border=NA)

cols <- brewer.pal(4, 'Dark2')
barplot(as.matrix(onesDataSummary), beside=T, col = cols, ylim = c(0,20), main = 'Misclassified 1\'s', ylab = '% Misclassified', border = NA)
legend('topleft', experiemntNames, fill=cols, bty = 'n', border=NA)

barplot(as.matrix(totalDataSummary), beside=T, col = cols, ylim = c(0,5), main = 'Misclassified 1\'s', ylab = '% Misclassified', border = NA)
legend('topleft', experiemntNames, fill=cols, bty = 'n', border=NA)


