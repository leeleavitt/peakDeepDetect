#Make a function to plot the Traces
gridPlotter <- function(traces, traceNames, fileName){


    #Calculate the number of squares to have in the plot
    gridDim <- ceiling(sqrt(length(traceNames)))
    graphics.off()
    #make a new dev
    #dev.new(width=20,height=20)
    fileName = paste0(fileName,'.png')
    png(fileName, width=960, height=960)
    par(mfrow=c(gridDim, gridDim), mai=c(0,0,0,0), bg='black')

    #Plot Each trace!
    for(i in 1:length(traceNames)){
        par(bg='black', mai=c(0,0,0,0))
        plot(as.numeric(traces[traceNames[i],]), ylim=c(-0.1, 1.1), col='white', lwd=1, type='l')
        box('plot', lty='solid', col='white')
    }
    dev.off()

}

# Here I am going to visualize the misclassified traces with R.
super_dir<-getwd()

#Go to the Response folder
setwd("./data_in_csv")
main_dir <- getwd() #where i will hangout

expDirs <- list.dirs(recursive = F)

for(i in 1:length(expDirs)){
    setwd(main_dir)
    setwd(expDirs[i])

    labsOrig <- read.csv('labels.csv', row.names=1) # original labels
    labsComp <- read.csv('lstm.experiment2_labcomp.csv', row.names=1) # Labels from predictive model
    traces <- read.csv('traces.csv', row.names=1) # Traces

    #Proof they are the same
    #identical(labsOrig[row.names(labscomp),], labscomp[,1])

    #Find the examples where the predicted doesn't equal the original
    mismatchLogic <- labsComp[,1] != labsComp[,2]
    misMatched <- labsComp[mismatchLogic,]

    # How many are 0's and how many are 1's
    cat('\nThis is the Run Down of your mismatched in \n',expDirs[i], '\nOut of', dim(traces)[1], 'Traces\n')
    print(summary(as.factor(misMatched[,1])))

    # Find traces that were 0 and classified as 1's
    logicZeros <- misMatched[,1] == 0
    zeroNames <- row.names(misMatched[logicZeros,])
    # Find traces that were 1 and classified as 0's
    logicOnes <- misMatched[,1] == 1
    oneNames <- row.names(misMatched[logicOnes,])

    #Plot the zeros as Ones
    gridPlotter(traces, zeroNames, 'lstmExp2ZerosAsOnes')
    #Plot the Ones as Zeros
    gridPlotter(traces, oneNames, 'lstmExp2OneAsZeros')

}


