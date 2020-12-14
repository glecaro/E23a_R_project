setwd("C:/Users/IPAUser/Desktop/R Project/")

#devtools::install_github("rstudio/reticulate")
#install.packages('keras', type = "source") # Install the package from CRAN. 
# Note that this requires Rtools to be already installed
library(keras)
install_keras() #to setup the Keras library and TensorFlow backend
#if (!requireNamespace('BiocManager', quietly = TRUE))
#install.packages('BiocManager')
#BiocManager::install('EBImage')
library(EBImage)

#install.packages("dplyr") #download necessary package.
#install.packages("RSpectra") #download necessary package for Principal Components Analysis.
#install.packages('magick')

library(RSpectra)
library(magick)
library(dplyr)
library(here)

plt_img <- function(x){ image(x, col=grey(seq(0, 1, length=256)))}

##########################################################################################################
########################### REPLICATION: IMAGE CLASSIFIER WITH NEW DATASET ###############################
##########################################################################################################

# Note: For the replication part, we largely follow Hou, Janpu (2019), "A Simple Image Classifier with Eigenfaces 
# Link: https://rpubs.com/JanpuHou/469414

###################################
## Part I: Constructing the dataset

female <- c('Amy Adams', 'Angelina Jolie', 'Anne Hathaway', 'Emily Blunt', 'Emma Stone',
            'Jennifer Lawrence', 'Julianne Moore', 'Salma Hayek') ## To add female subjects, just add name here and re-run code
male <- c('Ben Affleck', 'Daniel Craig', 'Christian Bale', 'Dwayne Johnson', 'Keanu Reeves',
          'Matt Damon', 'Will Smith', 'Denzel Washington') ## To add male subjects, just add name here and re-run code

  for (namef in female) {
    setwd(here::here(namef))
    img.card <- sample(dir())
    cards <- list(NULL)
    for(i in 1:length(img.card))
    { 
      cards[[i]]<- readImage(img.card[i])
      cards[[i]]<- resize(cards[[i]], 64, 64)
    }
    assign(namef, cards)
  }

  for (namem in male) {
    setwd(here::here(namem))
    img.card <- sample(dir())
    cards <- list(NULL)
    for(i in 1:length(img.card))
    { 
      cards[[i]]<- readImage(img.card[i])
      cards[[i]]<- resize(cards[[i]], 64, 64)
    }
    assign(namem, cards)
  }
  rm(cards)


###############################################
## Part II: Splitting into training, testing and rotating

  celebs <- c(female, male)
  
  ## KEEP WORKING ON THIS PART
  
  train_pool <- c(`Amy Adams`[1:8], `Angelina Jolie`[1:8],
                  `Anne Hathaway`[1:8], `Emily Blunt`[1:8],
                  `Emma Stone`[1:8],`Jennifer Lawrence`[1:8],
                  `Julianne Moore`[1:8], `Salma Hayek`[1:8],
                  `Ben Affleck`[1:8], `Daniel Craig`[1:8],
                  `Christian Bale`[1:8], `Dwayne Johnson`[1:8],
                  `Keanu Reeves`[1:8], `Matt Damon`[1:8],
                  `Will Smith`[1:8], `Denzel Washington`[1:8]) 
  train <- aperm(EBImage::combine(train_pool)) %>% as.matrix.data.frame()
  
  test_pool <- c(`Amy Adams`[9:10], `Angelina Jolie`[9:10],
                 `Anne Hathaway`[9:10], `Emily Blunt`[9:10],
                 `Emma Stone`[9:10],`Jennifer Lawrence`[9:10],
                 `Julianne Moore`[9:10], `Salma Hayek`[9:10],
                 `Ben Affleck`[9:10], `Daniel Craig`[9:10],
                 `Christian Bale`[9:10], `Dwayne Johnson`[9:10],
                 `Keanu Reeves`[9:10], `Matt Damon`[9:10],
                 `Will Smith`[9:10], `Denzel Washington`[9:10])
  test <- aperm(EBImage::combine(test_pool)) %>% as.matrix.data.frame()

  
  par(mfrow=c(3,4)) # To contain all images in single frame
  for(i in 1:12){
    plot(test_pool[[i]])
  }
  par(mfrow=c(1,1)) # Reset the default

  X_df <- train
  
  b <- matrix(as.numeric(X_df[1, ]), nrow=64, byrow=T)
  plt_img(b)
  c <- t(apply(matrix(as.numeric(X_df[1, ]), nrow=64, byrow=T), 2, rev))
  plt_img(c)
  
  newdata<-NULL
  # Rotate every image and save to a new file for easy display in R
  for(i in 1:nrow(X_df))
  {
    # Rotated Image 90 degree
    c <- as.numeric((apply(matrix(as.numeric(X_df[i, ]), nrow=64, byrow=T), 2, rev)))
    # Vector containing the image
    newdata <- rbind(newdata,c)
  }
  
  df=as.data.frame(newdata)

  
#########################################
## Part III: Average faces and eigenfaces
  
  par(mfrow=c(2,2))
  par(mar=c(0.1,0.1,0.1,0.1))
  
  AV1=colMeans(data.matrix(df[1:8,]))
  plt_img(matrix(AV1,nrow=64,byrow=F))
  
  AV2=colMeans(data.matrix(df[9:16,]))
  plt_img(matrix(AV2,nrow=64,byrow=F))
  
  AV3=colMeans(data.matrix(df[17:24,]))
  plt_img(matrix(AV3,nrow=64,byrow=F))
  
  AV4=colMeans(data.matrix(df[25:32,]))
  plt_img(matrix(AV4,nrow=64,byrow=F))

  D <- data.matrix(df)
  
  ## Let's look at the average face, and need to be subtracted from all image data
  average_face=colMeans(df)
  AVF=matrix(average_face,nrow=1,byrow=F)
  plt_img(matrix(average_face,nrow=64,byrow=F))

  # Perform PCA on the data
  
  # Step 1: scale data
  # Scale as follows: mean equal to 0, stdv equal to 1
  D <- scale(D)
  
  # Step 2: calculate covariance matrix
  A <- cov(D)
  A_ <- t(D) %*% D / (nrow(D)-1)
  # Note that the two matrices are the same
  max(A - A_) # Effectively zero
  
  rm(A_)
  
  # Note: diagonal elements are variances of images, off diagonal are covariances between images
  identical(var(D[, 1]), A[1,1])
  identical(var(D[, 2]), A[2,2])
  identical(cov(D[, 1], D[, 2]), A[1,2])
  
  eigs <- eigs(A, 40, which = "LM")
  # Eigenvalues
  eigenvalues <- eigs$values
  # Eigenvectors (also called loadings or "rotation" in R prcomp function: i.e. prcomp(A)$rotation)
  eigenvectors <- eigs$vectors
  
  par(mfrow=c(1,1))
  par(mar=c(2.5,2.5,2.5,2.5))
  y=eigenvalues[1:40]
  # First 40 eigenvalues dominate
  plot(1:40, y, type="o", log = "y", main="Magnitude of the 40 biggest eigenvalues", xlab="Eigenvalue #", ylab="Magnitude")
  
  #sum(eigenvalues)/sum(eigen(A)$values) #the 40 largest eigenvalues account for approximately __ of the total variance in the dataset
  
  D_new <- D %*% eigenvectors #Principal components (AKA scores)
  
#################################
## Part IV: Plotting, projecting and reconstructing faces
  
  #Plot the first 6 eigenfaces
  
  par(mfrow=c(3,2))
  par(mar=c(0.2,0.2,0.2,0.2))
  for (i in 1:6){
    plt_img(matrix(as.numeric(eigenvectors[, i]),nrow=64,byrow=F))
  }
  
  #Projecting the photo onto the eigenvector space
  
  par(mfrow=c(2,2))
  par(mar=c(2,2,2,2))
  
  #Project first photo and reduce the dimension from 4096 to 40
  
  PF1 <- data.matrix(df[1,]) %*% eigenvectors
  barplot(PF1,main="projection coefficients in eigen space", col="green",ylim = c(-40,10))
  legend("topright", legend = "1st photo")
  
  #Reconstructing a photo from the eigenvector space
  
  par(mfrow=c(2,2))
  par(mar=c(1,1,1,1))
  
  plt_img(matrix(as.numeric(df[1, ]), nrow=64, byrow=F)) #first person on the first photo
  
  #First reconstruction with the projection onto the eigenspace
  
  PF1 <- data.matrix(df[1,]) %*% eigenvectors
  RE1 <- PF1 %*% t(eigenvectors)
  plt_img(matrix(as.numeric(RE1),nrow=64,byrow=F))
  
  
  #Add the average face
  
  par(mfrow=c(2,2))
  par(mar=c(1,1,1,1))
  
  plt_img(matrix(as.numeric(df[1, ]), nrow=64, byrow=F)) #first person on the first photo
  
  #average face
  
  average_face=colMeans(df)
  AVF=matrix(average_face,nrow=1,byrow=T)
  plt_img(matrix(average_face,nrow=64,byrow=T))
  
  #project into eigenspace and return
  
  PF1 <- data.matrix(df[1,]) %*% eigenvectors
  RE1 <- PF1 %*% t(eigenvectors)
  plt_img(matrix(as.numeric(RE1),nrow=64,byrow=F))
  
  #add the average face
  
  RE1AVF=RE1+AVF
  plt_img(matrix(as.numeric(RE1AVF),nrow=64,byrow=F))
  
  par(mfrow=c(2,2))
  par(mar=c(1,1,1,1))
  
  plt_img(matrix(as.numeric(df[31, ]), nrow=64, byrow=F)) #third person on the thirty-first photo
  
  
  #average face
  
  average_face=colMeans(df)
  AVF=matrix(average_face,nrow=1,byrow=T)
  plt_img(matrix(average_face,nrow=64,byrow=F))
  
  #project into eigenspace and return
  
  PF1 <- data.matrix(df[31,]) %*% eigenvectors
  RE1 <- PF1 %*% t(eigenvectors)
  plt_img(matrix(as.numeric(RE1),nrow=64,byrow=F))
  
  #add the average face
  
  RE1AVF=RE1+AVF
  plt_img(matrix(as.numeric(RE1AVF),nrow=64,byrow=F))
  
  
##########################################
## Part V: Classifying and matching
  
  #Classify images based on Euclidean distance
  
  PF1 <- data.matrix(df[22,]) %*% eigenvectors #test the 22nd photo, which should be the third person on our list - Anne Hathaway
  
  # Note: For testing different photos, just change number in the brackets above
  
  PFall <- data.matrix(df) %*% eigenvectors #transform photos in the dataset onto eigen space and get their coefficients
  
  #In order to avoid a negative value, find the simple difference and square it
  
  test <- matrix(rep(1,128),nrow=128,byrow=T)
  test_PF1 <- test %*% PF1
  Diff <- PFall - test_PF1
  y <- (rowSums(Diff)*rowSums(Diff))
  
  #Find the minimum number of photos to match
  
  x=c(1:128)
  newdf=data.frame(cbind(x,y))
  
  the_number = newdf$x[newdf$y == min(newdf$y)]
  
  par(mfrow=c(1,1))
  par(mar=c(1,1,1,1))
  barplot(y,main = "Similarity Plot: 0 = Most Similar")
  
  cat("the minimum number occurs at row = ", the_number) #result
  
  plt_img(matrix(as.numeric(df[the_number, ]), nrow=64, byrow=F)) #result - It is Anne Hathaway!
  
  cat("The photo matches the number #",the_number,"photo in the files") # Correctly identifies the 22nd photo out of the 128 photos, 
                                                  #by finding the match with the highest similarity score / lowest diff (0 in the plot)

  
##########################################################################################################
########################### EXTENSION: IMAGE CLASSIFIER AND PREDICTOR ####################################
##########################################################################################################

# This syntax takes the following 'Towards Data Science' article as reference: 'A Layman's Guide to Building Your First Image Classification Model in R Using Keras'
# Link: https://towardsdatascience.com/a-laymans-guide-to-building-your-first-image-classification-model-in-r-using-keras-b285deac6572

  train2 <- aperm(EBImage::combine(train_pool)) %>% as.matrix.data.frame()
  test2 <- aperm(EBImage::combine(test_pool)) %>% as.matrix.data.frame()
  
  par(mfrow=c(4,5)) # To contain all images in single frame
  for(i in 1:20){
    plot(test_pool[[i]])
  }
  par(mfrow=c(1,1)) # Reset the default
  
  #one hot encoding
  train_y<-c(rep(0,8),rep(1,8),rep(2,8),rep(3,8), rep(4,8),
             rep(5,8),rep(6,8),rep(7,8),rep(8,8))
  test_y<-c(rep(0,2),rep(1,2),rep(2,2))
  train_lab <- to_categorical(train_y) #Categorical vector for training 
  #classes
  test_lab <- to_categorical(test_y) #Categorical vector for test classes
  
  # Model Building
  model.card <- keras_model_sequential() #-Keras Model composed of a linear stack of layers
  model.card %>%                   #---------Initiate and connect to 
  #----------------------------(A)-----------------------------------#
  layer_conv_2d(filters = 40,       #----------First convoluted layer
                kernel_size = c(4,4),             #---40 Filters with dimension 4x4
                activation = 'relu',              #-with a ReLu activation function
                input_shape = c(64,64,4)) %>%   
  #----------------------------(B)-----------------------------------#
  layer_conv_2d(filters = 40,       #---------Second convoluted layer
                  kernel_size = c(4,4),             #---40 Filters with dimension 4x4
                  activation = 'relu') %>%          #-with a ReLu activation function
  #---------------------------(C)-----------------------------------#
  layer_max_pooling_2d(pool_size = c(4,4) )%>%   #--------Max Pooling
  #-----------------------------------------------------------------#
  layer_dropout(rate = 0.25) %>%   #-------------------Drop out layer
  #----------------------------(D)-----------------------------------#
  layer_conv_2d(filters = 80,      #-----------Third convoluted layer
                  kernel_size = c(4,4),            #----80 Filters with dimension 4x4
                  activation = 'relu') %>%         #--with a ReLu activation function
  #-----------------------------(E)----------------------------------#
  layer_conv_2d(filters = 80,      #----------Fourth convoluted layer
                  kernel_size = c(4,4),            #----80 Filters with dimension 4x4
                  activation = 'relu') %>%         #--with a ReLu activation function
  #-----------------------------(F)----------------------------------#
  layer_max_pooling_2d(pool_size = c(4,4)) %>%  #---------Max Pooling
  #-----------------------------------------------------------------#
  layer_dropout(rate = 0.35) %>%   #-------------------Drop out layer
  #------------------------------(G)---------------------------------#
  layer_flatten()%>%   #---Flattening the final stack of feature maps
  #------------------------------(H)---------------------------------#
  layer_dense(units = 256, activation = 'relu')%>% #-----Hidden layer
  #---------------------------(I)-----------------------------------#
  layer_dropout(rate= 0.25)%>%     #-------------------Drop-out layer
  #-----------------------------------------------------------------#
  layer_dense(units = 4, activation = 'softmax')%>% #-----Final Layer
  #----------------------------(J)-----------------------------------#
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_adam(),
          metrics = c("accuracy"))   # Compiling the architecture
  
  summary(model.card)
  
  #fit model
  history<- model.card %>% fit(train2, 
                              train_lab, 
                              epochs = 10,
                              batch_size = 8,
                              validation_split = 0.2)
  
  #Model Evaluation
  model.card %>% evaluate(train2,train_lab) #Evaluation of training set 
  pred<- model.card %>% predict_classes(train2) #-----Classification
  Train_Result<-table(Predicted = pred, Actual = train_y) #----Results
  model.card %>% evaluate(test2, test_lab) #-----Evaluation of test set
  pred1<- model.card  %>% predict_classes(test2)   #-----Classification
  Test_Result<-table(Predicted = pred1, Actual = test_y) #-----Results
  rownames(Train_Result) <- rownames(Test_Result) <- colnames(Train_Result) <- colnames(Test_Result) <- celebs
  print(Train_Result)
  print(Test_Result)
  