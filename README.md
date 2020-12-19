# Math E23a R Project: Simple Image Classifier using Eigenfaces and Principal Component Analysis
## Authors: Lecaro, Gabriela; Pinto, Carlos Eduardo

### This is the ReadMe file for the Harvard Extension School Math E23a Linear Algebra and Real Analysis R project.

Video Narration: http://bit.ly/PintoLecaroRProject}{http://bit.ly/PintoLecaroRProject
GitHub Repository: https://github.com/glecaro/E23a_R_project}{https://github.com/glecaro/E23a\_R\_project

Instructions to run the script:

1. To run our script, the following 5 packages are necessary, please download if you do not have them already (instructions can be found in the setup section right at the top): RSpectra, magick, dplyr, keras, EBImage.

2. Please download our dataset by using the following \hyperlink{https://drive.google.com/drive/folders/1CCCsZ2gQmQE8OIwd0Nm0vXrWRs6od85l?usp=sharing}{link}

3. Then, simply run the lines in the script (eigenfaces\_main.Rmd) the order in which they are presented and “voilà“, amazing things will happen.

Outline of which points we believe we have earned:

1.1 The topic is related to linear algebra, real analysis, or multivariate calculus:
- Our topic is intrinsically related to linear algebra as it is largely based and dependent on eigenvectors.


1.2 The script is commented well enough to be reasonably self-explanatory:
- We have included cohesive comments throughout the script to explain it.


1.3 The script generates at least one nice-looking graphic:
- Our script generates nice-looking graphics on almost all parts of the script through the plt\_image function


1.4 The assignment is submitted on-time and includes all required materials, including a video narration that both (all) team members participated in creating:\newline
- The assignment will be submitted on time and the following is the link to our video narration of the code: http://bit.ly/PintoLecaroRProject


2.1 The project is related to a topic that you have studied in another course this term:
- This project is related to the concept of mean, which Carlos is studying in an AP Statistics course, as the code computes an “average face” to make the image classification possible in Parts 2 and 3 of the Replication side.


2.2 Used an R function that has not appeared in any Math 23 lecture script:
- The R functions that we used and that have not appeared in any Math 23 lecture script can be found in the set up section and throughout the script.


2.3 Uses a random-number function and a for loop:
- We used random sampling without replacement in the first part, and nested for loops throughout the script.


2.4 Incorporates ideas both from linear algebra and from real analysis:
- The main idea is related to linear algebra we are incorporating is the eigenvector and we are also incorporating the idea of Euclidean distance from real analysis in Part 5.


2.5 Defines and uses at least two functions:
- We have created streamlined functions (that perform multiple, consecutive steps) by using magrittr (tidyverse package) operators and pipes and create a function to plot images in lines, all throughout the script.


2.6 Uses a permutation test, bootstrap, or other similar statistical technique:
- We have used sample function for sampling without replacement and an array transposition function to permute the matrix dimensions to be able to exploit our small sample size in parts 1 and 3.


2.7 Includes two related but distinct topics:
- This project performs Principal Component Analysis which is a dimensionality-reduction technique widely used in Data Science and Statistics, and based on eigen-decomposition for multivariate analysis in Part 3. We also use resampling and covariance matrices, which are topics from Statistics.


2.8 Delves into a new library/R package:
- Some of the new new library/R package we delved into in this project include RSpectra, magick, and EBImage.


2.9 Professional-looking software engineering:
- We have included commentary throughout the code and used the R package Magrittr, which allows us to favor more intuitive, readable syntax.


2.10 Integrating well-written LaTeX with R code in a final write-up via R markdown (2 points):
- Our R project is coded in an Rmd file, so as to integrate both R and Markdown syntax, and use knitr to produce a pdf output.


2.11 Having a 2-person team:
- Our team is composed of two students.
