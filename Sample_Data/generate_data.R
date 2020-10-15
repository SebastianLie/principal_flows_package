
###########################################################
# This function produces a data set

###########################################################


### Find the norm of vector x

norm2 <- function(x) {
return(sqrt(sum (x^2)))
}


### Set up initial parameters.

epsilon <- 0.001

para1 <- 0.9
para2 <- 0.96

### Create a data set
### data1
delta <- 0.6
gamma <- sqrt(1- delta^2 - 0.7^2)

x1 <- matrix(c(0.6,0,0.8),3,1)
x2 <- matrix(c(-0.6,0,0.8),3,1)
x3 <- matrix(c(delta,gamma,0.7) ,3,1)
x4 <- matrix(c(-delta,gamma,0.7) ,3,1)
x5 <- matrix(c(0,gamma, sqrt(1- gamma^2)),3,1)
x6 <- matrix(c(0.5,gamma,sqrt(1 - gamma^2 - 0.25)),3,1)
x7 <- matrix(c(-0.5,gamma,sqrt(1 - gamma^2 - 0.25)),3,1)
x8 <- matrix(c(0,0.95,sqrt(1-0.95^2)),3,1)
x9 <- matrix(c(0.5,0.7, sqrt(1-0.49-0.25)), 3,1)
x10 <- matrix(c(0,0,1),3,1)
x11 <- matrix(c(0.8,0, sqrt(1-0.64)),3,1)
x12 <- matrix(c(0.9,0, sqrt(1-0.81)),3,1)
x13 <- matrix(c(0.7,0.2, sqrt(0.47)),3,1)
x14 <- matrix(c(0.8,0.1,sqrt(1-0.65)),3,1)

data1 <- cbind(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14)


### Generate a data set in the circular form
### data2

data2 <- matrix(NA,3,20)

z <- 0.8
vec1 <- matrix(c(-0.3,sqrt(1-z^2 - 0.09)),2,1)
vec2 <- matrix(c(sqrt(1-z^2),0),2,1)

for (i in 1:20) {
    tmp <- i* vec1 + (20-i) * vec2
    data2[1:2,i] <- sqrt(1-z^2) * tmp/norm2(tmp)
    data2[3,i] <- z
}

### data3
data3 <- matrix(NA,3,20)
z <- 0.95
vec1 <- matrix(c(-0.3,sqrt(1-z^2 - 0.09)),2,1)
vec2 <- matrix(c(sqrt(1-z^2),0),2,1)

for (i in 1:20) {
tmp <- i* vec1 + (20-i) * vec2
data3[1:2,i] <- sqrt(1-z^2) * tmp/norm2(tmp)
data3[3,i] <- z
}

data3 <- cbind(data2,data3)

### data4
data4 <- matrix(NA, 3,50)

for (i in 1: 50) { 
data4[1,i] <- para1 * sin(i) 
data4[2,i] <- para1 * cos(i)
data4[3,i] <- sqrt(1 - para1^2)
}


### data5

data5 <- matrix(NA,3,50)
for (i in 1:50) { 
data5[1,i] <- runif(1,-0.7,0.7)
data5[2,i] <- runif(1,0,0.8)  # 0.8
data5[3,i] <- sqrt(1 - data5[1,i]^2 - data5[2,i]^2)
}

### data6

data6 <- matrix(NA,3,20)
for(i in 1:10) {

data6[1,i] <- sin(i)
data6[2,i] <- 0
data6[3,i] <- cos(i)

data6[1,i+10] <- 0.8 * sin(i+10)
data6[2,i+10] <- 0.6
data6[3,i+10] <- 0.8 * cos(i+10)
}

### data set 7
data7 <- matrix(NA,3,20)

for(i in 1:10) {
    data7[1,i] <- i/20 +0.1
    data7[2,i] <- 0.2
    data7[3,i] <- sqrt(1 - data7[1,i]^2 - data7[2,i]^2)
    
    data7[1,i+10] <- i/30 
    data7[2,i+10] <- -0.25
    data7[3,i+10] <- sqrt(1 - data7[1,i+10]^2 - data7[2,i+10]^2)
}

### data set 8

data8 <- matrix(NA,3,30)

for (i in 1:5) {
data8[1,i]<- sin(i)
data8[2,i]<- 0
data8[3,i]<- abs(cos(i))

data8[1,i+5]<- para1 * sin(i)
data8[2,i+5]<- sqrt(1-para1^2)
data8[3,i+5]<- para1 * cos(i)
}

for(i in 11:30) {
data8[,i] <- matrix(c(0,0,1),3,1)
}

#### Data set 9

data9 <- matrix(NA,3,16)

for (i in 1:8) {
    
data9[3,i] <- cos(i*pi/4)
data9[2,i] <-  0
data9[1,i] <- sin(i*pi/4)

data9[3,i+8] <- para1 * cos(i*pi/4)
data9[2,i+8] <- sqrt(1 - para1^2)
data9[1,i+8] <- para1 * sin(i*pi/4)

}

### Data set 10


data10 <- matrix(NA,3,90)
para1 <- 0.8
eps=0.2

for (j in 1:10) {
data10[1,j]<- para1 * sin(10)  + runif(1,0,eps)
data10[2,j]<- sqrt(1-para1^2) + runif(1,0,eps)
data10[3,j]<- para1 * cos(10)  + runif(1,0,eps)
}

for (j in 1:10) {
data10[,j]<- data10[,j]/norm2(data10[,j])
}

for (j in 11:80) { 
data10[1,j]<- para1 * sin(6)  + runif(1,0,eps)
data10[2,j]<- sqrt(1-para1^2) + runif(1,0,eps)
data10[3,j]<- para1 * cos(6)  + runif(1,0,eps)
}

for (j in 11:80) {
data10[,j]<- data10[,j]/norm2(data10[,j])
}

for(i in 81:90) {
data10[,i] <- matrix(c(0+runif(1,0,eps),0+runif(1,0,eps),1+runif(1,0,eps)),3,1)
}

for (j in 81:90) {
data10[,j]<- data10[,j]/norm2(data10[,j])
}

### Data set 11

data11 <- matrix(NA,3,80)
para1 <- 0.5
eps=0.9

for (j in 1:20) {
data11[1,j]<- para1 * sin(6)  + runif(1,0,eps)
data11[2,j]<- sqrt(1-para1^2) + runif(1,0,eps)
data11[3,j]<- para1 * cos(6)  + runif(1,0,eps)
}

for (j in 1:20) {
data11[,j]<- data10[,j]/norm2(data10[,j])
}

for (j in 21:40) { 
data11[1,j]<- para1 * sin(6)  + runif(1,0,eps)
data11[2,j]<- sqrt(1-para1^2) + runif(1,0,eps)
data11[3,j]<- para1 * cos(6)  + runif(1,0,eps)
}

for (j in 21:40) {
data11[,j]<- data10[,j]/norm2(data10[,j])
}

for(i in 41:80) {
data11[,i] <- matrix(c(0+runif(1,0,eps),0+runif(1,0,eps),1+runif(1,0,eps)),3,1)
}

for (j in 41:80) {
data11[,j]<- data10[,j]/norm2(data10[,j])
}


### Data set 12

r1=0.9  # ridius of small circle
r2=0.95  # ridius of small circle
r3=1  # ridius of small circle

eps=0

theta1 = asin(r1/1); #angel between v and vector connecting points on the cicle 
theta2 = asin(r2/1); #angel between v and vector connecting points on the cicle 
theta3 = asin(r3/1); 
 
phi = seq(0,2*pi,length=30); # sample equal distant phi on the circle
 
 
data12 = matrix(0,3,90);
 
 
for (i in 1:30)
 {
   # get cart coordinates
   data12[1,i]=sin(theta1)*cos(phi[i])+runif(1,0,eps);
   data12[2,i]=sin(theta1)*sin(phi[i])+runif(1,0,eps);
   data12[3,i]=cos(theta1)+runif(1,0,eps);
   
   data12[1,i+30]=sin(theta2)*cos(phi[i])+runif(1,0,eps);
   data12[2,i+30]=sin(theta2)*sin(phi[i])+runif(1,0,eps);
   data12[3,i+30]=cos(theta2)+runif(1,0,eps);

   data12[1,i+60]=sin(theta3)*cos(phi[i])+runif(1,0,eps);
   data12[2,i+60]=sin(theta3)*sin(phi[i])+runif(1,0,eps);
   data12[3,i+60]=cos(theta3)+runif(1,0,eps);
   
 }    

# Data set 13
num_points <- 40
data13 <- matrix(NA,3,num_points + 1)
for(i in 1: (num_points + 1)) {
	data13[1,i] <-(i - (num_points/2)) /(num_points)
  data13[2,i] <-sin(4 * data13[1,i])/2
  data13[3,i] <-sqrt(1 - data13[1,i]^2 - data13[2,i]^2)
}


# Data set 14
num_points <- 200
eps<-0.3
data14 = matrix(0,3,num_points);
 

for (i in 1:num_points){
z<-runif(1,-0.6,-0.4)
phi<-runif(1,0,pi)
theta<-asin(z/1)
data14[1,i]<-1*cos(theta)*cos(phi)+runif(1,0,eps)
data14[2,i]<-sin(theta)*sin(phi)+runif(1,0,eps)
data14[3,i]<-z+runif(1,0,eps)
data14[,i]<-data14[,i]/norm2(data14[,i])

}


#####

### Data set 15

num_points <- 40

data14 <- matrix(NA,3,num_points + 1)



for(i in 1:(num_points+1)){
data14[1,i]<-(i-(num_points/2))/(num_points)
data14[2,i] <- sin(4.5*data14[1,i])/2 # data14[2,i] <- sin(6*data14[1,i])/2
data14[3,i] <-sqrt(1-data14[1,i]^2-data14[2,i]^2)
}

num_points <- 40
data15 <- matrix(NA,3,num_points + 1)



for(i in 1:(num_points+1)){
data15[1,i]<-(i-(num_points/2))/(num_points)
data15[2,i] <- sin(5*data15[1,i])/2  # data14[2,i] <- sin(6*data14[1,i])/2
data15[3,i] <-sqrt(1-data15[1,i]^2-data15[2,i]^2)
}







### A list of all data sets created.





alldatasets <- list(data1,data2,data3,data4,data5,data,data7,data8,data9,data10,data11, data12, data14, data15)



###  Choose data set.

generate_data <- function(n) {

return(alldatasets[[n]])
}









