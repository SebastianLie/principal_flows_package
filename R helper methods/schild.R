distsq <- function(x,y){
  
  return(sum((x-y)^2))
  
}

### Find the norm of vector x

norm2 <- function(x) {
  
  return(sqrt(sum (x^2)))
  
}

### The xyangle function used to find the angle

### between two vectors x and y

xyangle <- function(x,y) {
  
  normx <- norm2(x)
  
  normy <- norm2(y)
  
  normxy <- sum(x*y)
  
  xy <- normxy/(normx * normy)
  
  if (abs(xy) > 1) {xy <- round(xy)}
  
  return(acos(xy))
  
}

### Exponential map from tangent space to the sphere

expmapsph <- function(x,v) {
  
  normv <- norm2(v)
  
  normx <- norm2(x)
  
  alpha = normv/normx
  
  if (alpha == 0) { y <- x }
  
  else { y <- cos(alpha) * x + sin(alpha) * v /alpha }
  
  return(y)
  
}

### Log map (inverse of Exponential map) from

### sphere to tangent space

logmapsph <- function(x,y) {
  
  alpha <- xyangle(x,y)
  
  if (alpha == 0) { v <- rep(0, length(x))}
  
  else { v <- (y - cos(alpha) * x) * alpha / sin(alpha) }
  
  return(v)
  
}

data1 = alldatasets[[1]]
p = data1[,1]

for (i in (1:14)) {
  X = cbind()
}



schild<-function(A0,A1,v){
  v0<-v/norm2(v)
  X0<-expmapsph(A0,v0)
  G<-logmapsph(A0,A1)
  dist.g<-norm2(G)
  step.up<-0.1
  N<-ceiling(dist.g/step.up)+1
  step<-dist.g/N
  e<-step*G/norm2(G)
  A<-matrix(NA,3,(N+1))
  A[ ,1]<-A0
  for(i in 1:N){
    A[ ,i+1]<-expmapsph(A[ ,i],e)
  }
  print(A)
  #A[ ,N+1]=A1
  X<-A
  X[ ,1]<-X0
  for(j in 1:N){
    t1<-logmapsph(A[ ,j+1],X[,j])
    P<-expmapsph(A[ ,j+1],0.5*t1)
    t2<-logmapsph(A[,j],P)
    X[ ,j+1]<-expmapsph(A[,j],2*t2)
  }
  print(X)
  res<-logmapsph(A1,X[ ,N+1])
  return(res)
}

schild.general<-function(data_curve,v){
  l<-ncol(data_curve)
  v0<-v/norm2(v)
  V<-data_curve
  V[ ,1]<-v0
  for(i in 2:l){
    V[ ,i]<-schild(data_curve[ ,i-1],data_curve[ ,i],V[ ,i-1])
  }
  return(V)
}

p = c(0, 0,  0.99961464)
u = c(0,  0.14242402,  0.91922261)
v = c(-0.42068611, -0.1, -0.0274965)

print(schild(p, u, v))




