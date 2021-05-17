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


