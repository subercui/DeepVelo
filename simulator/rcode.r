a <- 1
b <- 2
x <- rnorm(100, mean = 3, sd = 1)
plot(1:100, (1:100)^2, main = "plot(1:100, (1:100) ^ 2)")

mtcars0 <- mtcars
y <- 1:10
res <- lapply(1:10, rnorm)

test1 <- rnorm(100)

test2 <- function(x, y) {
    x + y
}

test2(1, 2)

data1 <- list(a = 1, b = 2)

plot(rnorm(100))
abline(h = 0, col = "blue")

library(ggplot2)
ggplot(mpg, aes(displ, hwy, colour = class)) +
    geom_point()

library(plotly)

library(shiny)
shiny::runExample("01_hello")