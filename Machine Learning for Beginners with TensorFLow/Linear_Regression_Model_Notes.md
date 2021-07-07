# Linear Regression (What we will solve with ML)
## What is Linear Regression
* Linear regression is essentially fitting a line through data to minimize average distance between all points and the line
* Usually corresponds with a line of best fit
* Can be used for predictive models
* We will be focusing on simple linear regression which only takes into account one dpendent and one independent variable

## How does Linear regression work?
* Start by plotting data on a graph, typically use a scatter plot
* Draw a line through the data that runs roughly through the middle of the points to give an even number above and below the line
* We can get the error by totalling the absolute value of the distances between each points y value and the line at that point
* We adjust the line by changing the y intercept and the slope so as to minimize the error

## What is Gradient Descent
* Optimization algorithm to find the minimum of a function
* We use this with Linear Regression to find the minimum error
* When we use a Gradient Descent optimizer function, it will change slope and y intercept so as to get the optimal line through the data
* Generally we see the error decrease sharply at fist then begin to slow down so slope decreases over time

## Remember y=mxb
* Y 
    * Dependent Variable
    * Our Outputs
* X 
    * Independent Variable
    * Our Inputs
* M 
    * Slope
    * Our weighted values
* B 
    * Y intercept
    * Our bias values

## Our goal
* Program the function to find the minimal error

# Linear Regression Model Part 1

## Part 1- Gathering Data
* Create some data for our model
    * Use numpy linspace to generate some points
* Plot the points on a graph
* 
