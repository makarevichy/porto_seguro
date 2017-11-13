library(data.table)
library(caret)
library(xgboost)
library(verification)

# Read train and test data
dtrain <- fread('train.csv')
dtest <- fread('test.csv')

# Check data size in memory
print("Training data size in RAM:");
print(object.size(dtrain), units = 'Mb')

# print training data dimensions
print(dim(dtrain))

# new variables

dtrain[,`:=`(v = round(ps_car_12 ^ 2 * 10000),
             mileage = ps_car_13 ^ 2 * 48400,
             year = round(14 - ps_car_15 ^ 2))]
dtrain[,`:=`(mil_per_year = mileage / year)][mil_per_year == -Inf, mil_per_year := 0]
dtrain[,`:=`(mileage = mileage ^ (-1 / 4.5),
             mil_per_year = mil_per_year ^ (-1 / 4.5))][mil_per_year == Inf, mil_per_year := 0]
cat_v <- data.table(id = dtrain$id, sjmisc::split_var(dtrain, v, n = 10))
dtrain <- dtrain[cat_v, on = .(id)]
cat_to <- dtrain[,.(cat_to_doub = mean(target)), by = .(ps_car_11_cat)]
dtrain <- dtrain[cat_to, on = .(ps_car_11_cat)][, 'ps_car_11_cat' := NULL]

dtest[,`:=`(v = round(ps_car_12 ^ 2 * 10000),
            mileage = ps_car_13 ^ 2 * 48400,
            year = round(14 - ps_car_15 ^ 2))]
dtest[,`:=`(mil_per_year = mileage / year)][mil_per_year == -Inf, mil_per_year := 0]
dtest[,`:=`(mileage = mileage ^ (-1 / 4.5),
            mil_per_year = mil_per_year ^ (-1 / 4.5))][mil_per_year == Inf, mil_per_year := 0]
cat_v_test <- data.table(id = dtest$id, sjmisc::split_var(dtest, v, n = 10))
dtest <- dtest[cat_v_test, on = .(id)]
dtest <- dtest[cat_to, on = .(ps_car_11_cat)][, 'ps_car_11_cat' := NULL]

# Missing value -----------------------------------------------------------

x <- dtrain[,sapply(.SD, function(x) sum(x == -1) > 0)]
names_na <- names(x[x==TRUE])

dtrain[ps_ind_02_cat == -1, ps_ind_02_cat := NA]
dtrain[ps_ind_04_cat == -1, ps_ind_04_cat := NA]
dtrain[ps_ind_05_cat == -1, ps_ind_05_cat := NA]
dtrain[ps_reg_03 == -1, ps_reg_03 := NA]
dtrain[ps_car_02_cat == -1, ps_car_02_cat := NA]
dtrain[ps_car_03_cat == -1, ps_car_03_cat := NA]
dtrain[ps_car_05_cat == -1, ps_car_05_cat := NA]
dtrain[ps_car_07_cat == -1, ps_car_07_cat := NA]
dtrain[ps_car_09_cat == -1, ps_car_09_cat := NA]
dtrain[ps_car_11 == -1, ps_car_11 := NA]
dtrain[ps_car_12 == -1, ps_car_12 := NA]
dtrain[ps_car_14 == -1, ps_car_14 := NA]

# Preprocess missing value

mis <- preProcess(dtrain[,3:ncol(dtrain)], method = 'knnImpute')
d_train <- predict(mis, dtrain[1:3000,3:ncol(dtrain)])

# collect names of all categorical variables
cat_vars <- names(dtrain)[grepl('_cat$', names(dtrain))]

# turn categorical features into factors
dtrain[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtest[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]


# one hot encode the factor levels
dtrain <- as.data.frame(model.matrix(~. - 1, data = dtrain))
dtest <- as.data.frame(model.matrix(~ . - 1, data = dtest))

# create index for train/test split
train_index <- sample(c(TRUE, FALSE), size = nrow(dtrain), replace = TRUE, prob = c(0.8, 0.2))

# perform x/y ,train/test split.
x_train <- dtrain[train_index, 3:ncol(dtrain)]
y_train <- as.factor(dtrain$target[train_index])

x_test <- dtrain[!train_index, 3:ncol(dtrain)]
y_test <- as.factor(dtrain$target[!train_index])

# Convert target factor levels to 0 = "No" and 1 = "Yes" to avoid this error when predicting class probs:
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
levels(y_train) <- c("No", "Yes")
levels(y_test) <- c("No", "Yes")

# normalized gini function taked from:
# https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

# create the normalized gini summary function to pass into caret
giniSummary <- function (data, lev = "Yes", model = NULL) {
  levels(data$obs) <- c('0', '1')
  out <- normalizedGini(as.numeric(levels(data$obs))[data$obs], data[, lev[2]])  
  names(out) <- "NormalizedGini"
  out
}

# create the training control object. Two-fold CV to keep the execution time under the kaggle
# limit. You can up this as your compute resources allow. 
trControl = trainControl(
  method = 'cv',
  number = 5,
  summaryFunction = giniSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE)

# create the tuning grid. Again keeping this small to avoid exceeding kernel memory limits.
# You can expand as your compute resources allow. 
tuneGridXGB <- expand.grid(
  nrounds=c(350),
  max_depth = c(4),
  eta = c(0.03),
  gamma = c(0.01),
  colsample_bytree = c(0.75),
  subsample = c(0.5),
  min_child_weight = c(2))

start <- Sys.time()

# train the xgboost learner
xgbmod <- train(
  x = x_train,
  y = y_train,
  method = 'xgbTree',
  metric = 'NormalizedGini',
  trControl = trControl,
  tuneGrid = tuneGridXGB)


print(Sys.time() - start)

# make predictions
preds <- predict(xgbmod, newdata = x_test, type = "prob")
preds_final <- predict(xgbmod, newdata = dtest, type = "prob")


# convert test target values back to numeric for gini and roc.plot functions
levels(y_test) <- c("0", "1")
y_test_raw <- as.numeric(levels(y_test))[y_test]

# Diagnostics
print(xgbmod$results)
print(xgbmod$resample)

# plot results (useful for larger tuning grids)
plot(xgbmod)

# score the predictions against test data
normalizedGini(y_test_raw, preds$Yes)

# plot the ROC curve
roc.plot(y_test_raw, preds$Yes, plot.thres = c(0.02, 0.03, 0.04, 0.05))

# prep the predictions for submissions
sub <- data.frame(id = as.integer(dtest$id), target = preds_final$Yes)

# write to csv
write.csv(sub, 'sub_xgb_second.csv', row.names = FALSE)
