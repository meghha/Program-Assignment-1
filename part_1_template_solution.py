# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        nu.scale(Xtrain)
        nu.scale(Xtest)
        nu.integerCheck_(ytrain)
        nu.integerCheck_(ytest)

        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}

        # Enter your code and fill the `answer` dictionary
        print("Length of X train: ", len(Xtrain))
        print("Length of X test: ", len(Xtest))
        print("Length of y train: ", len(ytrain))
        print("Length of y test: ", len(ytest))
        print("Max value of X train: ", np.max(Xtrain))
        print("Max value of X test: ", np.max(Xtest))

        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ): 
        # Enter your code and fill the `answer` dictionary
        # X, y, Xtest, ytest = u.prepare_data()
        # Xtrain, ytrain = u.filter_out_7_9s(X, y)
        # Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        # Xtrain = nu.scale_data(Xtrain)
        # Xtest = nu.scale_data(Xtest)

        answer = {}

        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5)
        metrics = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv)
        answer["clf"] = clf  # the estimator (classifier instance)
        answer["cv"] =  cv # the cross validator instance
        
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        scores = nu.scores_(metrics=metrics)

        answer["scores"] = scores

        print("The mean of accuracy scores for Decision Tree classifier is: ", scores['mean_accuracy'])
        print("The STD of accuracy scores for Decision Tree classifier is: ", scores['std_accuracy'])

        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        # X, y, Xtest, ytest = u.prepare_data()
        # Xtrain, ytrain = u.filter_out_7_9s(X, y)
        # Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
 
        # Xtrain = nu.scale_data(X)
        # 

        answer = {}
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5,random_state=self.seed)
        metrics = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv)
        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'
        
        scores = nu.scores_(metrics=metrics)

        answer = {}
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = scores
        answer["explain_kfold_vs_shuffle_split"] = """Both are resampling techniques. 
                                                      K fold cv divides the data into k folds, and runs the training k times, with k-1 folds as training data, and k-th split as test data. 
                                                      Random shuffle split works by shuffling and splitting the data randomly k times, into train and test. 
                                                      One of the disadvantages of Shuffle split over K-fold validation, is that, if the data is very less, and there class imbalance, Shuffle split can lead to overfitting of the majority class.
                                                      Advantages of shuffle split over k-fold is that it is less computationally intensive."""
        
        print("The mean of accuracy scores for Decision Tree classifier with shuffle split is: ", scores['mean_accuracy'])
        print("The STD of accuracy scores for Decision Tree classifier with shuffle split is: ", scores['std_accuracy'])

        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`
        # X, y, Xtest, ytest = u.prepare_data()
        # Xtrain, ytrain = u.filter_out_7_9s(X, y)
        # Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
    
        # Xtrain = nu.scale_data(X)
        # Xtest = nu.scale_data(X)

        answer = {}

        # Enter your code, construct the `answer` dictionary, and return it.
        clf = DecisionTreeClassifier(random_state=self.seed)

        for k in [2,5,8,16]:
            cv = ShuffleSplit(n_splits=k,random_state=self.seed)
            metrics = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv)
            scores = nu.scores_(metrics=metrics)
            answer[k] = {"scores": scores, "cv":cv, "clf": clf}
             
            print("For k={} the mean accuracy score is {}".format(k,scores["mean_accuracy"]))
        
        
        print("Nothing significant is observed for the mean/standard deviation of accuracy for each k")
        
        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        
        """ """
        # X, y, Xtest, ytest = u.prepare_data()
        # Xtrain, ytrain = u.filter_out_7_9s(X, y)
        # Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        # Xtrain = nu.scale_data(Xtrain)
        # Xtest = nu.scale_data(Xtest)

        answer_D = self.partD(X=X,y=y)
        clf_RF = RandomForestClassifier(random_state=self.seed)
        # clf_DT = DecisionTreeClassifier(random_state=self.seed)
        clf_DT = answer_D["clf"]

        # cv = ShuffleSplit(random_state=self.seed)
        cv = answer_D["cv"]

        # metrics_DT = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf_DT,cv=cv)
        metrics_RF = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf_RF,cv=cv)
        metrics_DT = answer_D["scores"]

        answer = {}
        # Enter your code, construct the `answer` dictionary, and return it.

        answer["clf_RF"] = clf_RF
        answer["clf_DT"] = clf_DT
        answer["cv"] = cv


        answer["scores_RF"] = nu.scores_(metrics_RF)
        answer["scores_DT"] = metrics_DT
        answer["model_highest_accuracy"] = np.where(answer["scores_RF"]["mean_accuracy"]>answer["scores_DT"]["mean_accuracy"],"Random-Forest","Decision-Tree")
        
        answer["model_lowest_variance"] = np.where(answer["scores_RF"]["std_accuracy"]>answer["scores_DT"]["std_accuracy"],"Random-Forest","Decision-Tree")
        answer["model_fastest"] = np.where(answer["scores_RF"]["mean_fit_time"]>answer["scores_DT"]["mean_fit_time"],"Random-Forest","Decision-Tree")

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float) **
            "model_fastest" (float) **
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """
    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        # X, y, Xtest, ytest = u.prepare_data()
        # Xtrain, ytrain = u.filter_out_7_9s(X, y)
        # Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        # Xtrain = nu.scale_data(Xtrain)
        # Xtest = nu.scale_data(Xtest)

        answer_F = self.partF(X,y)
        clf = answer_F["clf_RF"]
        cv  = answer_F["cv"]

    
        # clf = RandomForestClassifier(random_state=self.seed)
        # cv = ShuffleSplit(random_state=self.seed)
        

        answer = {}
        answer['clf'] = clf
        answer['default_parameters'] = clf.get_params()
        
        grid_search, best_estimator, mean_accuracy_cv = nu.gridSearch_(clf,cv,X,y)
        answer["best_estimator"] = best_estimator
        answer['grid_search'] = grid_search
        answer["mean_accuracy_cv"] = mean_accuracy_cv

        clf.fit(X,y)

        ytrain_pred_orig = clf.predict(X)
        ytest_pred_orig = clf.predict(Xtest)

        answer['confusion_matrix_train_orig'] = confusion_matrix(y,ytrain_pred_orig)
        answer['confusion_matrix_test_orig'] = confusion_matrix(ytest,ytest_pred_orig)
        confusion_matrix_train_orig = confusion_matrix(y,ytrain_pred_orig)
        confusion_matrix_test_orig = confusion_matrix(ytest,ytest_pred_orig)

        
        best_model = grid_search.best_estimator_
        best_model.fit(X, y)

        ytrain_pred_best = best_model.predict(X)
        ytest_pred_best = best_model.predict(Xtest)
        answer['confusion_matrix_train_best'] = confusion_matrix(y,ytrain_pred_best)
        answer['confusion_matrix_test_best'] = confusion_matrix(ytest,ytest_pred_best)
        confusion_matrix_train_best = confusion_matrix(y,ytrain_pred_best)
        confusion_matrix_test_best = confusion_matrix(ytest,ytest_pred_best)

        answer["accuracy_orig_full_training"] = (confusion_matrix_train_orig.diagonal().sum()) / (confusion_matrix_train_orig.sum())
        answer["accuracy_best_full_training"] = (confusion_matrix_train_best.diagonal().sum()) / (confusion_matrix_train_best.sum())
        answer["accuracy_orig_full_testing"] = (confusion_matrix_test_orig.diagonal().sum()) / (confusion_matrix_test_orig.sum())
        answer["accuracy_best_full_testing"] = (confusion_matrix_test_best.diagonal().sum()) / (confusion_matrix_test_best.sum())
        
        print("Mean Train accuracy during cross validation is: ", mean_accuracy_cv)
        print("Mean Train accuracy for full training using grid search is: ", (confusion_matrix_train_best.diagonal().sum()) / (confusion_matrix_train_best.sum()))
        print("Mean Test accuracy for full training using grid search is: ", (confusion_matrix_test_best.diagonal().sum()) / (confusion_matrix_test_best.sum()))
        
        print("As compared to the training accuracy, the test accruacy reduces slightly. When compared to the cross validation score, both the train and test accuracy is higher.")
        # Enter your code, construct the `answer` dictionary, and return it.

        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        return answer

