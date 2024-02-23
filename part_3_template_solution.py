import numpy as np
from numpy.typing import NDArray
from typing import Any
import new_utils as nu
import utils as u
from sklearn import svm
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary


        answer = {}

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """
 
        clf = LogisticRegression(max_iter=300, random_state=42, solver='saga')
        clf.fit(Xtrain, ytrain)
        k_values = [1, 2, 3, 4, 5]

        # Calculate top k scores
        train_scores, test_scores = nu.calculate_top_k_scores(clf, Xtrain, ytrain, Xtest, ytest, k_values)

        # Plot k vs score for both training and testing data


        # Plot for training data
        # nu.plot_k_vs_score(train_scores, "Training Data")

        # Plot for testing data
        # nu.plot_k_vs_score(test_scores, "Testing Data")

        # Return the results
        plot_k_vs_score_train = train_scores
        plot_k_vs_score_test = test_scores

        answer[1] = {}
        answer[2] = {}
        answer[3] = {}
        answer[4] = {}
        answer[5] = {}
        answer[1]["score_train"] = train_scores[0][1]
        answer[1]["score_test"] = test_scores[0][1]
        answer[2]["score_train"] = train_scores[1][1]
        answer[2]["score_test"] = test_scores[1][1]
        answer[3]["score_train"] = train_scores[2][1]
        answer[3]["score_test"] = test_scores[2][1]
        answer[4]["score_train"] = train_scores[3][1]
        answer[4]["score_test"] = test_scores[3][1]
        answer[5]["score_train"] = train_scores[4][1]
        answer[5]["score_test"] = test_scores[4][1]
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = plot_k_vs_score_train
        answer["plot_k_vs_score_test"]  = plot_k_vs_score_test
        answer["text_rate_accuracy_change"] = "The accuracy of the test data rises rapidly, as the value of k increases. It also plateaus after about k=3"
        answer["text_is_topk_useful_and_why"] = "The top k accuracy is useful because In multi-class classification problems where the number of classes is large, traditional accuracy might not provide a clear picture of how well the model is performing. Top-k accuracy allows for a more comprehensive evaluation by considering a range of possible correct predictions."

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        Xtrain, ytrain = nu.filter_out_7_9s(X, y)
        Xtest, ytest = nu.replace_7_9s(Xtest, ytest)

        # Xtrain = nu.scale_data(Xtrain)
        # Xtest = nu.scale_data(Xtest)

        answer = {}

        # Answer is a dictionary with the same keys as part 1.B
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

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary

        # Xtrain, ytrain = nu.filter_out_7_9s(X, y)
        # Xtest, ytest = nu.replace_7_9s(Xtest, ytest)

        # Xtrain = nu.scale_data(Xtrain)
        # Xtest = nu.scale_data(Xtest)

        clf = svm.SVC(kernel="linear",random_state=self.seed)
        cv = StratifiedKFold(n_splits=5)

        scorers = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'F1': make_scorer(f1_score, average='macro')
            }
        # Perform cross-validation with stratified k-fold
        metrics = {}
        for key, value in scorers.items():
            metrics[key]=cross_validate(clf, X, y, cv=cv,scoring=value)['test_score']
        

        scores = nu.scores_part3(metrics)

        clf.fit(X, y)

        ytrain_pred = clf.predict(X)
        ytest_pred = clf.predict(Xtest)

        answer = {}
        answer["scores"] = scores
        answer["cv"] = cv
        answer["clf"] = clf
        answer["is_precision_higher_than_recall"] = True if scores["mean_precision"]>scores["mean_recall"] else False
        answer["explain_is_precision_higher_than_recall"] = "Precision being higher than recall typically occurs when the model has a higher tendency to make fewer false positive predictions while potentially missing some true positive predictions. This scenario is commonly encountered in situations where the cost of false positives is higher than false negatives, or when the dataset is imbalanced."
        answer["confusion_matrix_train"] = confusion_matrix(y,ytrain_pred)
        answer["confusion_matrix_test"] = confusion_matrix(ytest,ytest_pred)


        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  
    Print out the class weights, and comment on the performance difference. 
    Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Xtrain, ytrain = nu.filter_out_7_9s(X, y)
        # Xtest, ytest = nu.replace_7_9s(Xtest, ytest)

        Xtrain = X.copy()
        ytrain = y.copy()

        # Compute class weights
        class_labels = np.unique(ytrain)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(ytrain), y=ytrain)

        # Create dictionary of class weights
        class_weight_dict = {class_label: weight for class_label, weight in zip(class_labels, class_weights)}

        # Print class weights
        for class_label, class_weight in class_weight_dict.items():
            print(f"Class {class_label}: Weight {class_weight}")

        clf = svm.SVC(kernel="linear",random_state=self.seed,class_weight=class_weight_dict)
        cv = StratifiedKFold(n_splits=5)

        scorers = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'F1': make_scorer(f1_score, average='macro')
            }
        # Perform cross-validation with stratified k-fold
        metrics = {}
        for key, value in scorers.items():
            metrics[key]=cross_validate(clf, Xtrain, ytrain, cv=cv,scoring=value)['test_score']
        

        scores = nu.scores_part3(metrics)

        clf.fit(Xtrain, ytrain)

        ytrain_pred = clf.predict(Xtrain)
        ytest_pred = clf.predict(Xtest)

        answer = {}
        
        answer["scores"] = scores
        answer["cv"] = cv
        answer["clf"] = clf
        answer["class_weights"] = class_weight_dict
        answer["is_precision_higher_than_recall"] = True if scores["mean_precision"]>scores["mean_recall"] else False
        answer["explain_is_precision_higher_than_recall"] = "Precision being higher than recall typically occurs when the model has a higher tendency to make fewer false positive predictions while potentially missing some true positive predictions. This scenario is commonly encountered in situations where the cost of false positives is higher than false negatives, or when the dataset is imbalanced."
        answer["confusion_matrix_train"] = confusion_matrix(ytrain,ytrain_pred)
        answer["confusion_matrix_test"] = confusion_matrix(ytest,ytest_pred)
        answer["explain_purpose_of_class_weights"] = "The purpose of using class weights in SVM functions is to give more importance to the minority class during the training process. This is achieved by assigning higher weights to misclassifications of the minority class compared to the majority class. By doing so, the SVM algorithm learns to optimize the margin with respect to both classes, not just the majority class."
        answer["explain_performance_difference"] = "The accuracy has improved vastly. This time the recall is higher than precision. This is because of the class beiing balanced."



        # Enter your code and fill the `answer` dictionary
       



        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
