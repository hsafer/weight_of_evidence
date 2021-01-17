# weight_of_evidence

**Weight of evidence (WOE)** is used to assess the relative risk of the values taken by a categorical feature. For example, suppose that the goal is to predict the probability of default on a loan. One feature might be home ownership, with values "own outright," "own with mortgage," "rent," "live with parents," and "other." 

A feature like this, with unordered, non-numeric values, cannot be used directly with most machine-learning methods. WOE addresses this challenge by converting the values to numbers. The number assigned to a value, say "own with mortgage," is the relative risk of an applicant in the "own with mortgage" group paying off the loan. Specifically, WOE is the log odds of the categorical value being associated with a negative target value.

When WOE of a categorical value is positive, the probability of the loan being paid in full is above average among all applicants, and vice versa when WOE is negative. 

WOE has advantages over the one-hot encoding that is commonly used for categorical features. WOE avoid the proliferation of features that occurs when one-hot encoding is used with categorical features that take many values. WOE also does not lose information the way that one-hot encoding does when tree-based algorithms select a subset of features at each node.

