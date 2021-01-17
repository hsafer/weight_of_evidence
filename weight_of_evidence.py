# Weight of Evidence
# Hershel Safer <hsafer@alum.mit.edu>

# Todo
# - Consider returning dataframe rather than array. Keep the index values and column names.
# - It seems that running WOE on a column of a dataframe somehow changes the shape of the column
#   to be (n_instances, 1) rather than (n_instances,). I don't understand how that happens or what
#   it means, but I think that I've seen that in practice. Reset the column by doing
#   df.my_col = df.my_col.values[:, 0]. I don't get this at all.
# - Can I get around the issues in defining an sklearn transformer by using
#   preprocessing.FunctionTransformer?
# - Add feature_names parameter to fit(). Use to label WOE plots and printed output.
# - Add a method to create a printable summary for each feature: A dataframe with woe and
#   info_val_contrib per category, and overall info_val with description (e.g., weak, ok, good).
# - Allow continuous input and add code to do binning.
#   If variable is continuous, check if WoE of binned version is monotonically increasing or
#   decreasing, or perhaps has a single peak or valley.
#   make_categorical() should verify that the bin edges are unique, otherwise pd.qcut() will fail.
#   Or else trap the ValueError that this raises.
#   See pd.cut() example in Pandas documentation > Categorical Data > Object Creation
#   Perhaps use KBinsDiscretizer, as described in sklearn documentation 4.3.5.
#   Perhaps allow the user to set the breaks that define the bins.
#   Add other binning methods, e.g., the same number of goods in each group or with specified cut
#   points.
# - Consider using for Population Stability Index (PSI). Perhaps implement underlying code for WOE
#   and PSI in a way that is similar to how GradientBoostingClassifier and GradientBoostingRegressor
#   are based on BaseGradientBoosting.

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class WeightOfEvidence(BaseEstimator, TransformerMixin):
    """Convert categories to values of relative risk using Weight of Evidence.

    Weight of evidence (WOE) is used to assess the relative risk of each value that a feature takes.
    WOE is computed separately for each feature.

    Consider a feature (characteristic) x with multiple possible values (attributes or categories,
    indexed by $j$) and a binary classification y of the same length as x. Suppose that the sample
    has $n$ negative instances and $p$ positive instances, with $n_j$ and $p_j$ being the numbers of
    negative and positive instances in category $j$. The WOE of category $j$ is:
    $ln[ (n_j / n) / (p_j / p) ]$, i.e.,
    $ln[(fraction of negative instances in j-th category) /
        (fraction of positive instances in j-th category)]$.
    For a detailed explanation, see, e.g., the book "The Credit Scoring Toolkit" by Raymond Anderson
    (Oxford University Press, 2007), p. 192.
    Kim Larsen's post on the Stitch Fix Algorithms blog discusses this but with some non-standard
    aspects (http://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/).

    Although the term "attribute" is often used to refer to a specific value of a characteristic,
    the term "category" is used here to avoid confusion with Python object attributes.

    In practice, calculate WOE as [ln(n_j / p_j) - ln(n / p)]. This should be more stable than the
    calculation as defined above, as the latter divides small numbers.

    If any count $n_j$ or $p_j$ is zero, the WOE is undefined for the corresponding category $j$.
    Change the 0 counts to 1; this small change to the data causes WOE to be (approximately) defined
    for all categories. If the other count for a category is so small that changing the 0 to 1 makes
    a substantial difference, then the category was not particularly informative anyhow.

    Parameters
    ----------
    good_class, bad_class : scalar, defaults 0 (good_class), 1 (bad_class)
        The values that represent the negative ("good") and positive ("bad") classes

    is_continuous : boolean, False by default
        If True, feature values are continuous.
        If False, feature values are categorical

    n_bins : int, default 10
        Number of bins for binning continuous features

    min_bin_size : int, default 5
        Minimum number of items in each bin for binning continuous features

    allow_unseen_categories : boolean, default True
        Determines what transform() should do when it is asked to transform a value that was not
        used to fit the transformation. If True, then such categories are assigned a WOE value of
        0. If False, then a ValueError is raised.

    Attributes
    ----------
    n_features_ : int
        Number of input features.

    woe_ : list of defaultdict, length n_features
        The i-th defaultdict is for feature i, and it maps each category name of feature i to the
        category's WOE value. If allow_unseen_categories is True, then the default value is 0, and
        it is used when trying to transform() an unseen category. Otherwise, trying to transform()
        an unseen category raises an error.

    info_val_contrib_ : list of dict, length n_features
        The i-th dict is for feature i, and it maps each category name of feature i to the
        category's contribution to the information value of feature i.

    info_val_ : list of float, length n_features
        The information value of each feature.

    categories_ : list of lists, length n_features
        The i-th list is for feature i, and contains the names of valid categories for feature i,
        i.e., the categories that were seen in the data that were used to fit the transform. A
        feature's list is sorted if the feature's categories are types that can be sorted together.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from code.ml_python.utils.weight_of_evidence import WeightOfEvidence
    >>>
    >>> # Set parameters for generating sample data.
    >>> n_samples = 100
    >>> n_features = 3
    >>> prob_good = 0.8
    >>>
    >>> good_class = 'pif'
    >>> bad_class = 'default'
    >>>
    >>> # Generate the target.
    >>> y = np.random.default_rng(9).choice([good_class, bad_class], size=n_samples,
    ...                                     p=[prob_good, 1 - prob_good])
    >>>
    >>> # Generate data to fit the transform (n_samples x n_features).
    >>> fit_data = [
    ...     ['cat{:02}'.format(int(x))
    ...      for x in np.random.default_rng(feature_num + 10).exponential(
    ...          scale=(1 + feature_num / 2), size=n_samples)]
    ...     for feature_num in range(n_features)]
    >>> x_fit = np.array(list(zip(*fit_data)))
    >>>
    >>> # Create and fit a WOE instance.
    >>> w = WeightOfEvidence(good_class=good_class, bad_class=bad_class)
    >>> w.fit(x_fit, y)
    WeightOfEvidence(allow_unseen_categories=True, bad_class='default',
             good_class='pif', is_continuous=False, min_bin_size=5, n_bins=10)
    >>>
    >>> # Print info about the fitted instance.
    >>> for feature_num in range(n_features):
    ...     print('feature {}'.format(feature_num))
    ...     print(pd.DataFrame({'woe': w.get_woe()[feature_num],
    ...                         'info_val_contrib': w.get_info_val_contrib()[feature_num]}))
    ...     print('\ninfo_val={:0.4}\n'.format(w.get_info_val()[feature_num]))
    ...
    feature 0
                woe  info_val_contrib
    cat00 -0.160193          0.018443
    cat01  1.415970          0.245473
    cat02  0.263291          0.004406
    cat03 -0.835322          0.026279
    cat04 -1.528469          0.066500

    info_val=0.3611

    feature 1
                woe  info_val_contrib
    cat00 -0.105949          0.005639
    cat01  0.800292          0.115448
    cat02  0.011834          0.000025
    cat03 -0.904456          0.031669
    cat04 -0.211309          0.002368
    cat06 -1.597603          0.074958

    info_val=0.2301

    feature 2
                woe  info_val_contrib
    cat00  0.383540          0.042689
    cat01 -0.075035          0.001550
    cat02 -0.362717          0.021018
    cat03  0.417441          0.010298
    cat04  0.011976          0.000007
    cat05  0.417441          0.010298
    cat06 -0.275706          0.003164
    cat07 -1.374318          0.048886
    cat10 -1.374318          0.048886

    info_val=0.1868

    >>> # Create new data to use with transform(). Include the categories from the fitting
    >>> # data + a missing category (instead of the NaN values).
    >>> x_new = pd.DataFrame({'feature_{}'.format(feature_num):
    ...                       pd.Series(np.unique(x_fit[:, feature_num]))
    ...                       for feature_num in range(n_features)})
    >>> x_new[x_new.isna()] = 'missing'
    >>>
    >>> # Look at the new data.
    >>> print(x_new)
      feature_0 feature_1 feature_2
    0     cat00     cat00     cat00
    1     cat01     cat01     cat01
    2     cat02     cat02     cat02
    3     cat03     cat03     cat03
    4     cat04     cat04     cat04
    5   missing     cat06     cat05
    6   missing   missing     cat06
    7   missing   missing     cat07
    8   missing   missing     cat10
    >>>
    >>> # Transform the new data using the fitted WOE instance.
    >>> print(pd.DataFrame(w.transform(x_new)))
              0          1          2
    0 -0.160193  -0.105949    0.38354
    1   1.41597   0.800292 -0.0750352
    2  0.263291  0.0118345  -0.362717
    3 -0.835322  -0.904456   0.417441
    4  -1.52847  -0.211309  0.0119762
    5         0    -1.5976   0.417441
    6         0          0  -0.275706
    7         0          0   -1.37432
    8         0          0   -1.37432

    Notes
    -----
    X should not contain any NaN values. If it does, replace them with a non-NaN value that
    indicates missing values, such as "missing."
    """

    def __init__(self, good_class=0, bad_class=1, is_continuous=False, n_bins=10, min_bin_size=5,
                 allow_unseen_categories=True):
        self.good_class = good_class
        self.bad_class = bad_class
        self.is_continuous = is_continuous
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.allow_unseen_categories = allow_unseen_categories

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""
        if self.good_class == self.bad_class:
            msg = 'Good class and bad class must be different, but they are both {}.'.format(
                self.good_class)
            raise ValueError(msg)

        if self.is_continuous:
            if self.n_bins < 1:
                raise ValueError('n_bins must be at least 1, but was {}'.format(self.n_bins))
            if self.min_bin_size < 1:
                raise ValueError('min_bin_size must be at least 1, but was {}'.format(
                    self.min_bin_size))

    def _validate_y(self, y):
        """Check validity of y values and raise ValueError if not valid.

        Parameters
        ----------
        y : array-like, shape (n_samples)
            Class values corresponding to X.

        Returns
        -------
        y
        """
        # Make sure that y contains only the values good_class and bad_class.
        invalid_classes = set(y) - {self.good_class, self.bad_class}
        if len(invalid_classes) > 0:
            msg = ('y should include only good_class ({}) and bad_class ({}), '
                   'but it includes other values: {}.'.format(
                    self.good_class, self.bad_class, list(invalid_classes)))
            raise ValueError(msg)
        return y

    def _init_state(self):
        """Initialize model state. """
        self.woe_ = [None] * self.n_features_
        self.info_val_contrib_ = [None] * self.n_features_
        self.info_val_ = [None] * self.n_features_
        self.categories_ = [None] * self.n_features_

    def fit(self, X, y):
        """Fit the transform to training data.

        Parameters
        ----------
        X : array-like, shape [n_samples]
            The data used to compute the relative risk of each category.

        y : array-like, shape (n_samples)
            Class values corresponding to X.

        Returns
        -------
        self : object
        """
        # Check input.
        X, y = check_X_y(X, y, dtype=None, ensure_2d=False, estimator='Weight of Evidence')

        # Make sure that X is 2D. If it's a 1D array, then add a second dimension of length 1.
        # This enables consistent subscripting in the call to pd.crosstab().
        if X.ndim == 1:
            X.shape = (-1, 1)

        self._check_params()
        y = self._validate_y(y)

        n_samples, self.n_features_ = X.shape
        self._init_state()

        # Calculate WOE separately for each feature.
        for feature_num in range(self.n_features_):
            # Count the frequency of each (category, label) combination.
            counts = pd.crosstab(X[:, feature_num], y, dropna=False, margins=False)

            # crosstab() sorts columns by column name. Re-order so that the good class is first.
            # The straightforward way to do this is:
            #     counts = counts.loc[:, [self.good_class, self.bad_class]]
            # If one class is the Boolean value False, however, then this would omit the
            # corresponding column. The following code circumvents the problem.
            if counts.columns[0] != self.good_class:
                counts = counts.iloc[:, [1, 0]]

            # Fix up naming in the counts dataframe.
            counts.rename_axis('category', axis='index', inplace=True)
            counts.columns = ('n_good', 'n_bad')

            # check_X_y() above screens out NaN values in numeric input. It will not catch NaN
            # values in input with mixed types. Catch them here by using the fact that crosstab()
            # does not include a row with counts for the NaN category.
            # For some kinds of input, such as lists and ndarrays, np.nan values will be converted
            # to the string 'nan', I think in check_X_y. These will simply remain as a category with
            # that string as the name. This may cause problems if transform() is given np.nan as an
            # input to transform. That's the user's fault for including NaN values in the input.
            n_counts_from_X = counts.sum().sum()
            if n_counts_from_X < n_samples:
                raise ValueError('X contains NaN values in feature number {}, '
                                 'but it should not.'.format(feature_num))

            # Change 0 counts to 1. See explanation in class docstring.
            counts[counts == 0] = 1

            # Compute WOE.
            col_sums = counts.sum(axis=0)  # Calculate this after replacing zeros with ones.
            population_log_odds = np.log(col_sums.n_good / col_sums.n_bad)
            category_log_odds = counts.n_good.div(counts.n_bad).map(np.log)
            woe = category_log_odds - population_log_odds

            if self.allow_unseen_categories:
                # Accessing the value of an unseen category returns 0.
                self.woe_[feature_num] = woe.to_dict(defaultdict(int))
            else:
                # Accessing the value of an unseen category raises a KeyError.
                self.woe_[feature_num] = woe.to_dict(defaultdict(None))

            try:
                self.categories_[feature_num] = sorted(woe.index.tolist())
            except TypeError:
                # The categories include types that cannot be sorted together, so don't sort.
                self.categories_[feature_num] = woe.index.tolist()

            # Compute information value.
            info_val_contrib = ((col_sums.n_bad * counts.n_good - col_sums.n_good * counts.n_bad) /
                                (col_sums.n_good * col_sums.n_bad)) * woe
            self.info_val_contrib_[feature_num] = info_val_contrib.to_dict()
            self.info_val_[feature_num] = info_val_contrib.sum()

        return self

    def transform(self, X):
        """Apply the weight of evidence transformation.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_transformed : array-like, shape [n_samples, n_features]
            The transformed data, with each value of X replaced by its WOE score.
        """
        check_is_fitted(self, 'n_features_')
        X = check_array(X, dtype=None, ensure_2d=False)

        # Make sure that X is 2D. If it's a 1D array, then add a second dimension of length 1.
        # This enables consistent subscripting in the list comprehension.
        if X.ndim == 1:
            X.shape = (-1, 1)

        # Check that X has the correct number of features.
        n_samples_to_transform, n_features_to_transform = X.shape
        if n_features_to_transform != self.n_features_:
            raise ValueError('X has {} features, but the transform was fit to {} features.'.format(
                n_features_to_transform, self.n_features_))

        # For each feature, replace each input category with its WOE value.
        # Trap invalid categories if self.allow_unseen_categories is False.
        X_transformed = np.empty_like(X)  # Pre-allocate array of same shape as X.
        for feature_num in range(self.n_features_):
            try:
                transformed_values_for_feature = (
                    [self.woe_[feature_num][xij] for xij in X[:, feature_num]])
                X_transformed[:, feature_num] = transformed_values_for_feature
            except KeyError:
                categories_to_be_transformed = set(X[:, feature_num])
                invalid_categories = (categories_to_be_transformed -
                                      set(self.categories_[feature_num]))
                raise ValueError('In X, feature {} contains categories that were not used to fit '
                                 'the WOE transform.\n'
                                 'Valid categories: {}\n'
                                 'Invalid categories that were passed: {}'.format(
                                    feature_num, self.categories_, list(invalid_categories)))

        return X_transformed

    def get_woe(self):
        """Return the weight of evidence of each category within each feature."""
        return self.woe_

    def get_info_val_contrib(self):
        """Return the contribution to the information value of each category within each feature."""
        return self.info_val_contrib_

    def get_info_val(self):
        """Return the information value of each feature."""
        return self.info_val_

    def get_categories(self):
        """Return a list of the categories for each feature."""
        return self.categories_

    def plot_woe(self, bar_color='lightblue', zero_line_color='darkblue'):
        """Create a barplot for each feature showing the WOE of each of its categories."""
        plt.figure(figsize=(10, 6 * self.n_features_))
        for feature_num in range(self.n_features_):
            n_categories = len(self.categories_[feature_num])
            # Cannot just use self.woe_[feature_num].keys() here. If transform() was called with
            # unseen categories, then those categories will have been added to self.woe_ and they
            # will appear in the barplot.
            woe_for_valid_categories = pd.Series({category: self.woe_[feature_num][category]
                                                  for category in self.categories_[feature_num]})
            plt.subplot(self.n_features_, 1, (feature_num + 1))
            ax = woe_for_valid_categories.sort_values(ascending=True).plot.barh(color=bar_color)
            ax.set_title('Feature {}: Weight of evidence'.format(str(feature_num)))
            ax.set_xlabel('Weight of evidence (from high risk to low risk)')
            plt.plot([0, 0], [-1, n_categories], color=zero_line_color)

    # --------------------------------------------------------------------------------
    # Old code for binning.

    # def _make_categorical(self):
    #     """Put feature values into categorical pd.Series. Bin values if they are continuous."""
    #     if self._is_continuous:
    #         # Check that we have enough features values that this is worth doing.
    #         min_number_of_vals_needed = self._n_bins * self._min_bin_size
    #         if len(self._feature_vals) < min_number_of_vals_needed:
    #             msg = ('The feature has {} values. To split into {} bins of minimum size {}, use '
    #                    'at least {} values.'.format(
    #                         len(self._feature_vals), self._n_bins, self._min_bin_size,
    #                         min_number_of_vals_needed))
    #             raise ValueError(msg)
    #
    #         # Bin the data.
    #         binned_feature_vals, bins = pd.qcut(self._feature_vals, self._n_bins, retbins=True)
    #
    #         # Change the labels of the first and last bins so that they include potential future
    #         # values that are outside the range of the input values.
    #         new_category_names = list(binned_feature_vals.cat.categories)
    #         new_category_names[0] = re.sub('^\S+\s+', '<=', new_category_names[0])
    #         new_category_names[0] = re.sub(']$', '', new_category_names[0])
    #         new_category_names[-1] = re.sub(',\s.*$', '', new_category_names[-1])
    #         new_category_names[-1] = re.sub('^[([]', '>', new_category_names[-1])
    #         binned_feature_vals.cat.categories = new_category_names
    #
    #         # Save the info that we need for later
    #         self._feature_vals = binned_feature_vals
    #         self._bins = bins
    #     else:  # Discrete data
    #         # This automatically puts categories into lexical order, AFAICT.
    #         self._feature_vals = self._feature_vals.astype('category')
    #
    # def _find_bins(self, test_feature_vals):
    #     """Find bins corresponding to continuous feature values.
    #
    #     :param test_feature_vals: pd.Series of continuous feature values whose bins are to be found
    #     :return: pd.Series of bins corresponding to feature values
    #     """
    #     # Use with map(na_action='ignore') or else NaN values will get mapped to the final category.
    #     def f(x):
    #         """Find bin for a single feature value."""
    #         for i in range(1, self._bins.size - 2):
    #             if x <= self._bins[i]:
    #                 return self._feature_vals.cat.categories[i - 1]
    #         return self._feature_vals.cat.categories[self._bins.size - 2]
    #
    #     return test_feature_vals.map(f, na_action='ignore')
    #
    #
    # info_val_predictiveness_levels : list of 2-tuples, default None
    # Cutoffs for classifying the information value of the feature.
    # It should be a list of 2-tuples (upper_cutoff_value, description).
    # The largest cutoff should be greater than 1 in order to capture all instances.
    #
    # info_val_predictiveness_levels = None
    #
    #     if self.info_val_predictiveness_levels is None:
    #         self.info_val_predictiveness_levels = [(0.1, 'weak'), (0.3, 'ok'), (1.1, 'good')]
    #
    #     # Make sure that info_val_predictiveness_levels is in a convenient format.
    #     # Order the cutoffs in increasing order.
    #     cutoff_dict = {upper_cutoff: (upper_cutoff, description)
    #                    for upper_cutoff, description in self.info_val_predictiveness_levels}
    #     cutoff_sorted_list = [cutoff_dict[upper_cutoff]
    #                           for upper_cutoff in sorted(cutoff_dict.keys())]
    #
    #     # Make sure that the final cutoff is greater than 1.
    #     last_cutoff, last_description = cutoff_sorted_list[-1]
    #     if last_cutoff <= 1:
    #         cutoff_sorted_list[-1] = (1.1, last_description)
