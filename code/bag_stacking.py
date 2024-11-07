from sklearn.model_selection import GroupKFold

class BagStacking:
    def __init__(self, base_models, meta_model, n_splits=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
     #   kfold = GroupKFold(n_splits=self.n_splits)#, shuffle=True, random_state=156)
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y, groups):
                train_index = pd.Series(train_index).sample(n=n_samples,random_state=42+i).values
             #   holdout_index = pd.Series(holdout_index).sample(n=4000,random_state=42).values
                instance = clone(model)
                self.base_models_[i].append(instance)

                X_train = X[train_index]
                X_val = X[holdout_index]
                y_train = y[train_index]

                instance.fit(
                    X_train, 
                    y_train, 
                    eval_set=[(X_val, y[holdout_index])], 
                    early_stopping_rounds=3, 
                    verbose=False
                )
                y_pred = instance.predict_proba(X_val)[:, 1]
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            np.median(np.column_stack([model.predict_proba(X)[:, 1] for model in base_models]), axis=1)
            for base_models in self.base_models_ 
        ])
        return self.meta_model_.predict_proba(meta_features)[:, 1]



# Example usage of BagStacking
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Generate a toy dataset
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

    # Instantiate base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('lr', LogisticRegression())
    ]

    # Instantiate the BagStacking classifier
    bag_stacking = BagStacking(base_models=base_models, meta_model=LogisticRegression(), n_jobs=-1)

    # Fit the model
    bag_stacking.fit(X, y, n_samples=5000)
    
