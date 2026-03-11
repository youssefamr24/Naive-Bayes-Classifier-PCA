import numpy as np

class PCA_Scratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        # Calculate the mean
        self.mean = np.mean(X, axis = 0)

        # Center the data
        X_centered = X - self.mean

        # Calculate the Covariance Matrix
        covMatrix = np.cov(X_centered, rowvar = False)

        # Finding Eigenvalues & Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covMatrix)

        # Sort components by eigencalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        self.explained_variance_ratio = eigenvalues[sorted_indices] / np.sum(eigenvalues)

        # Store the top components
        # The slicing starts from the top until the number of components I specified
        self.components = sorted_eigenvectors[:, :self.n_components]
    
    # Project data on the new components
    def transform(self,X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    


class NaiveBayes_Scratch:
    def __init__(self, model_type='gaussian'):
        self.model_type = model_type
        self.priors = {}
        self.parameters = {}
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]

            # Calculate the Prior
            self.priors[c] = X_c.shape[0] / n_samples

            # Calculate Parameters for Categorical & Gaussian
            if self.model_type == 'gaussian':
                self.parameters[c] = {
                    "mean": X_c.mean(axis=0),
                    "var": X_c.var(axis=0) + 1e-9
                }
            else:
                self.parameters[c] = {
                    "probs": [(X_c[:, i] == 1).mean() for i in range(n_features)]
                }

    def predict(self, X):
        # Apply the Naive Bayes classification rule
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(self._get_likelihood(c, x))
            posteriors.append(prior + likelihood)
            
        return self.classes[np.argmax(posteriors)]

    def _get_likelihood(self, class_idx, x):
        if self.model_type == 'gaussian':
            mean = self.parameters[class_idx]["mean"]
            var = self.parameters[class_idx]["var"]

            # Gaussian Formula
            numerator = np.exp(-((x - mean) ** 2) / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)    
            return np.log(numerator / denominator)
        
        else:
            # Convert list to numpy array for vectorized math
            probs = np.array(self.parameters[class_idx]["probs"])
            # Bernoulli Likelihood
            return x * np.log(probs + 1e-9) + (1 - x) * np.log(1 - probs + 1e-9)



