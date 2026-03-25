import numpy as np

class NaiveBayes():

    def fit(self, x, y):
        self.sample = x
        self.label = y
        self.classes = np.unique(y)

        self.means = {}
        self.sds = {}
        self.priors = {}
        self.eps = 1e-9

        for c in self.classes:
            idx = np.where(self.label == c)
            self.means[c] = np.mean(self.sample[idx],axis=0)
            self.sds[c] = np.maximum(np.std(self.sample[idx],axis=0), self.eps)
            self.priors[c] = len(idx[0]) / len(self.label)
        
        self.theta_ = np.array([self.means[c] for c in self.classes])
        self.var_ = np.array([self.sds[c]**2 for c in self.classes])

    def _norm_pdf(self, x, mean, sd):
        d = -((x - mean)**2 / (2*sd**2))
        f = (1/(np.sqrt(2*np.pi)*sd)) * np.exp(d)
        return f

    def _posteriors(self, sample):
        return {
            c: self._norm_pdf(sample, self.means[c], self.sds[c])
            for c in self.classes
        }

    def classify(self, sample):
        posts = self._posteriors(sample)
        post_array = np.array([
            np.sum(np.log(posts[c] + 1e-300), axis=-1) + np.log(self.priors[c] + 1e-300)
            for c in self.classes
        ])
        best_indices = np.argmax(post_array, axis=0)
        return self.classes[best_indices]

    def classify_match(self, sample):
        posts = self._posteriors(sample)
        post_array = np.array([
            np.prod(posts[c] + 1e-300, axis=-1) * self.priors[c] + 1e-300
            for c in self.classes
        ])
        best_indices = np.argmax(post_array, axis=0)
        return self.classes[best_indices]

    def scores_mul(self, X):
        posts = self._posteriors(X)  
        S = np.array([np.prod(posts[c], axis=-1) * self.priors[c] for c in self.classes])  
        return S

    def scores_log(self, X):
        posts = self._posteriors(X)
        L = np.array([
            np.sum(np.log(posts[c] + 1e-300), axis=-1) + np.log(self.priors[c] + 1e-300)
            for c in self.classes
        ])  
        return L

    def classify_prob(self, sample):   
        posts = self._posteriors(sample)
        post_array = np.array([
            np.sum(np.log(posts[c] + 1e-300), axis=-1) + np.log(self.priors[c] + 1e-300)
            for c in self.classes
        ])
        log_shifted = post_array - np.max(post_array, axis=0, keepdims=True)
        exp_posts = np.exp(log_shifted)
        probs = exp_posts / np.sum(exp_posts, axis=0, keepdims=True)
        return probs.T  

    def classify_joint(self, sample):
        return self._posteriors(sample)