import numpy as np
import matplotlib.pylab as plt

def gen_data(*classes):

    samples, labels = [], []
    for i, (n, mean, sd) in enumerate(classes):
        samples.append(sd * np.random.randn(n) + mean)
        labels.append(np.full(n, i))

    sample = np.hstack(samples)
    label = np.hstack(labels)

    ind = np.random.permutation(len(label))
    return sample[ind], label[ind]


class NaiveBayes():

    def fit(self, x, y):
        self.sample = x
        self.label = y
        self.classes = np.unique(y)

        self.means = {}
        self.sds = {}
        self.priors = {}

        for c in self.classes:
            idx = np.where(self.label == c)
            self.means[c] = np.mean(self.sample[idx])
            self.sds[c] = np.std(self.sample[idx])
            self.priors[c] = len(idx[0]) / len(self.label)

    def _norm_pdf(self, x, mean, sd):
        d = -((x - mean)**2 / (2*sd**2))
        f = (1/(np.sqrt(2*np.pi)*sd)) * np.exp(d)
        return f

    def _posteriors(self, sample):
        return {
            c: self._norm_pdf(sample, self.means[c], self.sds[c]) * self.priors[c]
            for c in self.classes
        }

    def classify(self, sample):
        posts = self._posteriors(sample)
        post_array = np.array([np.sum(np.log(posts[c] + 1e-300), axis=-1) for c in self.classes])
        best_indices = np.argmax(post_array, axis=0)
        return self.classes[best_indices]

    def classify_prob(self, sample):
        posts = self._posteriors(sample)
        total = sum(posts.values())
        return {c: posts[c] / total for c in self.classes}

    def classify_joint(self, sample):
        return self._posteriors(sample)

    def classify_distance(self, sample):
        dists = {c: abs(sample - self.means[c]) for c in self.classes}
        return min(dists, key=lambda c: dists[c])

    def classify_distance_prob(self, sample):
        return {c: abs(sample - self.means[c]) for c in self.classes}