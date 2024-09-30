class EarlyStopping:
    def __init__(self, patience=5, delta=0):

        self.patience = patience
        self.delta = delta
        self.best_accuracy = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, accuracy):
        if self.best_accuracy is None:
            self.best_accuracy = accuracy
        elif accuracy < self.best_accuracy + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_accuracy = accuracy
            self.counter = 0

