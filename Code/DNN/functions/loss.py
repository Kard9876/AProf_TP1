from abc import abstractmethod


class Function:

    @abstractmethod
    def function(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError
