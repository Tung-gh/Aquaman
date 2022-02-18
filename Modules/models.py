class Model:
    def train(self, inputs, outputs):
        """

        :param inputs:
        :param outputs:
        :return:
        """
        raise NotImplementedError

    def save(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def load(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        raise NotImplementedError
