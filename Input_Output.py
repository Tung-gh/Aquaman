# class Document:
#
# class Sentence:
#
# class Token:

class Input:
    def __init__(self, text):
        self.text = text


class Output:
    def __init__(self, aspects, scores):
        self.aspects = aspects
        self.scores = scores

    def __str__(self):
        return str(self.scores)


# class PolarityOutput:
#     def __init__(self, labels, aspects, scores):
#         self.labels = labels
#         self.aspects = aspects
#         self.scores = scores