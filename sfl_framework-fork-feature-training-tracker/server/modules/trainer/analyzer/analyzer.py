from .CKA_analyzer import CKAAnalyzer
from .conflict_ratio_analyzer import ConflictRatioAnalyzer
from .cosine_similarity_analyzer import CosineSimilarityAnalyzer
from .L2_analyzer import L2Analyzer
from .mean_variance_analyzer import MeanVarianceAnalyzer


class Analyzer:
    def __init__(self):
        self.L2 = L2Analyzer()
        self.CKA = CKAAnalyzer()
        self.conflict = ConflictRatioAnalyzer()
        self.cosine_similarity = CosineSimilarityAnalyzer()
        self.mean_variance = MeanVarianceAnalyzer()


def get_analyzer():
    return Analyzer()
