# -*- coding: utf-8 -*-
from collections import namedtuple
from typing import List, Union, Iterable
import numpy as np

MatchedPair = namedtuple('MatchedPair', 't_box p_box iou score intensity')
FalsePositive = namedtuple('FalsePositive', 'p_box score')
FalseNegative = namedtuple('FalseNegative', 't_box intensity')

class FullMetrics(object):
    MATCHED_PAIR_KEYS = ('matched_t_boxes', 'matched_p_boxes', 'matched_ious', 'matched_scores', 'matched_intensities')
    FP_KEYS = ('fp_boxes', 'fp_scores')
    FN_KEYS = ('fn_boxes', 'missed_intensities')
    NUM_KEYS = ('num_matched_per_image', 'num_fp_per_image', 'num_fn_per_image')
    KEYS = (*MATCHED_PAIR_KEYS, *FP_KEYS, *FN_KEYS, *NUM_KEYS)

    def __init__(self,
                 matched_pairs: List[MatchedPair] = (),
                 false_positives: List[FalsePositive] = (),
                 false_negatives: List[FalseNegative] = (),
                 num_matched: List[int] = (),
                 num_fp: List[int] = (),
                 num_fn: List[int] = (),
                 ):
        self._matched_pairs = list(matched_pairs)
        self._fp = list(false_positives)
        self._fn = list(false_negatives)
        self._num_matched = list(num_matched)
        self._num_fp = list(num_fp)
        self._num_fn = list(num_fn)

    @property
    def num_images(self) -> int:
        return len(self._num_matched)

    @classmethod
    def from_dict(cls, data_dict):
        matched_pairs = [MatchedPair(*d) for d in zip(*[data_dict[key] for key in cls.MATCHED_PAIR_KEYS])]
        false_positives = [FalsePositive(*d) for d in zip(*[data_dict[key] for key in cls.FP_KEYS])]
        false_negatives = [FalseNegative(*d) for d in zip(*[data_dict[key] for key in cls.FN_KEYS])]
        return cls(
            matched_pairs=matched_pairs,
            false_positives=false_positives,
            false_negatives=false_negatives,
            num_matched=list(data_dict['num_matched_per_image']),
            num_fp=list(data_dict['num_fp_per_image']),
            num_fn=list(data_dict['num_fn_per_image']),
        )

    # Properties für einfachen Zugriff
    @property
    def matched_ious(self) -> np.ndarray:
        return np.array([pair.iou for pair in self._matched_pairs])

    @property
    def num_matched_per_image(self) -> np.ndarray:
        return np.array(self._num_matched)
    
    @property
    def matched_scores(self) -> np.ndarray:
        return np.array([pair.score for pair in self._matched_pairs])

    @property
    def fp_scores(self) -> np.ndarray:
        return np.array([fp.score for fp in self._fp])
        
    @property
    def matched_intensities(self) -> np.ndarray:
        return np.array([pair.intensity for pair in self._matched_pairs])

    @property
    def missed_intensities(self) -> np.ndarray:
        return np.array([fn.intensity for fn in self._fn])
    
    @property
    def false_negatives(self):
        return list(self._fn)

    def append(self, other: 'FullMetrics'):
        self._matched_pairs += other._matched_pairs
        self._fp += other._fp
        self._fn += other._fn
        self._num_matched += other._num_matched
        self._num_fp += other._num_fp
        self._num_fn += other._num_fn

    # WICHTIG: Hier nutzen wir positionale Argumente, um Namenskonflikte zu vermeiden
    def __add__(self, other):
        if not isinstance(other, FullMetrics):
            return NotImplemented
        return FullMetrics(
            self._matched_pairs + other._matched_pairs,
            self._fp + other._fp,
            self._fn + other._fn,
            self._num_matched + other._num_matched,
            self._num_fp + other._num_fp,
            self._num_fn + other._num_fn,
        )

    def __iadd__(self, other):
        if not isinstance(other, FullMetrics):
            return NotImplemented
        self.append(other)
        return self
