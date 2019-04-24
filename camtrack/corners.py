#! /usr/bin/env python3
# https://github.com/KatyaKos/computer_vision/blob/cbab8bc0aefdc2ce5642879bea0250a9c2ff7eae/camtrack/corners.py
__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


WIN_SIZE = (15, 15)
MAX_LEVEL = 2
MAX_CORNERS = 4000
MIN_DISTANCE = 7
QUALITY_LEVEL = 0.3
BLOCK_SIZE = 7


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    lks = dict(winSize=WIN_SIZE,
               maxLevel=MAX_LEVEL,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    features = dict(maxCorners=MAX_CORNERS,
                    qualityLevel=QUALITY_LEVEL,
                    minDistance=MIN_DISTANCE,
                    blockSize=BLOCK_SIZE)
    corners = cv2.goodFeaturesToTrack(image=image_0,
                                      mask=None,
                                      **features)
    n = len(corners)
    ids = np.array(range(n))
    next_corner = n
    prev_corner = n
    frame_corners = FrameCorners(ids=ids,
                                 points=corners,
                                 sizes=np.full(n, MIN_DISTANCE))

    builder.set_corners_at_frame(0, frame_corners)
    print('Corners:', len(frame_sequence))
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        print(frame, end=' ', flush=True)
        prev_img = np.uint8(image_0 * 255. / image_0.max())
        next_img = np.uint8(image_1 * 255. / image_1.max())
        point_0, _, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, corners, None, **lks)
        point_1, _, _ = cv2.calcOpticalFlowPyrLK(next_img, prev_img, point_0, None, **lks)
        delta = np.abs(corners - point_1).reshape(-1, 2).max(-1)
        pred = delta < 1
        ids = ids[pred]
        corners = point_0[pred]
        n = len(corners)

        if n < MAX_CORNERS:
            mask = np.full(image_1.shape, 255, dtype=np.uint8)
            for coord in corners:
                cv2.circle(mask, (coord[0][0], coord[0][1]), MIN_DISTANCE, 0, -1)

            candidates = cv2.goodFeaturesToTrack(image_1, mask=mask, **features)
            if candidates is None:
                frame_corners = FrameCorners(
                    ids=ids,
                    points=corners,
                    sizes=np.full(n, MIN_DISTANCE),
                )
                builder.set_corners_at_frame(frame, frame_corners)
                image_0 = image_1
                continue

            new_corners = []
            delta_n = 0
            for coord in candidates:
                if n + delta_n < MAX_CORNERS:
                    new_corners.append(coord)
                    next_corner += 1
                    delta_n += 1
            corners = np.concatenate([corners, new_corners])
            n = len(corners)
            ids = np.concatenate([ids, np.array(range(prev_corner, next_corner))])
            prev_corner = next_corner
            frame_corners = FrameCorners(
                ids=ids,
                points=corners,
                sizes=np.full(n, MIN_DISTANCE),
            )
            builder.set_corners_at_frame(frame, frame_corners)
            image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)() # pylint:disable=no-value-for-parameter