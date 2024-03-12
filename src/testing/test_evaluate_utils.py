import numpy as np
from scipy.ndimage import binary_erosion

def test_get_boundaries():
    # Test case 1: 3D binary mask
    binary_mask_3d = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
    expected_boundary_mask_3d = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
    expected_core_mask_3d = np.array([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ])
    boundary_mask_3d, core_mask_3d = get_boundaries(binary_mask_3d)
    assert np.array_equal(boundary_mask_3d, expected_boundary_mask_3d)
    assert np.array_equal(core_mask_3d, expected_core_mask_3d)

    # Test case 2: 4D binary mask
    binary_mask_4d = np.array([
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ],
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]
    ])
    expected_boundary_mask_4d = np.array([
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ],
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]
    ])
    expected_core_mask_4d = np.array([
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]
    ])
    boundary_mask_4d, core_mask_4d = get_boundaries(binary_mask_4d)
    assert np.array_equal(boundary_mask_4d, expected_boundary_mask_4d)
    assert np.array_equal(core_mask_4d, expected_core_mask_4d)

    print("All test cases passed!")

test_get_boundaries()