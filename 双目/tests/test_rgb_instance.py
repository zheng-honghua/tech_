import numpy as np
from sorting_vision.rgb_instance import recover_rgb_masks


def test_visible_surface_recovered_without_mutating_depth_mask():
    image = np.full((80, 100, 3), 220, np.uint8)
    image[10:65, 20:80] = (180, 120, 10)
    mask = np.zeros((80, 100), np.uint8)
    mask[15:60, 25:75] = 255
    mask[30:40, 35:45] = 0
    original = mask.copy()
    roi = np.ones(mask.shape, np.uint8) * 255
    result = recover_rgb_masks(image, [mask], roi)[0]
    assert result[35, 40] == 255
    assert result[11, 21] == 255
    assert result[0, 0] == 0
    np.testing.assert_array_equal(mask, original)


def test_touching_rgb_component_with_two_owners_is_not_merged():
    image = np.full((60, 100, 3), (180, 120, 10), np.uint8)
    first = np.zeros((60, 100), np.uint8)
    second = first.copy()
    first[5:55, 5:45] = 255
    second[5:55, 55:95] = 255
    result = recover_rgb_masks(image, [first, second], np.ones_like(first) * 255)
    np.testing.assert_array_equal(result[0], first)
    np.testing.assert_array_equal(result[1], second)
