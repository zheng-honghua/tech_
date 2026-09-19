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


def test_small_depth_face_recovers_complete_local_rgb_silhouette():
    image = np.full((100, 120, 3), 220, np.uint8)
    image[20:80, 25:85] = (10, 40, 200)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[40:65, 45:70] = 255
    result = recover_rgb_masks(image, [mask], np.ones_like(mask) * 255)[0]
    assert np.count_nonzero(result) == 3600
    assert np.count_nonzero(mask) == 625


def test_distant_connected_rgb_component_not_attached_to_depth():
    image = np.full((120, 300, 3), 220, np.uint8)
    image[35:85, 20:70] = (10, 40, 200)
    image[35:85, 220:270] = (10, 40, 200)
    image[58:62, 70:220] = (10, 40, 200)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[40:80, 25:65] = 255
    result = recover_rgb_masks(image, [mask], np.ones_like(mask) * 255)[0]
    np.testing.assert_array_equal(result, mask)
