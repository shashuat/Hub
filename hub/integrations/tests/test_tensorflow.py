from hub.tests.common import assert_all_samples_have_expected_compression
from hub.constants import UNCOMPRESSED
from hub.api.dataset import Dataset
import numpy as np

from hub.util.check_installation import requires_tensorflow


@requires_tensorflow
def test_tensorflow_with_compression(local_ds: Dataset):
    # TODO: when chunk-wise compression is done, `labels` should be compressed using lz4, so this test needs to be updated
    images = local_ds.create_tensor("images", htype="image")
    labels = local_ds.create_tensor("labels", htype="class_label")

    images.extend(np.ones((16, 100, 100, 3), dtype="uint8"))
    labels.extend(np.ones((16, 1), dtype="int32"))

    # make sure data is appropriately compressed
    assert images.meta.sample_compression == "png"
    assert labels.meta.sample_compression == UNCOMPRESSED
    assert_all_samples_have_expected_compression(images, ["png"] * 16)
    assert_all_samples_have_expected_compression(labels, [UNCOMPRESSED] * 16)

    for batch in local_ds.tensorflow():
        # converting tf Tensors to numpy
        X = batch["images"].numpy()
        T = batch["labels"].numpy()
        assert X.shape == (100, 100, 3)
        assert T.shape == (1,)


@requires_tensorflow
def test_tensorflow_small(local_ds):
    local_ds.create_tensor("image")
    local_ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    local_ds.create_tensor("image2")
    local_ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        # converting tf Tensors to numpy
        np.testing.assert_array_equal(batch["image"].numpy(), i * np.ones((300, 300)))
        np.testing.assert_array_equal(batch["image2"].numpy(), i * np.ones((100, 100)))
    del local_ds


@requires_tensorflow
def test_tensorflow_large(local_ds):
    local_ds.create_tensor("image")
    arr = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ],
    )
    local_ds.image.extend(arr)
    local_ds.create_tensor("classlabel")
    local_ds.classlabel.extend(np.array([i for i in range(10)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        # converting tf Tensors to numpy
        np.testing.assert_array_equal(
            batch["image"].numpy(), (i + 1) * np.ones((4096, 4096))
        )
        np.testing.assert_array_equal(batch["classlabel"].numpy(), (i) * np.ones((1,)))
    del local_ds
