import numpy as np

from .ConfigType import ScheduledSampleConfig


class ScheduledSampler:
    def __init__(self, config: ScheduledSampleConfig) -> None:
        """Initialize the ScheduledSampler with configuration."""
        self.config = config
        self.current_sampling_rate = config.initial_sampling_rate
        self._validate_configs()

    def _validate_configs(self) -> None:
        """Validate scheduled sampling configuration parameters."""
        if self.config.enable_scheduled_sampling:
            assert 0.0 <= self.config.initial_sampling_rate <= 1.0, (
                "Sampling start value must be between 0.0 and 1.0"
            )
            assert self.config.stop_sampling_training_global_step >= 0, (
                "Sampling stop iteration must be non-negative"
            )
            assert self.config.sampling_decay_rate >= 0.0, (
                "Sampling changing rate must be non-negative"
            )

    def update_sampling_rate(self, training_step: int) -> None:
        """Update the sampling rate based on the global training step."""
        if not self.config.enable_scheduled_sampling:
            self.current_sampling_rate = 0.0
            return

        if training_step < self.config.stop_sampling_training_global_step:
            # self.current_sampling_rate = max(
            #     self.current_sampling_rate - self.config.sampling_decay_rate, 0.0
            # )

            progress = training_step / self.config.stop_sampling_training_global_step
            self.current_sampling_rate = self.config.initial_sampling_rate * (
                1.0 - progress * self.config.sampling_decay_rate
            )
        else:
            self.current_sampling_rate = 0.0
        self.current_sampling_rate = np.clip(self.current_sampling_rate, 0.0, 1.0)

    def generate_training_patch_mask(
        self,
        training_step: int,
        batchSize: int,
        targetLength: int,
        imgChannel: int,
        imgHeight: int,
        imgWidth: int,
    ) -> np.ndarray:
        """Generate sampling mask for training."""
        if (
            not self.config.enable_scheduled_sampling
            or self.current_sampling_rate == 0.0
        ):
            return self._create_zero_patch_mask(
                batchSize=batchSize,
                targetLength=targetLength,
                imgChannel=imgChannel,
                imgHeight=imgHeight,
                imgWidth=imgWidth,
            )

        self.update_sampling_rate(training_step=training_step)

        patch_height = imgHeight // self.config.patch_size
        patch_width = imgWidth // self.config.patch_size
        patch_channels = self.config.patch_size**2 * imgChannel

        random_mask: np.ndarray = np.random.random_sample((batchSize, targetLength - 1))
        true_token: np.ndarray = random_mask < self.current_sampling_rate
        ones = np.ones((patch_height, patch_width, patch_channels), dtype=np.float32)
        zeros = np.zeros_like(ones)
        # true_token[..., None, None, None] -> (B, T, 1, 1, 1)
        mask = np.where(true_token[..., None, None, None], ones, zeros)
        # mask -> (batchSize, targetLength - 1, patch_width, patch_width, patch_channels)

        return mask.astype(np.float32)

    def generate_test_patch_mask(
        self,
        batchSize: int,
        targetLength: int,
        imgChannel: int,
        imgHeight: int,
        imgWidth: int,
    ) -> np.ndarray:
        """Generate zero mask for testing/validation."""
        return self._create_zero_patch_mask(
            batchSize=batchSize,
            targetLength=targetLength,
            imgChannel=imgChannel,
            imgHeight=imgHeight,
            imgWidth=imgWidth,
        )

    def _create_zero_patch_mask(
        self,
        batchSize: int,
        targetLength: int,
        imgChannel: int,
        imgHeight: int,
        imgWidth: int,
    ) -> np.ndarray:
        """Create a zero-filled sampling mask."""
        patch_height = imgHeight // self.config.patch_size
        patch_height = imgWidth // self.config.patch_size
        patch_channels = self.config.patch_size**2 * imgChannel
        return np.zeros(
            (batchSize, targetLength - 1, patch_height, patch_height, patch_channels),
            dtype=np.float32,
        )

    def training_mask(
        self,
        training_step: int,
        batchSize: int,
        targetLength: int,
        channel: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Generate sampling mask for training."""
        if (
            not self.config.enable_scheduled_sampling
            or self.current_sampling_rate == 0.0
        ):
            return self._create_zero_patch_mask(
                batchSize=batchSize,
                targetLength=targetLength,
                imgChannel=channel,
                imgHeight=height,
                imgWidth=width,
            )

        self.update_sampling_rate(training_step=training_step)

        random_mask: np.ndarray = np.random.random_sample((batchSize, targetLength - 1))
        true_token: np.ndarray = random_mask < self.current_sampling_rate
        ones = np.ones((channel, height, width), dtype=np.float32)
        zeros = np.zeros_like(ones)
        # true_token[..., None, None, None] -> (B, T, 1, 1, 1)
        mask = np.where(true_token[..., None, None, None], ones, zeros)
        # mask -> (batchSize, targetLength - 1, imgChannel, imgHeight, imgWidth)

        return mask.astype(np.float32)

    def test_mask(
        self,
        batchSize: int,
        targetLength: int,
        channel: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Generate zero mask for testing/validation."""
        return np.zeros(
            (batchSize, targetLength - 1, channel, height, width),
            dtype=np.float32,
        )
