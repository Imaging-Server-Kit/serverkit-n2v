"""
Algorithm server definition.
Documentation: https://github.com/Imaging-Server-Kit/cookiecutter-serverkit
"""

from typing import List, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit

import os
import shutil
from careamics import CAREamist
from careamics.config import create_n2v_configuration


class Parameters(BaseModel):
    """Defines the algorithm parameters"""

    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D, 3D). If 2D, myst be grayscale (not RGB). If 3D, the image is considered a 2D series.",
        json_schema_extra={"widget_type": "image"},
    )

    epochs: int = Field(
        default=10,
        title="Epochs",
        description="Number of epochs for training",
        ge=1,  # Greater or equal to
        le=1000,  # Lower or equal to
        json_schema_extra={
            "widget_type": "int",
            "step": 1,  # The incremental step to use in the widget (only applicable to numbers)
        },
    )

    patch_size: int = Field(
        default=16,
        title="Patch size",
        description="Square patch size in pixels. Must be a power of two (16, 32, 64...).",
        ge=3,  # Greater or equal to
        le=1024,  # Lower or equal to
        json_schema_extra={
            "widget_type": "int",
            "step": 1,  # The incremental step to use in the widget (only applicable to numbers)
        },
    )

    batch_size: int = Field(
        default=4,
        title="Batch size",
        description="Batch size for training",
        ge=1,  # Greater or equal to
        le=1024,  # Lower or equal to
        json_schema_extra={
            "widget_type": "int",
            "step": 1,  # The incremental step to use in the widget (only applicable to numbers)
        },
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array


class N2VServer(serverkit.AlgorithmServer):
    def __init__(
        self,
        algorithm_name: str = "n2v",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self, image: np.ndarray, epochs: int, patch_size: int, batch_size: int, **kwargs
    ) -> List[tuple]:
        """Runs the algorithm."""
        image_ndim = len(image.shape)

        if image.ndim == 2:
            # Consider it a single image XY type
            config = create_n2v_configuration(
                experiment_name="foo",
                data_type="array",
                axes="YX",
                patch_size=(patch_size, patch_size),
                batch_size=batch_size,
                num_epochs=epochs,
            )

        elif image.ndim == 3:
            # Consider it a Sample-YX type
            config = create_n2v_configuration(
                experiment_name="foo",
                data_type="array",
                axes="SYX",
                patch_size=(patch_size, patch_size),
                batch_size=batch_size,
                num_epochs=epochs,
            )

        careamist = CAREamist(source=config)

        careamist.train(train_source=image)

        prediction = careamist.predict(source=image)
        prediction = np.squeeze(np.array(prediction))

        # Not sure how to prevent any logging and checkpoints saving, so for now we just remove the directories created by n2v...
        shutil.rmtree(Path(__file__).parent / "csv_logs")
        shutil.rmtree(Path(__file__).parent / "checkpoints")

        return [(prediction, {"name": "Denoised"}, "image")]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Loads one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = N2VServer()
app = server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
