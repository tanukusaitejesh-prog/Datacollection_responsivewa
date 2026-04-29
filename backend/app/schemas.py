from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CaptureMeta(BaseModel):
    fps_nominal: float = Field(default=60.0, gt=0)
    resolution: tuple[int, int]
    device: str = Field(min_length=1)
    camera_facing: Literal["front", "back"]
    session_id: str | None = None
    subject_name: str | None = None
    subject_id: str | None = None
    action_type: str | None = None
    age: int | None = Field(default=None, ge=1, le=120)
    gender: Literal["male", "female", "other", "prefer_not_to_say"] | None = None


class UploadPayload(BaseModel):
    keypoints: list[list[list[float]]]
    timestamps: list[float]
    meta: CaptureMeta
    face_keypoints: list[list[list[float]]] | None = None
    hand_keypoints: list[list[list[float]]] | None = None

    @model_validator(mode="after")
    def validate_shapes(self) -> "UploadPayload":
        frame_count = len(self.keypoints)
        if frame_count == 0:
            raise ValueError("keypoints must contain at least one frame.")

        if frame_count != len(self.timestamps):
            raise ValueError(
                "Number of keypoint frames must match number of timestamps."
            )

        for frame_idx, frame in enumerate(self.keypoints):
            if len(frame) != 33:
                raise ValueError(
                    f"Frame {frame_idx} does not contain 33 keypoints: {len(frame)}"
                )
            for joint_idx, joint in enumerate(frame):
                if len(joint) != 4:
                    raise ValueError(
                        f"Frame {frame_idx}, joint {joint_idx} does not contain 4 values."
                    )

        if self.meta.resolution[0] <= 0 or self.meta.resolution[1] <= 0:
            raise ValueError("meta.resolution must contain positive width and height.")

        return self
