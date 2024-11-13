"""An approach that takes random actions."""

from pathlib import Path

from multitask_personalization.methods.approach import BaseApproach, _ActType, _ObsType


class RandomActionsApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that samples random actions."""

    def _get_action(self) -> _ActType:
        return self._action_space.sample()

    def save(self, model_dir: Path) -> None:
        pass

    def load(self, model_dir: Path) -> None:
        pass
