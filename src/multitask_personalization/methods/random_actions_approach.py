"""An approach that takes random actions."""

from multitask_personalization.methods.approach import BaseApproach, _ActType, _ObsType


class RandomActionsApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that samples random actions."""

    def _get_action(self) -> _ActType:
        return self._action_space.sample()
