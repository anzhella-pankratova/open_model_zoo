<<<<<<< HEAD
from .async_pipeline import get_user_config, AsyncPipeline
=======
from .async_pipeline import AsyncPipeline
from .new_async_pipeline import NewAsyncPipeline
from .new_two_stage_pipeline import newTwoStagePipeline
from .sync_two_stage_pipeline import SyncTwoStagePipeline
from .two_stage_pipeline import TwoStagePipeline
>>>>>>> 8d598b424 (Add text_detection_demo and two_stage_pipeline)

__all__ = [
    'get_user_config',
    'AsyncPipeline',
    'NewAsyncPipeline',
    'newTwoStagePipeline',
    'SyncTwoStagePipeline',
    'TwoStagePipeline',
]
