<<<<<<< HEAD
from .async_pipeline import get_user_config, AsyncPipeline
=======
from .async_pipeline import AsyncPipeline
from .new_async_pipeline1 import NewAsyncPipeline1
from .new_async_pipeline2 import NewAsyncPipeline2
from .new_two_stage_pipeline import newTwoStagePipeline
from .sync_two_stage_pipeline import SyncTwoStagePipeline
from .two_stage_pipeline import TwoStagePipeline
>>>>>>> 8d598b424 (Add text_detection_demo and two_stage_pipeline)

__all__ = [
    'get_user_config',
    'AsyncPipeline',
    'NewAsyncPipeline1',
    'NewAsyncPipeline2',
    'newTwoStagePipeline',
    'SyncTwoStagePipeline',
    'TwoStagePipeline',
]
